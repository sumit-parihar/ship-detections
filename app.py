"""
ShipScan — SAR Ship Detection
Loads YOLOv8s best.pt from HuggingFace, runs inference on
JPG / PNG / TIFF images (including 1 GB+ GeoTIFF SAR scenes).
"""

import os, gc, io, time, tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
import rasterio
from rasterio.windows import Window
import torch
from ultralytics.nn.tasks import DetectionModel
from dotenv import load_dotenv
load_dotenv()

# Allow YOLO model loading (PyTorch 2.6 fix)
torch.serialization.add_safe_globals([DetectionModel])

from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────
HF_REPO      = "sumit3142857/ship-detection-yolo"   # ← your HF repo
HF_FILENAME  = "best.pt"
TILE_SIZE    = 640
TILE_OVERLAP = 64
MAX_DISPLAY  = 1500   # max px for overview image shown in browser

st.set_page_config(
    page_title="ShipScan",
    page_icon="🛳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Minimal CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 1rem 0 1.5rem;
}
.mcard {
    background: #0d1625;
    border: 1px solid #1a3050;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.mcard .label {
    font-size: 0.65rem;
    color: #5a7a99;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.mcard .value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
}
.mcard .sub {
    font-size: 0.65rem;
    color: #5a7a99;
    margin-top: 3px;
}

.det-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.det-table th {
    background: #111f35; color: #00c2ff;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #1a3050;
    text-align: left; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.det-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #1a3050; color: #cdd9e8; }
.det-table tr:hover td { background: #111f35; }

.info-box {
    background: #0d1625;
    border: 1px solid #1a3050;
    border-left: 3px solid #00c2ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #cdd9e8;
    margin: 0.5rem 0;
}
.ok   { border-left-color: #00ff9d !important; }
.warn { border-left-color: #ffb800 !important; }
.err  { border-left-color: #ff5050 !important; }

[data-testid="stImage"] > img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        hf_token = os.getenv("HF_TOKEN")
        try:
            hf_token = st.secrets.get("HF_TOKEN", None)
        except Exception:
            pass
        hf_token = hf_token or os.environ.get("HF_TOKEN", None)

        with st.spinner("⏳ Loading model from HuggingFace…"):
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=HF_FILENAME,
                repo_type="model",
                token=hf_token,
                cache_dir="/tmp"
            )
        return YOLO(path)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.info(
            "If your HuggingFace repo is private, add `HF_TOKEN` to "
            "Streamlit secrets (App Settings → Secrets)."
        )
        st.stop()


# ─── SAR / TIFF Utilities ─────────────────────────────────────────────────────

def sar_percentiles(src_path, method="percentile"):
    """Sample tiny blocks to compute normalization range — never OOMs."""
    with rasterio.open(src_path) as src:
        n_bands = min(src.count, 3)
        H, W    = src.height, src.width
        step    = max(1024, H // 4)
        samples = {b: [] for b in range(1, n_bands + 1)}

        for y in range(0, H, step):
            for x in range(0, W, step):
                win = Window(x, y, min(1024, W - x), min(1024, H - y))
                for b in range(1, n_bands + 1):
                    block = src.read(b, window=win).astype(np.float32)
                    samples[b].append(block[::4, ::4].ravel())

        pct = {}
        for b in range(1, n_bands + 1):
            arr = np.concatenate(samples[b])
            if method == "log":
                arr = np.log10(1.0 + np.maximum(arr, 0))
            lo = float(np.percentile(arr, 2))
            hi = float(np.percentile(arr, 98))
            if hi - lo < 1e-6:
                lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-6:
                hi = lo + 1.0
            pct[b] = (lo, hi)
    return pct, n_bands


def norm_tile(tile, lo, hi, method="percentile"):
    a = tile.astype(np.float32)
    if method == "log":
        np.maximum(a, 0, out=a)
        np.log10(1.0 + a, out=a)
    a = (a - lo) / (hi - lo)
    np.clip(a, 0, 1, out=a)
    return (a * 255).astype(np.uint8)


def tiles(H, W, size=640, overlap=64):
    step = size - overlap
    for y in range(0, H, step):
        for x in range(0, W, step):
            yield y, x, min(y + size, H), min(x + size, W)


def nms(dets, iou_thr=0.5):
    if not dets:
        return []
    boxes  = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
    scores = np.array([d["conf"] for d in dets])
    areas  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return [dets[i] for i in keep]


def predict_tiff(model, src_path, pct, n_bands, method,
                 tile_sz, overlap, conf_thr, iou_thr, imgsz, pbar):
    with rasterio.open(src_path) as src:
        H, W    = src.height, src.width
        all_tiles = list(tiles(H, W, tile_sz, overlap))
        dets    = []

        for t_i, (y0, x0, y1, x1) in enumerate(all_tiles):
            win = Window(x0, y0, x1 - x0, y1 - y0)

            if n_bands >= 3:
                ch = []
                for b in range(1, 4):
                    lo, hi = pct[b]
                    ch.append(norm_tile(src.read(b, window=win), lo, hi, method))
                rgb = np.stack(ch, axis=-1); del ch
            else:
                lo, hi = pct[1]
                g   = norm_tile(src.read(1, window=win), lo, hi, method)
                rgb = np.stack([g, g, g], axis=-1); del g

            if rgb.std() < 3:          # skip blank tiles
                del rgb
                pbar.progress((t_i + 1) / len(all_tiles))
                continue

            res = model.predict(rgb, conf=conf_thr, iou=iou_thr,
                                imgsz=imgsz, verbose=False)
            for box in res[0].boxes:
                x1b, y1b, x2b, y2b = box.xyxy[0].tolist()
                dets.append({"x1": x1b + x0, "y1": y1b + y0,
                              "x2": x2b + x0, "y2": y2b + y0,
                              "conf": float(box.conf[0])})
            del rgb; gc.collect()
            pbar.progress((t_i + 1) / len(all_tiles))

    return nms(dets, iou_thr), len(all_tiles)


def build_overview(src_path, pct, n_bands, method, max_px=MAX_DISPLAY):
    with rasterio.open(src_path) as src:
        H, W   = src.height, src.width
        ratio  = min(max_px / max(W, H, 1), 1.0)
        ow, oh = max(1, int(W * ratio)), max(1, int(H * ratio))
        canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
        step   = 2048

        for y in range(0, H, step):
            for x in range(0, W, step):
                win  = Window(x, y, min(step, W - x), min(step, H - y))
                dw   = max(1, int(win.width  * ratio))
                dh   = max(1, int(win.height * ratio))
                dx, dy = int(x * ratio), int(y * ratio)

                ch = []
                for b in range(1, n_bands + 1):
                    lo, hi = pct[b]
                    nm = norm_tile(src.read(b, window=win), lo, hi, method)
                    ch.append(np.array(
                        Image.fromarray(np.squeeze(nm))
                              .resize((dw, dh), Image.BILINEAR)
                    ))

                if n_bands == 1:
                    tile_rgb = np.stack([ch[0], ch[0], ch[0]], axis=-1)
                elif n_bands == 2:
                    tile_rgb = np.stack([ch[0], ch[1], ch[0]], axis=-1)
                else:
                    tile_rgb = np.stack([np.squeeze(c) for c in ch[:3]], axis=-1)

                ah, aw = tile_rgb.shape[:2]
                ey = min(dy + ah, oh); ex = min(dx + aw, ow)
                canvas[dy:ey, dx:ex] = tile_rgb[:ey - dy, :ex - dx]
                del tile_rgb, ch; gc.collect()

    return canvas, ratio


def draw_boxes(img_arr, dets, show_lbl=True, show_conf=True):
    img  = Image.fromarray(img_arr).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    for i, d in enumerate(dets):
        conf  = d["conf"]
        color = (0, 255, 157) if conf > 0.75 else \
                (255, 184, 0) if conf > 0.50 else (255, 80, 80)
        draw.rectangle([d["x1"], d["y1"], d["x2"], d["y2"]],
                       outline=color, width=2)
        if show_lbl or show_conf:
            parts = []
            if show_lbl:  parts.append(f"Ship #{i+1}")
            if show_conf: parts.append(f"{conf:.0%}")
            label = " ".join(parts)
            bb = draw.textbbox((d["x1"], d["y1"]), label, font=font)
            draw.rectangle([bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2], fill=color)
            draw.text((d["x1"], d["y1"]), label, fill=(0, 0, 0), font=font)

    return np.array(img)


def stream_to_disk(f_obj, ext):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix="." + ext)
    f_obj.seek(0)
    chunk = 32 * 1024 * 1024
    written = 0
    while True:
        buf = f_obj.read(chunk)
        if not buf: break
        tf.write(buf); written += len(buf)
    tf.flush(); tf.close()
    return tf.name, written / 1024**2


def safe_delete(path, retries=3):
    for _ in range(retries):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
            return
        except Exception:
            time.sleep(0.4)


def show_metrics(n_ships, avg_conf, max_conf, elapsed_s, w, h, bands, size_mb, is_tiff, n_tiles=None, imgsz=640):
    time_str  = f"{elapsed_s:.1f}s"
    time_sub  = f"{n_tiles} tiles @ {imgsz}px" if is_tiff and n_tiles else f"@ {imgsz}px"
    band_info = f"{bands} band{'s' if bands > 1 else ''} · SAR" if is_tiff else "optical"
    st.markdown(f"""
    <div class="metric-row">
        <div class="mcard">
            <div class="label">Ships Detected</div>
            <div class="value">{n_ships}</div>
            <div class="sub">conf ≥ {conf_thr:.2f}</div>
        </div>
        <div class="mcard">
            <div class="label">Avg Confidence</div>
            <div class="value">{avg_conf:.0%}</div>
            <div class="sub">max {max_conf:.0%}</div>
        </div>
        <div class="mcard">
            <div class="label">Inference Time</div>
            <div class="value">{time_str}</div>
            <div class="sub">{time_sub}</div>
        </div>
        <div class="mcard">
            <div class="label">Image</div>
            <div class="value" style="font-size:1.2rem">{w}×{h}</div>
            <div class="sub">{band_info} · {size_mb:.0f} MB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_table(dets):
    rows = ""
    for i, d in enumerate(dets):
        conf  = d["conf"]
        bw    = int(d["x2"] - d["x1"])
        bh    = int(d["y2"] - d["y1"])
        cert  = "High" if conf > 0.75 else "Medium" if conf > 0.50 else "Low"
        bar   = int(conf * 100)
        rows += f"""<tr>
            <td>#{i+1:02d}</td>
            <td>
              <div style="display:flex;align-items:center;gap:6px">
                <div style="width:{bar}px;height:6px;border-radius:3px;
                     background:linear-gradient(90deg,#00ff9d,#00c2ff)"></div>
                <span>{conf:.1%}</span>
              </div>
            </td>
            <td>{bw} × {bh} px</td>
            <td>{int(d['x1'])}, {int(d['y1'])}</td>
            <td>{cert}</td>
        </tr>"""
    st.markdown(f"""
    <table class="det-table">
      <thead><tr>
        <th>ID</th><th>Confidence</th><th>Box Size</th>
        <th>Origin (x,y)</th><th>Certainty</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)


def export_buttons(dets, annotated, fname):
    df = pd.DataFrame([{
        "ID": i+1, "Confidence": round(d["conf"], 4),
        "x1": int(d["x1"]), "y1": int(d["y1"]),
        "x2": int(d["x2"]), "y2": int(d["y2"]),
        "Width": int(d["x2"]-d["x1"]), "Height": int(d["y2"]-d["y1"]),
    } for i, d in enumerate(dets)])

    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        st.download_button("⬇ Export CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"detections_{fname}.csv", mime="text/csv")
    with c2:
        st.download_button("⬇ Save Image",
            data=buf.getvalue(),
            file_name=f"annotated_{fname}.png", mime="image/png")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("**Detection**")
    conf_thr = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
    iou_thr  = st.slider("IoU Threshold (NMS)",  0.10, 0.95, 0.45, 0.05)
    imgsz    = st.select_slider("Inference Resolution",
                                options=[320, 512, 640, 768, 1024], value=640)

    st.markdown("---")
    st.markdown("**SAR / TIFF**")
    tile_sz  = st.select_slider("Tile Size (px)",
                                options=[320, 512, 640, 800, 1024], value=640)
    overlap  = st.slider("Tile Overlap (px)", 0, 256, 64, 16)
    norm_method = st.selectbox("Normalization", ["percentile", "log"],
                               help="Use 'log' for high dynamic range SAR scenes.")

    st.markdown("---")
    st.markdown("**Display**")
    show_lbl      = st.toggle("Show Labels",         value=True)
    show_conf_box = st.toggle("Show Confidence",     value=True)
    show_orig     = st.toggle("Show Original Image", value=False)

    st.markdown("---")
    st.caption(f"Model: `{HF_REPO}`")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🛳️ ShipScan")
st.markdown("**SAR Ship Detection** — YOLOv8s · Upload JPG / PNG / TIFF")
st.markdown("---")

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown("### Upload Image(s)")
uploaded = st.file_uploader(
    "Supports JPG, PNG, GeoTIFF (up to 2 GB via browser upload)",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
)

if uploaded:
    for f in uploaded:
        mb = f.size / 1024**2
        if mb > 500:
            st.warning(
                f"⚠️ **{f.name}** is {mb:.0f} MB. "
                "If upload fails, use the local path input below."
            )

files = list(uploaded or [])


# ─── Inference Loop ───────────────────────────────────────────────────────────
if not files:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#5a7a99">
        <div style="font-size:3rem">🌊</div>
        <p style="font-size:1.1rem;margin-top:1rem">No image loaded yet.</p>
        <p style="font-size:0.85rem">Upload a file above or enter a local path.</p>
    </div>""", unsafe_allow_html=True)
else:
    model = load_model()
    st.success("✅ Model ready.")

    for idx, fobj in enumerate(files):
        fname    = fobj.name
        ext      = fname.rsplit(".", 1)[-1].lower()
        is_tiff  = ext in ("tif", "tiff")
        is_local = False
        size_mb  = getattr(fobj, "size", 0) / 1024**2

        if len(files) > 1:
            st.markdown(f"#### File {idx+1}/{len(files)}: `{fname}`")

        # ── TIFF ──────────────────────────────────────────────────────────────
        if is_tiff:
            tmp_path      = None
            need_cleanup  = False
            try:
                if is_local:
                    tmp_path = fobj.path
                else:
                    with st.spinner(f"Streaming {fname} ({size_mb:.0f} MB) to disk…"):
                        tmp_path, written = stream_to_disk(fobj, ext)
                        need_cleanup = True
                    st.markdown(
                        f'<div class="info-box ok">✅ Streamed {written:.0f} MB to disk.</div>',
                        unsafe_allow_html=True)

                orig_w, orig_h, bands, _ = (
                    lambda s: (s.width, s.height, s.count, s.transform)
                )(rasterio.open(tmp_path))

                with st.spinner("Computing SAR normalization…"):
                    pct, n_bands = sar_percentiles(tmp_path, norm_method)

                n_tiles_est = len(list(tiles(orig_h, orig_w, tile_sz, overlap)))
                st.markdown(
                    f'<div class="info-box">'
                    f'🗂️ {orig_w}×{orig_h} px · {bands} band(s) · '
                    f'~{n_tiles_est} tiles ({tile_sz}px, overlap {overlap}px)'
                    f'</div>',
                    unsafe_allow_html=True)

                pbar = st.progress(0, text="Running detection on tiles…")
                t0   = time.perf_counter()
                dets, n_tiles = predict_tiff(
                    model, tmp_path, pct, n_bands, norm_method,
                    tile_sz, overlap, conf_thr, iou_thr, imgsz, pbar
                )
                elapsed = time.perf_counter() - t0
                pbar.empty()
                # --- FILTER SMALL NOISE BOXES ---
                dets = [
                    d for d in dets
                    if (d["x2"] - d["x1"]) > 12 and (d["y2"] - d["y1"]) > 12
                ]
                # --- LIMIT MAX DETECTIONS ---
                # dets = dets[:50]
                n_ships  = len(dets)
                confs    = [d["conf"] for d in dets]
                avg_conf = float(np.mean(confs)) if confs else 0.0
                max_conf = float(np.max(confs))  if confs else 0.0

                show_metrics(n_ships, avg_conf, max_conf, elapsed,
                             orig_w, orig_h, bands, size_mb,
                             True, n_tiles, imgsz)

                with st.spinner("Building display overview…"):
                    overview, ratio = build_overview(
                        tmp_path, pct, n_bands, norm_method)

                scaled = [{
                    "x1": d["x1"]*ratio, "y1": d["y1"]*ratio,
                    "x2": d["x2"]*ratio, "y2": d["y2"]*ratio,
                    "conf": d["conf"],
                } for d in dets]
                annotated = draw_boxes(overview, scaled, show_lbl, show_conf_box)

                cols = st.columns(2 if show_orig else 1)
                if show_orig:
                    with cols[0]:
                        st.markdown("**Original (normalized)**")
                        st.image(overview, use_column_width=True)
                    with cols[1]:
                        st.markdown("**Detections**")
                        st.image(annotated, use_column_width=True)
                else:
                    st.image(annotated, use_column_width=True,
                             caption=f"Detections — {n_ships} ship(s) found")

                if n_ships > 0:
                    st.markdown("#### Detection Log")
                    show_table(dets)
                    st.markdown("<br>", unsafe_allow_html=True)
                    export_buttons(dets, annotated, fname)
                else:
                    st.markdown(
                        '<div class="info-box warn">No ships detected. '
                        'Try lowering the confidence threshold or switching '
                        'normalization to "log" in the sidebar.</div>',
                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error processing {fname}: {e}")
            finally:
                if need_cleanup:
                    safe_delete(tmp_path)

        # ── JPG / PNG ─────────────────────────────────────────────────────────
        else:
            try:
                if is_local:
                    img = Image.open(fobj.path).convert("RGB")
                else:
                    img = Image.open(fobj).convert("RGB")
                arr = np.array(img)
            except Exception as e:
                st.error(f"❌ Cannot open {fname}: {e}")
                continue

            with st.spinner("Running detection…"):
                t0  = time.perf_counter()
                res = model.predict(arr, conf=conf_thr, iou=iou_thr,
                                    imgsz=imgsz, verbose=False)
                elapsed = time.perf_counter() - t0

            boxes    = res[0].boxes
            n_ships  = len(boxes)
            confs    = [float(b.conf[0]) for b in boxes]
            avg_conf = float(np.mean(confs)) if confs else 0.0
            max_conf = float(np.max(confs))  if confs else 0.0

            show_metrics(n_ships, avg_conf, max_conf, elapsed,
                         arr.shape[1], arr.shape[0], 3, size_mb,
                         False, None, imgsz)

            annotated = res[0].plot(labels=show_lbl, conf=show_conf_box)
            if annotated.shape[-1] == 3:
                annotated = annotated[..., ::-1]   # BGR → RGB

            cols = st.columns(2 if show_orig else 1)
            if show_orig:
                with cols[0]:
                    st.markdown("**Original**")
                    st.image(img, use_column_width=True)
                with cols[1]:
                    st.markdown("**Detections**")
                    st.image(annotated, use_column_width=True)
            else:
                st.image(annotated, use_column_width=True,
                         caption=f"Detections — {n_ships} ship(s) found")

            if n_ships > 0:
                dets = [{
                    "x1": float(b.xyxy[0][0]), "y1": float(b.xyxy[0][1]),
                    "x2": float(b.xyxy[0][2]), "y2": float(b.xyxy[0][3]),
                    "conf": float(b.conf[0]),
                } for b in boxes]
                st.markdown("#### Detection Log")
                show_table(dets)
                st.markdown("<br>", unsafe_allow_html=True)
                export_buttons(dets, annotated, fname)
            else:
                st.markdown(
                    '<div class="info-box warn">No ships detected. '
                    'Try lowering the confidence threshold in the sidebar.</div>',
                    unsafe_allow_html=True)

        if idx < len(files) - 1:
            st.markdown("---")
