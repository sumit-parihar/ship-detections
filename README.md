# ShipScan — SAR Ship Detection

YOLOv8s ship detection on SAR imagery (JPG / PNG / GeoTIFF).

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to HuggingFace Spaces

1. Create a new Space at huggingface.co → Streamlit → CPU Basic (free)
2. Clone the space:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   ```
3. Copy all files into it:
   ```bash
   cp app.py requirements.txt README.md YOUR_SPACE/
   cp -r .streamlit YOUR_SPACE/
   ```
4. Push:
   ```bash
   cd YOUR_SPACE
   git add .
   git commit -m "deploy shipscan"
   git push
   ```
5. If your HF model repo is private, go to Space Settings → Secrets → add:
   ```
   HF_TOKEN = hf_your_token_here
   ```

## Large Files (> 500 MB)

Browser upload works up to ~2 GB.
For larger files, run the app locally and use the **Local Path** input field.

## Directory Structure

```
shipscan/
├── app.py                  # main application
├── requirements.txt        # dependencies
├── README.md
└── .streamlit/
    └── config.toml         # upload size + theme config
```
