# MovieSent - Gradio Deployment Guide

This document explains how to run and deploy the `gradio` version of MovieSent.

## Files added
- `gradio_app.py` — main Gradio application (already committed)

## Run locally

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (this project includes `requirements.txt`):

```bash
pip install -r requirements.txt
```

3. Run the Gradio app:

```bash
python gradio_app.py
```

The app will start on port 7860 by default (http://localhost:7860). If port is in use, the script will fail to bind; stop other services or change the `server_port` in `gradio_app.py`.

## Deploying to the cloud

Two easy options:

### 1) Hugging Face Spaces
- Create a new Space and select `Gradio` as the SDK.
- Push your repository to GitHub and connect it to the Space, or push the `gradio_app.py` and `requirements.txt` there.
- Hugging Face will install dependencies and host the app.

Notes:
- If model files are large (>100 MB), use Git LFS.
- For private models, consider a storage or API-based approach.

### 2) Gradio Cloud / self-host
- Gradio provides hosting options; you can also deploy on any server (e.g. VPS) with port forwarding and a reverse proxy.

## Notes about LSTM compatibility

- The LSTM model in `saved_models/` may have been trained/saved with an older Keras version and can fail to load on modern TF/Keras.
- `gradio_app.py` will attempt to load the LSTM and gracefully fall back to Logistic Regression if the LSTM does not load.
- If you need LSTM predictions on cloud, consider re-saving the model with the target TensorFlow version used by the hosting environment or exporting model weights and re-creating architecture in the deployment environment.

## Troubleshooting

- If you see an `ImportError` for `gradio`, ensure `gradio` is installed in the environment.
- If LSTM fails to load, read the error printed at startup (the app prints model load errors to help debugging).

## Updating the app

- Edit `gradio_app.py` locally, commit, and push to GitHub.
- If using Hugging Face Spaces, it will rebuild automatically on push.

---

If you want, I can (A) try to load your LSTM file here to diagnose the exact incompatibility and propose a fix, or (B) re-save a lightweight LSTM-compatible version for you if you can share the training notebook's model construction section.
