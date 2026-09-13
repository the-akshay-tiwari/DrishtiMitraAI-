# DrishtiMitra model training

The bundled IDRiD data contains 375 labelled training images and 80 labelled images whose
names end in `test`. The training script preserves those 80 images as a final held-out test
set and makes its validation split only from the 375 training images.

## Run locally

From the project root in PowerShell:

```powershell
.\.venv\bin\python.exe -m pip install -r backend\requirements.txt
.\.venv\bin\python.exe backend\train.py --epochs 20 --batch-size 16
.\.venv\bin\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

The trained checkpoint is saved to `artifacts/drishtimitra_efficientnet_b0.pt`, with held-out
test metrics at `artifacts/training_metrics.json`. Open `http://127.0.0.1:8000/docs` to try
the inference API.

The React app sends an uploaded fundus image to `POST /predict` when that local server is
running.

## Important limitations

- This is a small, class-imbalanced research dataset. Grade 1 has only 22 total images.
- The metrics are only for the supplied IDRiD-style dataset; they do not establish performance
  on cameras, populations, or image-quality conditions outside it.
- This is not a medical device and must not be used for autonomous diagnosis or treatment.
