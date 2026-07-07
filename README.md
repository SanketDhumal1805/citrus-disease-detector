
---
title: Citrus Disease Detector
emoji: 🍋
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🍋 Citrus Disease Detector V2

ResNet-50 fusion model (9 classes) wrapped in a Flask app.  
Features: **Grad-CAM · Severity · Treatment Cards · PDF Report · Prediction History · Mobile Camera · Weather Risk Alerts · Multilingual (EN / हिं / मर)**

---

## Quick Start (local dev)

```bash
# 1. Clone / unzip the project
cd citrus_v2

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables (optional — weather feature requires key)
cp .env.example .env
# Edit .env and add your OpenWeatherMap API key

# 5. Run
python app.py
# Open http://localhost:5000
```

---

## Project Structure

```
citrus_v2/
├── app.py                          # Flask app — all logic here
├── fusion_resnet50_v2_seed42.pth   # Trained model weights
├── requirements.txt
├── .env.example
├── templates/
│   ├── index.html                  # Main upload + result UI
│   └── history.html                # Prediction history + chart
├── static/
│   └── uploads/                    # Saved leaf images (auto-created)
└── instance/
    └── predictions.db              # SQLite DB (auto-created on first run)
```

---

## Production Deployment (Gunicorn + Nginx)

```bash
# Gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app

# Nginx proxy config (snippet)
location / {
    proxy_pass http://127.0.0.1:5000;
    proxy_set_header Host $host;
    client_max_body_size 10M;
}
```

> **Note:** `-w 2` keeps model memory manageable. Each worker loads the model once via `@lru_cache`.

---

## Deploy to Render (free tier)

See **[DEPLOY.md](DEPLOY.md)** for the full step-by-step guide (Git LFS for
the model file, `render.yaml` blueprint, env vars, and known free-tier
caveats like the ephemeral disk resetting your prediction history).

Quick version:
1. Push this folder to a GitHub repo (track the `.pth` file with Git LFS).
2. New Web Service → connect repo → Render auto-reads `render.yaml`.
3. Add env var `OPENWEATHER_API_KEY` in the Render dashboard.

---

## Deploy to Railway / Fly.io

Add a `Procfile`:
```
web: gunicorn -w 2 -b 0.0.0.0:$PORT app:app
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Main UI |
| POST | `/predict` | Upload image → JSON result |
| GET | `/report` | Download PDF report |
| GET | `/history` | Prediction history page |
| GET | `/history/data` | Disease distribution JSON |
| GET | `/set_lang/<en\|hi\|mr>` | Switch language |

### `/predict` Response Shape

```json
{
  "disease": "Citrus_Canker",
  "display": "Citrus Canker",
  "confidence": "97.3%",
  "confidence_raw": 0.973,
  "severity": "Severe",
  "sev_class": "danger",
  "cam_image": "<base64 PNG>",
  "top3": [
    {"name": "Citrus Canker", "prob": "97.3%"},
    {"name": "Citrus Blackspot", "prob": "1.8%"},
    {"name": "Anthracnose", "prob": "0.5%"}
  ],
  "treatment": {
    "cause": "...",
    "pesticide": "...",
    "prevention": "...",
    "recovery": "..."
  },
  "timestamp": "2025-06-29 14:32:01",
  "plot_id": "Block-A Row-3"
}
```

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | ResNet-50 |
| Classes | 9 |
| Input size | 224 × 224 |
| Weights file | `fusion_resnet50_v2_seed42.pth` |
| Normalisation | ImageNet mean/std |

**Classes:** Anthracnose · Citrus_Blackspot · Citrus_Canker · Citrus_Greening_HLB · Citrus_Leafminer · Citrus_Nutrient_Deficiency · Healthy_Leaf · Multiple_Diseases · Young_Healthy_Leaf

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENWEATHER_API_KEY` | *(empty)* | Weather risk feature (free tier API key) |
| `DEFAULT_CITY` | `Pune` | City for weather lookup |
| `SECRET_KEY` | hardcoded | Flask session key — **change in production** |

---

## Notes

- The `.pth` file is ~94 MB. Git LFS is recommended if committing to GitHub.
- `static/uploads/` grows over time — add a cron job to prune old images.
- SQLite is fine for a single-server deployment; migrate to PostgreSQL for multi-instance.
