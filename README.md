# Aura — ML Weather Predictor

A weather app that makes its **own** forecast. Instead of just showing someone
else's numbers, it downloads ~8 years of a city's weather history, trains a
machine-learning model on it, and uses that model to predict the next 7 days.
It then shows how its home-grown forecast compares to the professional one.

All data comes from **[Open-Meteo](https://open-meteo.com/)** — free, no API key.

## How it works

1. **Geocode** — turn the city name into coordinates.
2. **Download history** — ~8 years of daily weather (temperature, rain, wind).
3. **Train** — a Random Forest learns the local patterns using the season plus
   the last few days of weather (lag features).
4. **Predict** — the model forecasts the next 7 days one day at a time,
   feeding each prediction back in to predict the day after.
5. **Score** — accuracy is measured honestly on recent days the model never saw,
   and compared against a naïve "tomorrow = today" baseline.

## Run it locally

```bash
# from the weather-predictor folder
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Project layout

```
weather-predictor/
├── app.py              # Flask server + /api/predict endpoint
├── weather_model.py    # the prediction engine (data + ML)
├── requirements.txt
├── templates/
│   └── index.html      # the page
└── static/
    ├── style.css       # glassy UI
    └── app.js          # fetch + render + chart
```

## Deploy as a website (Render, free)

This repo is deploy-ready (`Procfile` + `render.yaml`).

1. Create a free account at [render.com](https://render.com) (sign in with GitHub).
2. **New → Web Service**, pick this repository.
3. Render auto-detects the settings from `render.yaml`. Click **Deploy**.
4. In ~2–3 minutes you get a public URL like `https://aura-weather.onrender.com`.

Note: the free tier sleeps after 15 min of inactivity, so the very first visit
after a nap takes ~50s to wake up. Every visit after that is fast.

## Tech

Python · Flask · scikit-learn (Random Forest) · Chart.js · Open-Meteo API · Render
