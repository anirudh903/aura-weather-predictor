"""
model_np.py — lightweight, pure-NumPy prediction engine (no scikit-learn/scipy).

Uses the ANALOG method (k-nearest-neighbours): to forecast tomorrow, find the
historical days most similar to today (in season + recent conditions) and look at
what actually happened the day after each of them. This is a real, classic
weather-forecasting technique — and it fits within Vercel's serverless size limit
because it needs only NumPy.

Data functions (geocode / fetch) are imported from weather_model for local testing;
the Vercel function inlines them so it stays self-contained.
"""
from __future__ import annotations

import datetime as dt
import numpy as np

from weather_model import (  # reuse the tested data layer for local eval
    geocode, fetch_history, fetch_recent_and_forecast, _merge_actuals,
    FORECAST_DAYS, RAIN_THRESHOLD_MM,
)

K = 25          # number of analog days to average
ROLL = 7        # rolling-window length


def _season(date: dt.date):
    ang = 2.0 * np.pi * date.timetuple().tm_yday / 365.25
    return np.sin(ang), np.cos(ang)


def _feature(records, i):
    """State vector describing day i (uses days i-6..i)."""
    r = records[i]
    s, c = _season(r["date"])
    lo = max(0, i - ROLL + 1)
    roll_t = np.mean([x["tmean"] for x in records[lo:i + 1]])
    roll_p = np.mean([x["precip"] for x in records[lo:i + 1]])
    return [s, c, r["tmax"], r["tmin"], r["tmean"], r["precip"], roll_t, roll_p]


def _build(records):
    """Feature[i] describes day i; Outcome[i] is day i+1's actual values."""
    F, O = [], []
    for i in range(ROLL, len(records) - 1):
        F.append(_feature(records, i))
        nxt = records[i + 1]
        O.append([nxt["tmax"], nxt["tmin"], nxt["tmean"], nxt["precip"]])
    return np.array(F, float), np.array(O, float)


class Analog:
    """Standardises features, then predicts by averaging the k nearest days."""

    def __init__(self, F, O, k=K):
        self.mean = F.mean(axis=0)
        self.std = F.std(axis=0) + 1e-6
        self.Fz = (F - self.mean) / self.std
        self.O = O
        self.k = min(k, len(F))

    def _neighbours(self, feat):
        q = (np.array(feat, float) - self.mean) / self.std
        d = np.sqrt(((self.Fz - q) ** 2).sum(axis=1))
        return np.argpartition(d, self.k - 1)[: self.k]

    def predict(self, feat):
        idx = self._neighbours(feat)
        out = self.O[idx]                      # k x 4  (tmax,tmin,tmean,precip)
        pred = out.mean(axis=0)
        rain_prob = float((out[:, 3] >= RAIN_THRESHOLD_MM).mean()) * 100.0
        return pred, rain_prob


def train_and_forecast(city: str):
    place = geocode(city)
    lat, lon, tz = place["latitude"], place["longitude"], place["timezone"]
    history = fetch_history(lat, lon, tz)
    live = fetch_recent_and_forecast(lat, lon, tz)
    records = _merge_actuals(history, live["recent_actuals"])
    if len(records) < 400:
        from weather_model import WeatherError
        raise WeatherError("Not enough historical data for this location.")

    F, O = _build(records)

    # time-based split — evaluate on the most recent 20% using only older days
    split = int(len(F) * 0.8)
    model_eval = Analog(F[:split], O[:split])
    pred_te = np.array([model_eval.predict(F[i])[0] for i in range(split, len(F))])
    yte = O[split:]
    mae_tmax = float(np.mean(np.abs(pred_te[:, 0] - yte[:, 0])))
    mae_tmin = float(np.mean(np.abs(pred_te[:, 1] - yte[:, 1])))
    # persistence baseline: tomorrow == today (today's tmax is F[:,2])
    base_tmax = float(np.mean(np.abs(F[split:, 2] - yte[:, 0])))
    base_tmin = float(np.mean(np.abs(F[split:, 3] - yte[:, 1])))
    # rain accuracy
    rain_pred = np.array([model_eval.predict(F[i])[1] for i in range(split, len(F))]) >= 50
    rain_true = yte[:, 3] >= RAIN_THRESHOLD_MM
    rain_acc = float((rain_pred == rain_true).mean() * 100.0)

    # final model on ALL data, then roll forward 7 days
    model = Analog(F, O)
    working = list(records)
    last_date = working[-1]["date"]
    forecast = []
    for step in range(1, FORECAST_DAYS + 1):
        feat = _feature(working, len(working) - 1)
        (tmax, tmin, tmean, precip), rain_prob = model.predict(feat)
        precip = max(0.0, precip)
        target = last_date + dt.timedelta(days=step)
        forecast.append({
            "date": target.isoformat(),
            "tmax": round(float(tmax), 1), "tmin": round(float(tmin), 1),
            "tmean": round(float(tmean), 1), "precip": round(float(precip), 1),
            "rain_prob": round(rain_prob, 0),
            "weather_code": _code(precip, rain_prob),
        })
        working.append({"date": target, "tmax": tmax, "tmin": tmin,
                        "tmean": tmean, "precip": precip, "wind": working[-1]["wind"]})

    def pub(r):
        return {"date": r["date"].isoformat() if isinstance(r["date"], dt.date) else r["date"],
                "tmax": round(r["tmax"], 1), "tmin": round(r["tmin"], 1),
                "tmean": round(r["tmean"], 1), "precip": round(r["precip"], 1),
                "weather_code": r.get("weather_code", 0)}

    return {
        "place": place, "current": live["current"], "current_units": live["current_units"],
        "recent_actuals": [pub(r) for r in records[-14:]],
        "my_forecast": forecast,
        "pro_forecast": [pub(r) for r in live["pro_forecast"]],
        "metrics": {
            "mae_tmax": round(mae_tmax, 2), "mae_tmin": round(mae_tmin, 2),
            "baseline_tmax": round(base_tmax, 2), "baseline_tmin": round(base_tmin, 2),
            "rain_accuracy": round(rain_acc, 1),
            "train_days": len(records), "test_days": len(yte),
            "history_years": 8, "method": "analog",
        },
    }


def _code(precip, rain_prob):
    if precip >= 10 or rain_prob >= 80:
        return 65
    if precip >= RAIN_THRESHOLD_MM or rain_prob >= 55:
        return 61
    if rain_prob >= 35:
        return 3
    if rain_prob >= 20:
        return 2
    return 1
