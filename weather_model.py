"""
weather_model.py
=================
The prediction engine for the weather app.

What it does, in plain terms:
  1. Turns a city name into coordinates (Open-Meteo geocoding, free, no key).
  2. Downloads ~8 years of *daily* historical weather for that spot.
  3. Trains a machine-learning model to learn the local weather patterns.
  4. Uses that model to make its OWN 7-day forecast (this is the "original
     prediction" part -- we are not copying anyone's forecast).
  5. Also grabs the professional Open-Meteo forecast so we can show, honestly,
     how our home-grown model stacks up against it.

Everything here is free. No API keys, no billing.

Resilience: free weather APIs rate-limit shared cloud IPs (429). Every request
retries with backoff, and if the *live* feed stays throttled we degrade
gracefully -- the 7-day model forecast (built from the archive, which is far more
lenient) is still delivered instead of failing outright.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------------------------------- #
# Open-Meteo endpoints (all free, no API key required)
# --------------------------------------------------------------------------- #
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
]

HISTORY_YEARS = 8
FORECAST_DAYS = 7
RAIN_THRESHOLD_MM = 1.0

LAGS = [1, 2, 3, 7]
MIN_LOOKBACK = max(LAGS)

HTTP_TIMEOUT = 30
RETRIES = 4  # total attempts per request before giving up


class WeatherError(Exception):
    """Friendly, user-facing error (e.g. city not found, API down)."""


# --------------------------------------------------------------------------- #
# Networking with retry/backoff (handles the 429s from shared cloud IPs)
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict, what: str) -> dict:
    last = "unknown error"
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 0.6 * (2 ** attempt)
                time.sleep(min(wait, 5.0))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(0.6 * (2 ** attempt))
    raise WeatherError(f"Could not download {what}: {last}")


# --------------------------------------------------------------------------- #
# 1. Geocoding: city name -> coordinates
# --------------------------------------------------------------------------- #
def geocode(city: str) -> dict[str, Any]:
    city = (city or "").strip()
    if not city:
        raise WeatherError("Please type a city name.")
    data = _get_json(
        GEO_URL,
        {"name": city, "count": 1, "language": "en", "format": "json"},
        "the location service",
    )
    results = data.get("results")
    if not results:
        raise WeatherError(f'Could not find a city called "{city}". Check the spelling?')

    top = results[0]
    parts = [top["name"]] + [top[k] for k in ("admin1", "country") if top.get(k)]
    return {
        "name": top["name"],
        "label": ", ".join(parts),
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "country": top.get("country", ""),
        "timezone": top.get("timezone", "auto"),
    }


# --------------------------------------------------------------------------- #
# 2. Data download
# --------------------------------------------------------------------------- #
def _daily_records(daily: dict[str, list]) -> list[dict[str, Any]]:
    times = daily.get("time", [])
    records = []
    for i, day in enumerate(times):
        def val(key):
            seq = daily.get(key) or []
            return seq[i] if i < len(seq) else None

        tmax, tmin, tmean = val("temperature_2m_max"), val("temperature_2m_min"), val("temperature_2m_mean")
        if tmean is None and tmax is not None and tmin is not None:
            tmean = (tmax + tmin) / 2.0
        if tmax is None or tmin is None or tmean is None:
            continue
        precip, wind = val("precipitation_sum"), val("wind_speed_10m_max")
        records.append(
            {
                "date": dt.date.fromisoformat(day),
                "tmax": float(tmax),
                "tmin": float(tmin),
                "tmean": float(tmean),
                "precip": float(precip) if precip is not None else 0.0,
                "wind": float(wind) if wind is not None else 0.0,
            }
        )
    return records


def fetch_history(lat: float, lon: float, tz: str) -> list[dict[str, Any]]:
    """~8 years of settled historical daily data from the archive."""
    end = dt.date.today() - dt.timedelta(days=6)  # archive lags ~5 days
    start = end - dt.timedelta(days=365 * HISTORY_YEARS)
    data = _get_json(
        ARCHIVE_URL,
        {
            "latitude": lat, "longitude": lon,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS), "timezone": tz,
        },
        "historical weather",
    )
    return _daily_records(data.get("daily", {}))


def fetch_recent_and_forecast(lat: float, lon: float, tz: str) -> dict[str, Any]:
    """Current conditions + last ~15 days of actuals + Open-Meteo's own forecast.
    (This hits the forecast endpoint, which is the one that gets rate-limited.)"""
    data = _get_json(
        FORECAST_URL,
        {
            "latitude": lat, "longitude": lon,
            "daily": ",".join(DAILY_VARS + ["weather_code"]),
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
            "past_days": 15, "forecast_days": FORECAST_DAYS, "timezone": tz,
        },
        "current conditions",
    )
    today = dt.date.today()
    daily = data.get("daily", {})
    codes, times = daily.get("weather_code", []) or [], daily.get("time", [])
    recent, pro = [], []
    for rec in _daily_records(daily):
        try:
            idx = times.index(rec["date"].isoformat())
            rec["weather_code"] = int(codes[idx]) if idx < len(codes) and codes[idx] is not None else 0
        except (ValueError, TypeError):
            rec["weather_code"] = 0
        (recent if rec["date"] <= today else pro).append(rec)
    return {
        "current": data.get("current", {}),
        "current_units": data.get("current_units", {}),
        "recent_actuals": recent,
        "pro_forecast": pro,
    }


def _merge_actuals(history: list[dict], recent: list[dict]) -> list[dict]:
    by_date = {}
    for rec in history + recent:
        by_date[rec["date"]] = rec
    return [by_date[d] for d in sorted(by_date)]


# --------------------------------------------------------------------------- #
# 3. Feature engineering (same function for training and forecasting)
# --------------------------------------------------------------------------- #
FEATURE_NAMES = (
    ["doy_sin", "doy_cos"]
    + [f"{v}_lag{l}" for l in LAGS for v in ("tmax", "tmin", "tmean", "precip")]
    + ["tmean_roll7", "precip_roll7", "wind_lag1"]
)


def _season(date: dt.date) -> tuple[float, float]:
    ang = 2.0 * np.pi * date.timetuple().tm_yday / 365.25
    return float(np.sin(ang)), float(np.cos(ang))


def features_for(history: list[dict], target_date: dt.date) -> list[float]:
    sin, cos = _season(target_date)
    feats = [sin, cos]
    for lag in LAGS:
        r = history[-lag]
        feats += [r["tmax"], r["tmin"], r["tmean"], r["precip"]]
    last7 = history[-7:]
    feats.append(float(np.mean([r["tmean"] for r in last7])))
    feats.append(float(np.mean([r["precip"] for r in last7])))
    feats.append(history[-1]["wind"])
    return feats


def _build_supervised(records: list[dict]):
    X, y_reg, y_clf = [], [], []
    for i in range(MIN_LOOKBACK, len(records)):
        X.append(features_for(records[:i], records[i]["date"]))
        tgt = records[i]
        y_reg.append([tgt["tmax"], tgt["tmin"], tgt["tmean"], tgt["precip"]])
        y_clf.append(1 if tgt["precip"] >= RAIN_THRESHOLD_MM else 0)
    return np.array(X, float), np.array(y_reg, float), np.array(y_clf, int)


# --------------------------------------------------------------------------- #
# 4. Train + evaluate + forecast
# --------------------------------------------------------------------------- #
def train_and_forecast(city: str) -> dict[str, Any]:
    place = geocode(city)
    lat, lon, tz = place["latitude"], place["longitude"], place["timezone"]

    history = fetch_history(lat, lon, tz)  # archive endpoint (lenient)

    # Live feed (current + recent + pro forecast) may be rate-limited on cloud
    # IPs. If so, degrade gracefully instead of failing the whole request.
    live_available = True
    try:
        live = fetch_recent_and_forecast(lat, lon, tz)
        records = _merge_actuals(history, live["recent_actuals"])
        current = live["current"]
        current_units = live["current_units"]
        pro_forecast = live["pro_forecast"]
    except WeatherError:
        live_available = False
        records = history
        last = history[-1]
        current = {
            "temperature_2m": round(last["tmean"], 1),
            "apparent_temperature": round(last["tmean"], 1),
            "relative_humidity_2m": None,
            "weather_code": last.get("weather_code", 0),
            "wind_speed_10m": round(last.get("wind", 0.0), 1),
        }
        current_units = {"temperature_2m": "°C", "wind_speed_10m": "km/h"}
        pro_forecast = []

    if len(records) < 400:
        raise WeatherError("Not enough historical data for this location to train a model.")

    X, y_reg, y_clf = _build_supervised(records)

    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr_r, yte_r = y_reg[:split], y_reg[split:]
    ytr_c, yte_c = y_clf[:split], y_clf[split:]

    reg = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=42)
    reg.fit(Xtr, ytr_r)
    pred_te = reg.predict(Xte)
    mae_tmax = float(mean_absolute_error(yte_r[:, 0], pred_te[:, 0]))
    mae_tmin = float(mean_absolute_error(yte_r[:, 1], pred_te[:, 1]))
    base_tmax = float(mean_absolute_error(yte_r[:, 0], Xte[:, FEATURE_NAMES.index("tmax_lag1")]))
    base_tmin = float(mean_absolute_error(yte_r[:, 1], Xte[:, FEATURE_NAMES.index("tmin_lag1")]))

    rain_accuracy, clf = None, None
    if len(set(ytr_c)) > 1:
        clf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=-1,
                                     random_state=42, class_weight="balanced")
        clf.fit(Xtr, ytr_c)
        rain_accuracy = float((clf.predict(Xte) == yte_c).mean() * 100.0)

    reg.fit(X, y_reg)
    if clf is not None:
        clf.fit(X, y_clf)

    my_forecast = _roll_forward(reg, clf, records)

    return {
        "place": place,
        "current": current,
        "current_units": current_units,
        "live_available": live_available,
        "recent_actuals": [_public(r) for r in records[-14:]],
        "my_forecast": my_forecast,
        "pro_forecast": [_public(r) for r in pro_forecast],
        "metrics": {
            "mae_tmax": round(mae_tmax, 2), "mae_tmin": round(mae_tmin, 2),
            "baseline_tmax": round(base_tmax, 2), "baseline_tmin": round(base_tmin, 2),
            "rain_accuracy": round(rain_accuracy, 1) if rain_accuracy is not None else None,
            "train_days": len(records), "test_days": len(Xte), "history_years": HISTORY_YEARS,
        },
    }


def _roll_forward(reg, clf, records: list[dict]) -> list[dict]:
    """Autoregressive forecast, always anchored to real future dates (handles the
    case where the newest data we have is a few days old)."""
    working = list(records)
    last_date = working[-1]["date"]
    today = dt.date.today()
    total_steps = (today - last_date).days + FORECAST_DAYS
    out = []
    for step in range(1, total_steps + 1):
        target = last_date + dt.timedelta(days=step)
        feats = np.array(features_for(working, target), float).reshape(1, -1)
        tmax, tmin, tmean, precip = (float(v) for v in reg.predict(feats)[0])
        precip = max(0.0, precip)
        rain_prob = round(float(clf.predict_proba(feats)[0][1]) * 100.0, 0) if clf is not None else None
        working.append({"date": target, "tmax": tmax, "tmin": tmin, "tmean": tmean,
                        "precip": precip, "wind": working[-1]["wind"]})
        if target > today:
            out.append({
                "date": target.isoformat(), "tmax": round(tmax, 1), "tmin": round(tmin, 1),
                "tmean": round(tmean, 1), "precip": round(precip, 1), "rain_prob": rain_prob,
                "weather_code": _guess_code(precip, rain_prob),
            })
        if len(out) >= FORECAST_DAYS:
            break
    return out


def _guess_code(precip: float, rain_prob: float | None) -> int:
    p = rain_prob if rain_prob is not None else (80 if precip >= RAIN_THRESHOLD_MM else 0)
    if precip >= 10 or p >= 80:
        return 65
    if precip >= RAIN_THRESHOLD_MM or p >= 55:
        return 61
    if p >= 35:
        return 3
    if p >= 20:
        return 2
    return 1


def _public(rec: dict) -> dict:
    return {
        "date": rec["date"].isoformat() if isinstance(rec["date"], dt.date) else rec["date"],
        "tmax": round(rec["tmax"], 1), "tmin": round(rec["tmin"], 1),
        "tmean": round(rec["tmean"], 1), "precip": round(rec["precip"], 1),
        "weather_code": rec.get("weather_code", 0),
    }
