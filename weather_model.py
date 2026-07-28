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
"""

from __future__ import annotations

import datetime as dt
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

# Daily variables we ask the API for.
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# How many years of history to train on, and how far ahead we forecast.
HISTORY_YEARS = 8
FORECAST_DAYS = 7
RAIN_THRESHOLD_MM = 1.0  # a day counts as "rainy" if it gets at least this much

# Lag features: we let the model look at the weather 1, 2, 3 and 7 days ago.
LAGS = [1, 2, 3, 7]
MIN_LOOKBACK = max(LAGS)  # need at least this many prior days to build a feature row

HTTP_TIMEOUT = 30


class WeatherError(Exception):
    """Friendly, user-facing error (e.g. city not found, API down)."""


# --------------------------------------------------------------------------- #
# 1. Geocoding: city name -> coordinates
# --------------------------------------------------------------------------- #
def geocode(city: str) -> dict[str, Any]:
    city = (city or "").strip()
    if not city:
        raise WeatherError("Please type a city name.")
    try:
        r = requests.get(
            GEO_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        raise WeatherError(f"Could not reach the location service: {exc}") from exc

    results = data.get("results")
    if not results:
        raise WeatherError(f'Could not find a city called "{city}". Check the spelling?')

    top = results[0]
    name = top["name"]
    parts = [name]
    if top.get("admin1"):
        parts.append(top["admin1"])
    if top.get("country"):
        parts.append(top["country"])
    return {
        "name": name,
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
    """Turn Open-Meteo's column-wise JSON into a clean, row-wise list of days."""
    times = daily.get("time", [])
    records = []
    for i, day in enumerate(times):
        def val(key):
            seq = daily.get(key) or []
            return seq[i] if i < len(seq) else None

        tmax = val("temperature_2m_max")
        tmin = val("temperature_2m_min")
        tmean = val("temperature_2m_mean")
        precip = val("precipitation_sum")
        wind = val("wind_speed_10m_max")

        # If the mean is missing, approximate it from max/min.
        if tmean is None and tmax is not None and tmin is not None:
            tmean = (tmax + tmin) / 2.0

        # Skip days with holes in the key fields.
        if tmax is None or tmin is None or tmean is None:
            continue

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
    try:
        r = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": ",".join(DAILY_VARS),
                "timezone": tz,
            },
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        raise WeatherError(f"Could not download historical weather: {exc}") from exc
    return _daily_records(data.get("daily", {}))


def fetch_recent_and_forecast(lat: float, lon: float, tz: str) -> dict[str, Any]:
    """
    One call to the forecast endpoint gets us three things at once:
      - current conditions (for the big hero card),
      - the last ~15 days of *actual* weather (to bridge the archive's 5-day gap
        and to seed our own forecast),
      - Open-Meteo's own 7-day forecast (our professional benchmark).
    """
    try:
        r = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": ",".join(DAILY_VARS + ["weather_code"]),
                "current": ",".join(
                    [
                        "temperature_2m",
                        "apparent_temperature",
                        "relative_humidity_2m",
                        "weather_code",
                        "wind_speed_10m",
                        "precipitation",
                    ]
                ),
                "past_days": 15,
                "forecast_days": FORECAST_DAYS,
                "timezone": tz,
            },
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        raise WeatherError(f"Could not download current conditions: {exc}") from exc

    today = dt.date.today()
    daily = data.get("daily", {})
    codes = daily.get("weather_code", []) or []
    times = daily.get("time", [])

    recent_actuals: list[dict[str, Any]] = []
    pro_forecast: list[dict[str, Any]] = []
    for i, rec in enumerate(_daily_records(daily)):
        # weather_code is not part of _daily_records, so re-attach it here
        try:
            idx = times.index(rec["date"].isoformat())
            rec["weather_code"] = int(codes[idx]) if idx < len(codes) and codes[idx] is not None else 0
        except (ValueError, TypeError):
            rec["weather_code"] = 0

        if rec["date"] <= today:
            recent_actuals.append(rec)
        else:
            pro_forecast.append(rec)

    return {
        "current": data.get("current", {}),
        "current_units": data.get("current_units", {}),
        "recent_actuals": recent_actuals,
        "pro_forecast": pro_forecast,
    }


def _merge_actuals(history: list[dict], recent: list[dict]) -> list[dict]:
    """Combine archive history with the last-15-days actuals, dedup by date."""
    by_date: dict[dt.date, dict] = {}
    for rec in history + recent:
        by_date[rec["date"]] = rec  # recent wins on overlap (fresher)
    return [by_date[d] for d in sorted(by_date)]


# --------------------------------------------------------------------------- #
# 3. Feature engineering
#    The SAME function builds features for training and for forecasting, so
#    there's no chance of train/inference mismatch.
# --------------------------------------------------------------------------- #
FEATURE_NAMES = (
    ["doy_sin", "doy_cos"]
    + [f"{v}_lag{l}" for l in LAGS for v in ("tmax", "tmin", "tmean", "precip")]
    + ["tmean_roll7", "precip_roll7", "wind_lag1"]
)


def _season(date: dt.date) -> tuple[float, float]:
    """Encode the day-of-year as a point on a circle so the model understands
    that Dec 31 and Jan 1 are neighbours, not opposites."""
    doy = date.timetuple().tm_yday
    ang = 2.0 * np.pi * doy / 365.25
    return float(np.sin(ang)), float(np.cos(ang))


def features_for(history: list[dict], target_date: dt.date) -> list[float]:
    """Build one feature row describing everything the model knows the day
    BEFORE `target_date`. `history` must end on target_date - 1 day."""
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
    """Slide over the history to build (X, y) pairs: from the days up to i-1,
    predict day i."""
    X, y_reg, y_clf = [], [], []
    for i in range(MIN_LOOKBACK, len(records)):
        X.append(features_for(records[:i], records[i]["date"]))
        tgt = records[i]
        y_reg.append([tgt["tmax"], tgt["tmin"], tgt["tmean"], tgt["precip"]])
        y_clf.append(1 if tgt["precip"] >= RAIN_THRESHOLD_MM else 0)
    return np.array(X, dtype=float), np.array(y_reg, dtype=float), np.array(y_clf, dtype=int)


# --------------------------------------------------------------------------- #
# 4. Train + evaluate + forecast
# --------------------------------------------------------------------------- #
def train_and_forecast(city: str) -> dict[str, Any]:
    """The whole pipeline for one city. Returns everything the UI needs."""
    place = geocode(city)
    lat, lon, tz = place["latitude"], place["longitude"], place["timezone"]

    history = fetch_history(lat, lon, tz)
    live = fetch_recent_and_forecast(lat, lon, tz)
    records = _merge_actuals(history, live["recent_actuals"])

    if len(records) < 400:
        raise WeatherError("Not enough historical data for this location to train a model.")

    X, y_reg, y_clf = _build_supervised(records)

    # Time-based split: train on the older 80%, test on the most recent 20%.
    # (We never let the model peek at the future during evaluation.)
    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr_r, yte_r = y_reg[:split], y_reg[split:]
    ytr_c, yte_c = y_clf[:split], y_clf[split:]

    reg = RandomForestRegressor(
        n_estimators=250, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )
    reg.fit(Xtr, ytr_r)

    # Honest accuracy check on unseen recent data.
    pred_te = reg.predict(Xte)
    mae_tmax = float(mean_absolute_error(yte_r[:, 0], pred_te[:, 0]))
    mae_tmin = float(mean_absolute_error(yte_r[:, 1], pred_te[:, 1]))

    # Persistence baseline = "tomorrow will be the same as today". A model is
    # only worth anything if it beats this.
    base_tmax = float(mean_absolute_error(yte_r[:, 0], Xte[:, FEATURE_NAMES.index("tmax_lag1")]))
    base_tmin = float(mean_absolute_error(yte_r[:, 1], Xte[:, FEATURE_NAMES.index("tmin_lag1")]))

    # Rain classifier (only if we've actually seen both rainy and dry days).
    rain_accuracy = None
    clf = None
    if len(set(ytr_c)) > 1:
        clf = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, n_jobs=-1,
            random_state=42, class_weight="balanced",
        )
        clf.fit(Xtr, ytr_c)
        rain_accuracy = float((clf.predict(Xte) == yte_c).mean() * 100.0)

    # Retrain on ALL data for the sharpest possible real forecast.
    reg.fit(X, y_reg)
    if clf is not None:
        clf.fit(X, y_clf)

    my_forecast = _roll_forward(reg, clf, records)

    return {
        "place": place,
        "current": live["current"],
        "current_units": live["current_units"],
        "recent_actuals": [_public(r) for r in records[-14:]],
        "my_forecast": my_forecast,
        "pro_forecast": [_public(r) for r in live["pro_forecast"]],
        "metrics": {
            "mae_tmax": round(mae_tmax, 2),
            "mae_tmin": round(mae_tmin, 2),
            "baseline_tmax": round(base_tmax, 2),
            "baseline_tmin": round(base_tmin, 2),
            "rain_accuracy": round(rain_accuracy, 1) if rain_accuracy is not None else None,
            "train_days": len(records),
            "test_days": len(Xte),
            "history_years": HISTORY_YEARS,
        },
    }


def _roll_forward(reg, clf, records: list[dict]) -> list[dict]:
    """Autoregressive forecast: predict tomorrow, then treat that prediction as
    'today' and predict the day after, and so on for a week."""
    working = list(records)  # a growing copy we append our own predictions to
    last_date = working[-1]["date"]
    out = []
    for step in range(1, FORECAST_DAYS + 1):
        target = last_date + dt.timedelta(days=step)
        feats = np.array(features_for(working, target), dtype=float).reshape(1, -1)
        tmax, tmin, tmean, precip = (float(v) for v in reg.predict(feats)[0])
        precip = max(0.0, precip)

        rain_prob = None
        if clf is not None:
            rain_prob = round(float(clf.predict_proba(feats)[0][1]) * 100.0, 0)

        out.append(
            {
                "date": target.isoformat(),
                "tmax": round(tmax, 1),
                "tmin": round(tmin, 1),
                "tmean": round(tmean, 1),
                "precip": round(precip, 1),
                "rain_prob": rain_prob,
                "weather_code": _guess_code(precip, rain_prob),
            }
        )
        # Feed the prediction back in so the next day builds on it.
        working.append(
            {"date": target, "tmax": tmax, "tmin": tmin, "tmean": tmean,
             "precip": precip, "wind": working[-1]["wind"]}
        )
    return out


def _guess_code(precip: float, rain_prob: float | None) -> int:
    """Rough WMO weather code from our predicted numbers, just for the icon."""
    p = rain_prob if rain_prob is not None else (80 if precip >= RAIN_THRESHOLD_MM else 0)
    if precip >= 10 or p >= 80:
        return 65  # heavy rain
    if precip >= RAIN_THRESHOLD_MM or p >= 55:
        return 61  # rain
    if p >= 35:
        return 3   # overcast
    if p >= 20:
        return 2   # partly cloudy
    return 1       # mainly clear


def _public(rec: dict) -> dict:
    """Trim an internal record down to JSON-friendly fields for the UI."""
    return {
        "date": rec["date"].isoformat() if isinstance(rec["date"], dt.date) else rec["date"],
        "tmax": round(rec["tmax"], 1),
        "tmin": round(rec["tmin"], 1),
        "tmean": round(rec["tmean"], 1),
        "precip": round(rec["precip"], 1),
        "weather_code": rec.get("weather_code", 0),
    }
