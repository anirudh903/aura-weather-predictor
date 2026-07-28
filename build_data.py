"""
build_data.py
=============
Runs on a schedule (hourly) via GitHub Actions — NOT on the live website.

For every city in CITIES it:
  1. downloads ~8 years of daily weather + the recent/live feed from Open-Meteo,
  2. trains the analog (k-nearest-days) model and forecasts the next 7 days,
  3. writes the finished result to docs/data/<slug>.json.

The website then just reads those JSON files. Because the site never calls a
weather API itself, there is nothing to rate-limit — this is what makes the
deployment rock-solid on free hosting (GitHub Pages).

Only needs: numpy + requests.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import numpy as np
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
              "precipitation_sum", "wind_speed_10m_max"]
HISTORY_YEARS = 8
FORECAST_DAYS = 7
RAIN_THRESHOLD_MM = 1.0
K = 25
ROLL = 7
HTTP_TIMEOUT = 30
RETRIES = 5

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "data")

# Curated list — major Indian cities + world capitals. Add/remove freely.
CITIES = [
    {"slug": "mumbai", "name": "Mumbai", "label": "Mumbai, India", "lat": 19.076, "lon": 72.877},
    {"slug": "delhi", "name": "Delhi", "label": "Delhi, India", "lat": 28.6139, "lon": 77.209},
    {"slug": "bengaluru", "name": "Bengaluru", "label": "Bengaluru, India", "lat": 12.9716, "lon": 77.5946},
    {"slug": "hyderabad", "name": "Hyderabad", "label": "Hyderabad, India", "lat": 17.385, "lon": 78.4867},
    {"slug": "chennai", "name": "Chennai", "label": "Chennai, India", "lat": 13.0827, "lon": 80.2707},
    {"slug": "kolkata", "name": "Kolkata", "label": "Kolkata, India", "lat": 22.5726, "lon": 88.3639},
    {"slug": "pune", "name": "Pune", "label": "Pune, India", "lat": 18.5204, "lon": 73.8567},
    {"slug": "ahmedabad", "name": "Ahmedabad", "label": "Ahmedabad, India", "lat": 23.0225, "lon": 72.5714},
    {"slug": "jaipur", "name": "Jaipur", "label": "Jaipur, India", "lat": 26.9124, "lon": 75.7873},
    {"slug": "lucknow", "name": "Lucknow", "label": "Lucknow, India", "lat": 26.8467, "lon": 80.9462},
    {"slug": "kochi", "name": "Kochi", "label": "Kochi, India", "lat": 9.9312, "lon": 76.2673},
    {"slug": "chandigarh", "name": "Chandigarh", "label": "Chandigarh, India", "lat": 30.7333, "lon": 76.7794},
    {"slug": "london", "name": "London", "label": "London, UK", "lat": 51.5074, "lon": -0.1278},
    {"slug": "new-york", "name": "New York", "label": "New York, USA", "lat": 40.7128, "lon": -74.006},
    {"slug": "tokyo", "name": "Tokyo", "label": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503},
    {"slug": "paris", "name": "Paris", "label": "Paris, France", "lat": 48.8566, "lon": 2.3522},
    {"slug": "dubai", "name": "Dubai", "label": "Dubai, UAE", "lat": 25.2048, "lon": 55.2708},
    {"slug": "singapore", "name": "Singapore", "label": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"slug": "sydney", "name": "Sydney", "label": "Sydney, Australia", "lat": -33.8688, "lon": 151.2093},
    {"slug": "los-angeles", "name": "Los Angeles", "label": "Los Angeles, USA", "lat": 34.0522, "lon": -118.2437},
    {"slug": "toronto", "name": "Toronto", "label": "Toronto, Canada", "lat": 43.6532, "lon": -79.3832},
    {"slug": "berlin", "name": "Berlin", "label": "Berlin, Germany", "lat": 52.52, "lon": 13.405},
    {"slug": "dublin", "name": "Dublin", "label": "Dublin, Ireland", "lat": 53.3498, "lon": -6.2603},
    {"slug": "bangkok", "name": "Bangkok", "label": "Bangkok, Thailand", "lat": 13.7563, "lon": 100.5018},
    {"slug": "cairo", "name": "Cairo", "label": "Cairo, Egypt", "lat": 30.0444, "lon": 31.2357},
    {"slug": "sao-paulo", "name": "São Paulo", "label": "São Paulo, Brazil", "lat": -23.5505, "lon": -46.6333},
    {"slug": "moscow", "name": "Moscow", "label": "Moscow, Russia", "lat": 55.7558, "lon": 37.6173},
    {"slug": "istanbul", "name": "Istanbul", "label": "Istanbul, Türkiye", "lat": 41.0082, "lon": 28.9784},
]


class WeatherError(Exception):
    pass


def _get_json(url, params, what):
    last = "unknown error"
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else 0.8 * (2 ** attempt)
                time.sleep(min(wait, 8.0))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(0.8 * (2 ** attempt))
    raise WeatherError(f"Could not download {what}: {last}")


def _daily_records(daily):
    times = daily.get("time", [])
    out = []
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
        out.append({"date": dt.date.fromisoformat(day), "tmax": float(tmax), "tmin": float(tmin),
                    "tmean": float(tmean), "precip": float(precip) if precip is not None else 0.0,
                    "wind": float(wind) if wind is not None else 0.0})
    return out


def fetch_history(lat, lon):
    end = dt.date.today() - dt.timedelta(days=6)
    start = end - dt.timedelta(days=365 * HISTORY_YEARS)
    data = _get_json(ARCHIVE_URL, {"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
                                   "end_date": end.isoformat(), "daily": ",".join(DAILY_VARS),
                                   "timezone": "auto"}, "historical weather")
    return _daily_records(data.get("daily", {}))


def fetch_recent_and_forecast(lat, lon):
    data = _get_json(FORECAST_URL, {"latitude": lat, "longitude": lon,
                     "daily": ",".join(DAILY_VARS + ["weather_code"]),
                     "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
                     "past_days": 15, "forecast_days": FORECAST_DAYS, "timezone": "auto"}, "current conditions")
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
    return {"current": data.get("current", {}), "current_units": data.get("current_units", {}),
            "recent_actuals": recent, "pro_forecast": pro}


def _merge(history, recent):
    by_date = {}
    for rec in history + recent:
        by_date[rec["date"]] = rec
    return [by_date[d] for d in sorted(by_date)]


def _season(date):
    ang = 2.0 * np.pi * date.timetuple().tm_yday / 365.25
    return np.sin(ang), np.cos(ang)


def _feature(records, i):
    r = records[i]
    s, c = _season(r["date"])
    lo = max(0, i - ROLL + 1)
    return [s, c, r["tmax"], r["tmin"], r["tmean"], r["precip"],
            float(np.mean([x["tmean"] for x in records[lo:i + 1]])),
            float(np.mean([x["precip"] for x in records[lo:i + 1]]))]


def _build(records):
    F, O = [], []
    for i in range(ROLL, len(records) - 1):
        F.append(_feature(records, i))
        nxt = records[i + 1]
        O.append([nxt["tmax"], nxt["tmin"], nxt["tmean"], nxt["precip"]])
    return np.array(F, float), np.array(O, float)


class Analog:
    def __init__(self, F, O, k=K):
        self.mean, self.std = F.mean(axis=0), F.std(axis=0) + 1e-6
        self.Fz, self.O, self.k = (F - self.mean) / self.std, O, min(k, len(F))

    def predict(self, feat):
        q = (np.array(feat, float) - self.mean) / self.std
        d = np.sqrt(((self.Fz - q) ** 2).sum(axis=1))
        idx = np.argpartition(d, self.k - 1)[: self.k]
        out = self.O[idx]
        return out.mean(axis=0), float((out[:, 3] >= RAIN_THRESHOLD_MM).mean()) * 100.0


def _code(precip, rp):
    if precip >= 10 or rp >= 80:
        return 65
    if precip >= RAIN_THRESHOLD_MM or rp >= 55:
        return 61
    if rp >= 35:
        return 3
    if rp >= 20:
        return 2
    return 1


def _pub(r):
    return {"date": r["date"].isoformat() if isinstance(r["date"], dt.date) else r["date"],
            "tmax": round(r["tmax"], 1), "tmin": round(r["tmin"], 1), "tmean": round(r["tmean"], 1),
            "precip": round(r["precip"], 1), "weather_code": r.get("weather_code", 0)}


def compute(city, generated_at):
    lat, lon = city["lat"], city["lon"]
    history = fetch_history(lat, lon)
    if len(history) < 400:
        raise WeatherError("not enough history")

    live_available = True
    try:
        live = fetch_recent_and_forecast(lat, lon)
        records = _merge(history, live["recent_actuals"])
        current, current_units, pro = live["current"], live["current_units"], live["pro_forecast"]
    except WeatherError:
        live_available = False
        records = history
        last = history[-1]
        current = {"temperature_2m": round(last["tmean"], 1), "apparent_temperature": round(last["tmean"], 1),
                   "relative_humidity_2m": None, "weather_code": last.get("weather_code", 0),
                   "wind_speed_10m": round(last.get("wind", 0.0), 1)}
        current_units = {"temperature_2m": "°C", "wind_speed_10m": "km/h"}
        pro = []

    F, O = _build(records)
    split = int(len(F) * 0.8)
    ev = Analog(F[:split], O[:split])
    pred = np.array([ev.predict(F[i])[0] for i in range(split, len(F))])
    yte = O[split:]
    mae_tmax = float(np.mean(np.abs(pred[:, 0] - yte[:, 0])))
    mae_tmin = float(np.mean(np.abs(pred[:, 1] - yte[:, 1])))
    base_tmax = float(np.mean(np.abs(F[split:, 2] - yte[:, 0])))
    base_tmin = float(np.mean(np.abs(F[split:, 3] - yte[:, 1])))
    rp = np.array([ev.predict(F[i])[1] for i in range(split, len(F))]) >= 50
    rain_acc = float((rp == (yte[:, 3] >= RAIN_THRESHOLD_MM)).mean() * 100.0)

    model = Analog(F, O)
    working = list(records)
    last_date = working[-1]["date"]
    today = dt.date.today()
    total_steps = (today - last_date).days + FORECAST_DAYS
    forecast = []
    for step in range(1, total_steps + 1):
        (tmax, tmin, tmean, precip), rain_prob = model.predict(_feature(working, len(working) - 1))
        precip = max(0.0, float(precip))
        target = last_date + dt.timedelta(days=step)
        working.append({"date": target, "tmax": tmax, "tmin": tmin, "tmean": tmean, "precip": precip,
                        "wind": working[-1]["wind"]})
        if target > today:
            forecast.append({"date": target.isoformat(), "tmax": round(float(tmax), 1),
                             "tmin": round(float(tmin), 1), "tmean": round(float(tmean), 1),
                             "precip": round(precip, 1), "rain_prob": round(rain_prob, 0),
                             "weather_code": _code(precip, rain_prob)})
        if len(forecast) >= FORECAST_DAYS:
            break

    return {"place": {"name": city["name"], "label": city["label"]},
            "current": current, "current_units": current_units, "live_available": live_available,
            "recent_actuals": [_pub(r) for r in records[-14:]], "my_forecast": forecast,
            "pro_forecast": [_pub(r) for r in pro], "generated_at": generated_at,
            "metrics": {"mae_tmax": round(mae_tmax, 2), "mae_tmin": round(mae_tmin, 2),
                        "baseline_tmax": round(base_tmax, 2), "baseline_tmin": round(base_tmin, 2),
                        "rain_accuracy": round(rain_acc, 1), "train_days": len(records),
                        "test_days": len(yte), "history_years": HISTORY_YEARS, "method": "analog"}}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    index = []
    ok = 0
    for city in CITIES:
        try:
            payload = compute(city, generated_at)
            with open(os.path.join(OUT_DIR, f"{city['slug']}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            index.append({"slug": city["slug"], "name": city["name"], "label": city["label"]})
            ok += 1
            print(f"  [ok]   {city['name']:14s} live={payload['live_available']} rain_acc={payload['metrics']['rain_accuracy']}%")
        except Exception as exc:  # noqa: BLE001 - keep going if one city fails
            print(f"  [FAIL] {city['name']:14s} {exc}")
        time.sleep(0.4)
    with open(os.path.join(OUT_DIR, "cities.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": generated_at, "cities": index}, f, ensure_ascii=False)
    print(f"\nDone: {ok}/{len(CITIES)} cities written to {OUT_DIR}")
    if ok == 0:
        raise SystemExit("No cities succeeded — failing the build.")


if __name__ == "__main__":
    main()
