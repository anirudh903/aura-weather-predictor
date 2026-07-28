"""
Vercel serverless function:  GET /api/predict?city=<name>

Self-contained (only numpy + requests) so it fits Vercel's function-size limit.
Downloads ~8 years of daily weather from the free Open-Meteo API and forecasts the
next 7 days with the ANALOG method (k-nearest historical days). Returns JSON.
"""
from http.server import BaseHTTPRequestHandler
import datetime as dt
import json
from urllib.parse import urlparse, parse_qs

import numpy as np
import requests

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
              "precipitation_sum", "wind_speed_10m_max"]
HISTORY_YEARS = 8
FORECAST_DAYS = 7
RAIN_THRESHOLD_MM = 1.0
K = 25
ROLL = 7
HTTP_TIMEOUT = 25


class WeatherError(Exception):
    pass


# ----------------------------- data layer ---------------------------------- #
def geocode(city):
    city = (city or "").strip()
    if not city:
        raise WeatherError("Please type a city name.")
    try:
        r = requests.get(GEO_URL, params={"name": city, "count": 1, "language": "en",
                                          "format": "json"}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results")
    except requests.RequestException as exc:
        raise WeatherError(f"Could not reach the location service: {exc}") from exc
    if not results:
        raise WeatherError(f'Could not find a city called "{city}". Check the spelling?')
    top = results[0]
    parts = [top["name"]] + [top[k] for k in ("admin1", "country") if top.get(k)]
    return {"name": top["name"], "label": ", ".join(parts),
            "latitude": top["latitude"], "longitude": top["longitude"],
            "country": top.get("country", ""), "timezone": top.get("timezone", "auto")}


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
        out.append({"date": dt.date.fromisoformat(day), "tmax": float(tmax),
                    "tmin": float(tmin), "tmean": float(tmean),
                    "precip": float(precip) if precip is not None else 0.0,
                    "wind": float(wind) if wind is not None else 0.0})
    return out


def fetch_history(lat, lon, tz):
    end = dt.date.today() - dt.timedelta(days=6)
    start = end - dt.timedelta(days=365 * HISTORY_YEARS)
    try:
        r = requests.get(ARCHIVE_URL, params={"latitude": lat, "longitude": lon,
                         "start_date": start.isoformat(), "end_date": end.isoformat(),
                         "daily": ",".join(DAILY_VARS), "timezone": tz}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherError(f"Could not download historical weather: {exc}") from exc
    return _daily_records(r.json().get("daily", {}))


def fetch_recent_and_forecast(lat, lon, tz):
    try:
        r = requests.get(FORECAST_URL, params={"latitude": lat, "longitude": lon,
                         "daily": ",".join(DAILY_VARS + ["weather_code"]),
                         "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
                         "past_days": 15, "forecast_days": FORECAST_DAYS,
                         "timezone": tz}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        raise WeatherError(f"Could not download current conditions: {exc}") from exc
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


# ----------------------------- analog model -------------------------------- #
def _season(date):
    ang = 2.0 * np.pi * date.timetuple().tm_yday / 365.25
    return np.sin(ang), np.cos(ang)


def _feature(records, i):
    r = records[i]
    s, c = _season(r["date"])
    lo = max(0, i - ROLL + 1)
    roll_t = np.mean([x["tmean"] for x in records[lo:i + 1]])
    roll_p = np.mean([x["precip"] for x in records[lo:i + 1]])
    return [s, c, r["tmax"], r["tmin"], r["tmean"], r["precip"], roll_t, roll_p]


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
            "tmax": round(r["tmax"], 1), "tmin": round(r["tmin"], 1),
            "tmean": round(r["tmean"], 1), "precip": round(r["precip"], 1),
            "weather_code": r.get("weather_code", 0)}


def train_and_forecast(city):
    place = geocode(city)
    lat, lon, tz = place["latitude"], place["longitude"], place["timezone"]
    history = fetch_history(lat, lon, tz)
    live = fetch_recent_and_forecast(lat, lon, tz)
    records = _merge(history, live["recent_actuals"])
    if len(records) < 400:
        raise WeatherError("Not enough historical data for this location.")

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
    forecast = []
    for step in range(1, FORECAST_DAYS + 1):
        (tmax, tmin, tmean, precip), rain_prob = model.predict(_feature(working, len(working) - 1))
        precip = max(0.0, float(precip))
        target = last_date + dt.timedelta(days=step)
        forecast.append({"date": target.isoformat(), "tmax": round(float(tmax), 1),
                         "tmin": round(float(tmin), 1), "tmean": round(float(tmean), 1),
                         "precip": round(precip, 1), "rain_prob": round(rain_prob, 0),
                         "weather_code": _code(precip, rain_prob)})
        working.append({"date": target, "tmax": tmax, "tmin": tmin, "tmean": tmean,
                        "precip": precip, "wind": working[-1]["wind"]})

    return {"place": place, "current": live["current"], "current_units": live["current_units"],
            "recent_actuals": [_pub(r) for r in records[-14:]],
            "my_forecast": forecast, "pro_forecast": [_pub(r) for r in live["pro_forecast"]],
            "metrics": {"mae_tmax": round(mae_tmax, 2), "mae_tmin": round(mae_tmin, 2),
                        "baseline_tmax": round(base_tmax, 2), "baseline_tmin": round(base_tmin, 2),
                        "rain_accuracy": round(rain_acc, 1), "train_days": len(records),
                        "test_days": len(yte), "history_years": HISTORY_YEARS, "method": "analog"}}


# ----------------------------- HTTP handler -------------------------------- #
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        city = (parse_qs(urlparse(self.path).query).get("city", [""])[0] or "").strip()
        try:
            if not city:
                return self._send({"error": "Please enter a city name."}, 400)
            payload = train_and_forecast(city)
            payload["cached"] = False
            self._send(payload, 200)
        except WeatherError as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send({"error": f"Something went wrong while building the forecast: {exc}"}, 500)

    def _send(self, payload, status):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
