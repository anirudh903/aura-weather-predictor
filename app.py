"""
app.py
======
Tiny Flask server that ties everything together:

  GET  /                     -> the web page (templates/index.html)
  GET  /api/predict?city=... -> trains a model for that city and returns its
                                own 7-day forecast, the live conditions, the
                                professional forecast, and accuracy metrics.

Run it with:  python app.py   (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import time
import traceback

from flask import Flask, jsonify, render_template, request

from weather_model import WeatherError, train_and_forecast

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Simple in-memory cache. Training takes a couple of seconds, so we keep each
# city's result for 30 minutes to make repeat clicks feel instant.
# --------------------------------------------------------------------------- #
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 30 * 60  # seconds


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict")
def predict():
    city = (request.args.get("city") or "").strip()
    if not city:
        return jsonify({"error": "Please enter a city name."}), 400

    key = city.lower()
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        payload = dict(cached[1])
        payload["cached"] = True
        return jsonify(payload)

    try:
        result = train_and_forecast(city)
    except WeatherError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - last-resort guard for a demo app
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong while building the forecast: {exc}"}), 500

    _CACHE[key] = (now, result)
    payload = dict(result)
    payload["cached"] = False
    return jsonify(payload)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5055))
    print(f"\n  Weather Predictor running at:  http://127.0.0.1:{port}\n")
    # use_reloader=False keeps it to a single process (nicer for previewing);
    # debug=True still gives readable error pages.
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
