/* ========================================================================
   Aura front-end
   Calls /api/predict, then paints the hero card, the 7-day forecast, the
   comparison chart, and the model-quality report.
   ======================================================================== */

const $ = (sel) => document.querySelector(sel);

// WMO weather codes -> emoji + human label
const WMO = {
  0:  ["☀️", "clear sky"],
  1:  ["🌤️", "mainly clear"],
  2:  ["⛅", "partly cloudy"],
  3:  ["☁️", "overcast"],
  45: ["🌫️", "fog"],
  48: ["🌫️", "rime fog"],
  51: ["🌦️", "light drizzle"],
  53: ["🌦️", "drizzle"],
  55: ["🌧️", "heavy drizzle"],
  56: ["🌧️", "freezing drizzle"],
  57: ["🌧️", "freezing drizzle"],
  61: ["🌧️", "light rain"],
  63: ["🌧️", "rain"],
  65: ["⛈️", "heavy rain"],
  66: ["🌧️", "freezing rain"],
  67: ["🌧️", "freezing rain"],
  71: ["🌨️", "light snow"],
  73: ["❄️", "snow"],
  75: ["❄️", "heavy snow"],
  77: ["🌨️", "snow grains"],
  80: ["🌦️", "light showers"],
  81: ["🌧️", "showers"],
  82: ["⛈️", "violent showers"],
  85: ["🌨️", "snow showers"],
  86: ["🌨️", "snow showers"],
  95: ["⛈️", "thunderstorm"],
  96: ["⛈️", "thunderstorm + hail"],
  99: ["⛈️", "severe thunderstorm"],
};
const codeInfo = (c) => WMO[c] || ["🌡️", "—"];

const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const fmtDay = (iso) => DOW[new Date(iso + "T00:00").getDay()];
const fmtDate = (iso) => {
  const d = new Date(iso + "T00:00");
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
};

let chart = null;

// ---------- UI state helpers ----------
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function setLoading(on, text) {
  const loader = $("#loader");
  if (on) {
    if (text) $("#loader-text").textContent = text;
    show(loader); hide($("#results")); hide($("#error"));
  } else {
    hide(loader);
  }
}

function showError(msg) {
  setLoading(false);
  const e = $("#error");
  e.textContent = "⚠️  " + msg;
  show(e);
  hide($("#results"));
}

// ---------- main fetch ----------
async function predict(city) {
  if (!city) return;
  $("#city").value = city;
  setLoading(true, `Downloading history for ${city} & training the model…`);
  try {
    const res = await fetch(`/api/predict?city=${encodeURIComponent(city)}`);
    const data = await res.json();
    if (!res.ok) return showError(data.error || "Request failed.");
    render(data);
  } catch (err) {
    showError("Could not reach the server. Is it still running?");
  }
}

// ---------- render ----------
function render(d) {
  setLoading(false);
  renderCurrent(d);
  renderForecast(d.my_forecast);
  renderChart(d);
  renderMetrics(d.metrics);
  show($("#results"));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderCurrent(d) {
  const cur = d.current || {};
  const u = d.current_units || {};
  const [emoji, label] = codeInfo(cur.weather_code);
  $("#cur-emoji").textContent = emoji;
  $("#cur-temp").textContent = Math.round(cur.temperature_2m ?? 0);
  $("#cur-desc").textContent = label;
  $("#cur-place").textContent = d.place?.label || "";
  $("#cur-feels").textContent = `${Math.round(cur.apparent_temperature ?? 0)}°`;
  $("#cur-hum").textContent = `${Math.round(cur.relative_humidity_2m ?? 0)}%`;
  $("#cur-wind").textContent = `${Math.round(cur.wind_speed_10m ?? 0)} ${u.wind_speed_10m || "km/h"}`;
}

function renderForecast(days) {
  const row = $("#forecast");
  row.innerHTML = "";
  days.forEach((day, i) => {
    const [emoji] = codeInfo(day.weather_code);
    const rp = day.rain_prob;
    const rainHtml =
      rp == null
        ? ""
        : `<div class="rain ${rp < 25 ? "dry" : ""}">💧 ${Math.round(rp)}%</div>`;
    const el = document.createElement("div");
    el.className = "day";
    el.style.animationDelay = `${i * 40}ms`;
    el.innerHTML = `
      <div class="dow">${i === 0 ? "Tomorrow" : fmtDay(day.date)}</div>
      <div class="date">${fmtDate(day.date)}</div>
      <div class="ic">${emoji}</div>
      <div class="hi">${Math.round(day.tmax)}°</div>
      <div class="lo">${Math.round(day.tmin)}°</div>
      ${rainHtml}
    `;
    row.appendChild(el);
  });
}

function renderChart(d) {
  const ctx = $("#chart").getContext("2d");

  // recent actuals (last ~10 days) + our forecast + pro forecast, aligned by date
  const recent = (d.recent_actuals || []).slice(-10);
  const mine = d.my_forecast || [];
  const pro = d.pro_forecast || [];

  const labels = [
    ...recent.map((r) => fmtDate(r.date)),
    ...mine.map((r) => fmtDate(r.date)),
  ];

  const pad = (arr, before, after) =>
    [...Array(before).fill(null), ...arr, ...Array(after).fill(null)];

  const actualSeries = [...recent.map((r) => r.tmax), ...mine.map(() => null)];
  const mineSeries = pad(mine.map((r) => r.tmax), recent.length, 0);
  const proSeries = pad(
    mine.map((_, i) => (pro[i] ? pro[i].tmax : null)),
    recent.length,
    0
  );

  const grid = "rgba(255,255,255,0.06)";
  const tick = "#8b95bd";
  const mk = (label, data, color, opts = {}) => ({
    label, data,
    borderColor: color,
    backgroundColor: color + "22",
    tension: 0.35, spanGaps: false, pointRadius: 3, pointHoverRadius: 5,
    borderWidth: 2.5, ...opts,
  });

  const datasets = [
    mk("Recent actual high", actualSeries, "#8b95bd", { borderWidth: 2, borderDash: [2, 3] }),
    mk("Our model (high)", mineSeries, "#7aa2ff", { fill: true }),
  ];
  if (pro.length) datasets.push(mk("Pro forecast (high)", proSeries, "#6ee7d1", { borderDash: [6, 5] }));

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#c7ceeb", usePointStyle: true, padding: 18, font: { size: 12 } } },
        tooltip: {
          backgroundColor: "rgba(12,16,32,0.95)", borderColor: "rgba(255,255,255,0.12)",
          borderWidth: 1, padding: 12, callbacks: { label: (c) => `${c.dataset.label}: ${c.formattedValue}°` },
        },
      },
      scales: {
        x: { grid: { color: grid }, ticks: { color: tick, font: { size: 11 } } },
        y: { grid: { color: grid }, ticks: { color: tick, font: { size: 11 }, callback: (v) => v + "°" } },
      },
    },
  });
}

function renderMetrics(m) {
  const wrap = $("#metrics");
  const beatTmax = (m.baseline_tmax - m.mae_tmax).toFixed(2);
  const beatsBaseline = m.mae_tmax < m.baseline_tmax;

  const cards = [
    {
      label: "High-temp accuracy",
      num: `±${m.mae_tmax}°`,
      sub: `Typical error when predicting the daily high.`,
    },
    {
      label: "Low-temp accuracy",
      num: `±${m.mae_tmin}°`,
      sub: `Typical error when predicting the daily low.`,
    },
    {
      label: "vs. naïve guess",
      num: beatsBaseline ? `${beatTmax}° better` : `${Math.abs(beatTmax)}° worse`,
      sub: beatsBaseline
        ? `Beats “tomorrow = today” (±${m.baseline_tmax}°).`
        : `The simple “tomorrow = today” guess (±${m.baseline_tmax}°) is hard to beat here.`,
      good: beatsBaseline,
    },
  ];

  if (m.rain_accuracy != null) {
    cards.push({
      label: "Rain / no-rain",
      num: `${m.rain_accuracy}%`,
      sub: `Correct rain-vs-dry calls on unseen days.`,
    });
  }

  cards.push({
    label: "Trained on",
    num: `${m.history_years} yrs`,
    sub: `${m.train_days.toLocaleString()} days of history · tested on ${m.test_days.toLocaleString()} recent days.`,
  });

  wrap.innerHTML = cards
    .map(
      (c) => `
      <div class="metric">
        <div class="label">${c.label}</div>
        <div class="num">${c.num}</div>
        <div class="sub ${c.good ? "good" : ""}">${c.sub}</div>
      </div>`
    )
    .join("");
}

// ---------- wire up events ----------
$("#search").addEventListener("submit", (e) => {
  e.preventDefault();
  predict($("#city").value.trim());
});
$("#chips").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-city]");
  if (btn) predict(btn.dataset.city);
});

// first load
predict("Mumbai");
