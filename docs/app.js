/* ========================================================================
   Aura front-end (static build).
   Reads pre-computed JSON generated hourly by a GitHub Action — the site
   itself never calls a weather API, so there is nothing to rate-limit.
   ======================================================================== */

const $ = (sel) => document.querySelector(sel);

const WMO = {
  0: ["☀️", "clear sky"], 1: ["🌤️", "mainly clear"], 2: ["⛅", "partly cloudy"], 3: ["☁️", "overcast"],
  45: ["🌫️", "fog"], 48: ["🌫️", "rime fog"], 51: ["🌦️", "light drizzle"], 53: ["🌦️", "drizzle"],
  55: ["🌧️", "heavy drizzle"], 56: ["🌧️", "freezing drizzle"], 57: ["🌧️", "freezing drizzle"],
  61: ["🌧️", "light rain"], 63: ["🌧️", "rain"], 65: ["⛈️", "heavy rain"], 66: ["🌧️", "freezing rain"],
  67: ["🌧️", "freezing rain"], 71: ["🌨️", "light snow"], 73: ["❄️", "snow"], 75: ["❄️", "heavy snow"],
  77: ["🌨️", "snow grains"], 80: ["🌦️", "light showers"], 81: ["🌧️", "showers"], 82: ["⛈️", "violent showers"],
  85: ["🌨️", "snow showers"], 86: ["🌨️", "snow showers"], 95: ["⛈️", "thunderstorm"],
  96: ["⛈️", "thunderstorm + hail"], 99: ["⛈️", "severe thunderstorm"],
};
const codeInfo = (c) => WMO[c] || ["🌡️", "—"];

const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const fmtDay = (iso) => DOW[new Date(iso + "T00:00").getDay()];
const fmtDate = (iso) => new Date(iso + "T00:00").toLocaleDateString(undefined, { day: "numeric", month: "short" });

function timeAgo(iso) {
  if (!iso) return "";
  const diff = (Date.now() - new Date(iso).getTime()) / 60000; // minutes
  if (diff < 1.5) return "just now";
  if (diff < 60) return `${Math.round(diff)} min ago`;
  const h = diff / 60;
  if (h < 24) return `${Math.round(h)} hr ago`;
  return `${Math.round(h / 24)} days ago`;
}

let chart = null;
let CITIES = [];

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function setLoading(on, text) {
  const loader = $("#loader");
  if (on) { if (text) $("#loader-text").textContent = text; show(loader); hide($("#results")); hide($("#error")); }
  else { hide(loader); }
}

function showError(msg) {
  setLoading(false);
  const e = $("#error");
  e.textContent = "⚠️  " + msg;
  show(e);
  hide($("#results"));
}

// ---------- boot ----------
async function init() {
  try {
    const res = await fetch("data/cities.json", { cache: "no-cache" });
    const data = await res.json();
    CITIES = data.cities || [];
    renderChips();
    if (data.generated_at) $("#updated").textContent = `🕒 Forecasts refreshed ${timeAgo(data.generated_at)} · updates every hour`;
    const first = (CITIES.find((c) => c.slug === "mumbai") || CITIES[0]);
    if (first) selectCity(first.slug);
    else showError("No forecast data available yet — the hourly build may still be running.");
  } catch (err) {
    showError("Could not load the forecast data. Please refresh in a moment.");
  }
}

function renderChips(filter = "") {
  const wrap = $("#chips");
  wrap.innerHTML = "";
  const f = filter.trim().toLowerCase();
  CITIES.forEach((c) => {
    const btn = document.createElement("button");
    btn.textContent = c.name;
    btn.dataset.slug = c.slug;
    if (f && !c.name.toLowerCase().includes(f) && !c.label.toLowerCase().includes(f)) btn.classList.add("hide");
    wrap.appendChild(btn);
  });
}

async function selectCity(slug) {
  setLoading(true, "Loading forecast…");
  try {
    const res = await fetch(`data/${slug}.json`, { cache: "no-cache" });
    if (!res.ok) throw new Error("not found");
    const data = await res.json();
    render(data);
    document.querySelectorAll("#chips button").forEach((b) => b.classList.toggle("active", b.dataset.slug === slug));
  } catch (err) {
    showError("Couldn't load that city's forecast. Try another from the list.");
  }
}

function render(d) {
  setLoading(false);
  $("#offline-note").classList.toggle("hidden", d.live_available !== false);
  renderCurrent(d);
  renderForecast(d.my_forecast);
  renderChart(d);
  renderMetrics(d.metrics);
  if (d.generated_at) $("#updated").textContent = `🕒 Forecasts refreshed ${timeAgo(d.generated_at)} · updates every hour`;
  show($("#results"));
}

function renderCurrent(d) {
  const cur = d.current || {}, u = d.current_units || {};
  const [emoji, label] = codeInfo(cur.weather_code);
  $("#cur-emoji").textContent = emoji;
  $("#cur-temp").textContent = Math.round(cur.temperature_2m ?? 0);
  $("#cur-desc").textContent = label;
  $("#cur-place").textContent = d.place?.label || "";
  $("#cur-feels").textContent = `${Math.round(cur.apparent_temperature ?? 0)}°`;
  $("#cur-hum").textContent = cur.relative_humidity_2m == null ? "—" : `${Math.round(cur.relative_humidity_2m)}%`;
  $("#cur-wind").textContent = `${Math.round(cur.wind_speed_10m ?? 0)} ${u.wind_speed_10m || "km/h"}`;
}

function renderForecast(days) {
  const row = $("#forecast");
  row.innerHTML = "";
  (days || []).forEach((day, i) => {
    const [emoji] = codeInfo(day.weather_code);
    const rp = day.rain_prob;
    const rainHtml = rp == null ? "" : `<div class="rain ${rp < 25 ? "dry" : ""}">💧 ${Math.round(rp)}%</div>`;
    const el = document.createElement("div");
    el.className = "day";
    el.innerHTML = `
      <div class="dow">${i === 0 ? "Tomorrow" : fmtDay(day.date)}</div>
      <div class="date">${fmtDate(day.date)}</div>
      <div class="ic">${emoji}</div>
      <div class="hi">${Math.round(day.tmax)}°</div>
      <div class="lo">${Math.round(day.tmin)}°</div>
      ${rainHtml}`;
    row.appendChild(el);
  });
}

function renderChart(d) {
  const ctx = $("#chart").getContext("2d");
  const recent = (d.recent_actuals || []).slice(-10);
  const mine = d.my_forecast || [];
  const pro = d.pro_forecast || [];
  const labels = [...recent.map((r) => fmtDate(r.date)), ...mine.map((r) => fmtDate(r.date))];
  const pad = (arr, before, after) => [...Array(before).fill(null), ...arr, ...Array(after).fill(null)];
  const actualSeries = [...recent.map((r) => r.tmax), ...mine.map(() => null)];
  const mineSeries = pad(mine.map((r) => r.tmax), recent.length, 0);
  const proSeries = pad(mine.map((_, i) => (pro[i] ? pro[i].tmax : null)), recent.length, 0);
  const grid = "rgba(255,255,255,0.06)", tick = "#8b95bd";
  const mk = (label, data, color, opts = {}) => ({
    label, data, borderColor: color, backgroundColor: color + "22", tension: 0.35,
    spanGaps: false, pointRadius: 3, pointHoverRadius: 5, borderWidth: 2.5, ...opts,
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
        tooltip: { backgroundColor: "rgba(12,16,32,0.95)", borderColor: "rgba(255,255,255,0.12)", borderWidth: 1, padding: 12, callbacks: { label: (c) => `${c.dataset.label}: ${c.formattedValue}°` } },
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
    { label: "High-temp accuracy", num: `±${m.mae_tmax}°`, sub: "Typical error when predicting the daily high." },
    { label: "Low-temp accuracy", num: `±${m.mae_tmin}°`, sub: "Typical error when predicting the daily low." },
    {
      label: "vs. naïve guess",
      num: beatsBaseline ? `${beatTmax}° better` : `${Math.abs(beatTmax)}° worse`,
      sub: beatsBaseline ? `Beats “tomorrow = today” (±${m.baseline_tmax}°).` : `The “tomorrow = today” guess (±${m.baseline_tmax}°) is hard to beat here.`,
      good: beatsBaseline,
    },
  ];
  if (m.rain_accuracy != null) cards.push({ label: "Rain / no-rain", num: `${m.rain_accuracy}%`, sub: "Correct rain-vs-dry calls on unseen days." });
  cards.push({ label: "Trained on", num: `${m.history_years} yrs`, sub: `${m.train_days.toLocaleString()} days of history · tested on ${m.test_days.toLocaleString()} recent days.` });
  wrap.innerHTML = cards.map((c) => `
      <div class="metric">
        <div class="label">${c.label}</div>
        <div class="num">${c.num}</div>
        <div class="sub ${c.good ? "good" : ""}">${c.sub}</div>
      </div>`).join("");
}

// ---------- events ----------
$("#city").addEventListener("input", (e) => renderChips(e.target.value));
$("#search").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#city").value.trim().toLowerCase();
  if (!q) return;
  const match = CITIES.find((c) => c.name.toLowerCase().includes(q) || c.label.toLowerCase().includes(q));
  if (match) { $("#city").value = ""; renderChips(); selectCity(match.slug); }
  else showError(`“${$("#city").value.trim()}” isn't in our tracked list yet. Pick one of the cities above.`);
});
$("#chips").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-slug]");
  if (btn) selectCity(btn.dataset.slug);
});

init();
