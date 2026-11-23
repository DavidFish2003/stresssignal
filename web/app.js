// StressSignal dashboard: reads data/results.json produced by `python -m stresssignal.train`.
const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (x) => `${Math.round(x * 100)}%`;
const charts = {};
let R, sig = "hr";

init();

async function init() {
  try {
    R = await (await fetch("data/results.json", { cache: "no-store" })).json();
  } catch {
    $("main").innerHTML = '<p class="muted">The results file could not be loaded.</p>';
    return;
  }
  const main = R.models["Gradient boosting (personal baseline)"];
  const d = R.dataset;
  $("#stats").textContent = `${d.people} people · ${d.windows.toLocaleString()} one-minute readings · every result tested on people the model never saw`;
  $("#tiles").innerHTML = [
    [main.auroc.mean.toFixed(2), "Detection score on new people (0.5 = chance, 1.0 = perfect)"],
    [pct(main.bal_acc.mean), "Stress and rest correctly told apart"],
    [pct(1 - R.false_alarm_rate), "Rest minutes correctly left unflagged"],
    [`${d.baseline_min_median} min`, "Rest needed to calibrate to a person"],
  ].map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join("");

  // Default to the person whose score is closest to the group median: typical, not cherry-picked.
  const sorted = [...R.people].sort((a, b) => a.auroc - b.auroc);
  const typical = sorted[Math.floor(sorted.length / 2)].id;
  $("#person").innerHTML = R.people.map((p) =>
    `<option value="${p.id}" ${p.id === typical ? "selected" : ""}>Participant ${p.id.slice(1)} · score ${p.auroc.toFixed(2)}${p.id === typical ? " (typical)" : ""}</option>`).join("");
  $("#person").addEventListener("change", renderPerson);
  document.querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => {
    sig = b.dataset.sig;
    document.querySelectorAll(".pill").forEach((x) => x.setAttribute("aria-pressed", x === b));
    renderPerson();
  }));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  try { await document.fonts.ready; } catch {}
  renderAll();
}

function base() {
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = css("--text-2");
}
const tooltip = () => ({ backgroundColor: css("--surface"), titleColor: css("--text"), bodyColor: css("--text-2"),
  borderColor: css("--line"), borderWidth: 1, padding: 10, displayColors: false });
const grid = () => ({ color: css("--grid") });
function draw(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($("#" + id), cfg); }

// Shaded task bands drawn behind the lines.
const bands = {
  id: "bands",
  beforeDatasetsDraw(chart, _, opts) {
    const { ctx, chartArea: a, scales: { x } } = chart;
    ctx.save();
    for (const t of opts.tasks || []) {
      const x0 = x.getPixelForValue(t.start), x1 = x.getPixelForValue(t.end);
      ctx.fillStyle = css("--band");
      ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
      if (opts.labels) {
        ctx.fillStyle = css("--text-2");
        ctx.font = "11px Inter, system-ui, sans-serif";
        let label = innerWidth < 640 ? t.name.split(" ")[0] : t.name;
        const room = a.right - x0 - 6; // keep labels inside the plot
        while (label.length > 3 && ctx.measureText(label).width > room) label = label.slice(0, -2) + "…";
        ctx.fillText(label, x0 + 4, a.top + 13);
      }
    }
    ctx.restore();
  },
};

function renderAll() { base(); renderPerson(); renderSensors(); renderImportance(); renderModels(); renderTasks(); }

function renderPerson() {
  const p = R.people.find((x) => x.id === $("#person").value);
  $("#legend").innerHTML = `<span><i style="background:${css("--accent")}"></i>Stress estimate</span><span><i style="background:${css("--band")}"></i>Stress task</span>`;
  const pts = p.t.map((t, i) => ({ x: t, y: p.p[i] }));
  const xmax = Math.ceil(p.t[p.t.length - 1]);
  const xScale = { type: "linear", min: 0, max: xmax, grid: grid(), border: { display: false }, title: { display: true, text: "Minutes from start", color: css("--text-3") } };
  draw("c-time", {
    type: "line",
    data: { datasets: [{ data: pts, borderColor: css("--accent"), borderWidth: 2, pointRadius: 0, pointHitRadius: 8, tension: 0.25, spanGaps: true }] },
    options: {
      maintainAspectRatio: false, animation: { duration: 200 }, interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: { legend: { display: false }, bands: { tasks: p.tasks, labels: true },
        tooltip: { ...tooltip(), callbacks: { title: (it) => `Minute ${it[0].parsed.x.toFixed(1)}`, label: (it) => `Stress estimate ${pct(it.parsed.y)}` } } },
      scales: { x: xScale, y: { min: 0, max: 1, grid: grid(), border: { display: false }, ticks: { callback: (v) => pct(v), stepSize: 0.25 } } },
    },
    plugins: [bands],
  });
  const vals = p[sig];
  const unit = sig === "hr" ? "bpm" : "µS";
  draw("c-sig", {
    type: "line",
    data: { datasets: [{ data: p.t.map((t, i) => ({ x: t, y: vals[i] })), borderColor: css("--text-2"), borderWidth: 1.5, pointRadius: 0, pointHitRadius: 8, tension: 0.25, spanGaps: true }] },
    options: {
      maintainAspectRatio: false, animation: { duration: 200 }, interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: { legend: { display: false }, bands: { tasks: p.tasks, labels: false },
        tooltip: { ...tooltip(), callbacks: { title: (it) => `Minute ${it[0].parsed.x.toFixed(1)}`, label: (it) => `${it.parsed.y} ${unit}` } } },
      scales: { x: { ...xScale, title: { display: false } }, y: { grid: grid(), border: { display: false }, title: { display: true, text: unit, color: css("--text-3") } } },
    },
    plugins: [bands],
  });
}

function hbar(id, labels, values, { max, fmt = (v) => v, tip }) {
  draw(id, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: css("--accent"), borderRadius: 4, borderSkipped: "start", barThickness: 14 }] },
    options: {
      indexAxis: "y", maintainAspectRatio: false, animation: { duration: 200 }, layout: { padding: { left: 4, right: 8 } },
      plugins: { legend: { display: false }, tooltip: { ...tooltip(), callbacks: { label: tip || ((it) => fmt(it.raw)) } } },
      scales: { x: { min: 0, max, grid: grid(), border: { display: false }, ticks: { callback: fmt } }, y: { grid: { display: false }, border: { display: false }, ticks: { autoSkip: false } } },
    },
  });
}

function renderSensors() {
  const order = ["All sensors", "All except movement", "Skin conductance", "Heart", "Skin temperature", "Movement"];
  const keys = order.filter((k) => R.sensors[k]);
  hbar("c-sensor", keys, keys.map((k) => R.sensors[k].mean), { max: 1, fmt: (v) => Number(v).toFixed(2),
    tip: (it) => `Score ${it.raw.toFixed(2)} (± ${R.sensors[keys[it.dataIndex]].sd.toFixed(2)} across people)` });
}

function renderImportance() {
  const top = R.importance.slice(0, 8);
  hbar("c-imp", top.map((d) => d.feature), top.map((d) => Math.max(0, d.importance)), { fmt: (v) => Number(v).toFixed(3),
    tip: (it) => [`Score drop when scrambled: ${it.raw.toFixed(3)}`, `Sensor: ${top[it.dataIndex].sensor}`] });
}

function renderModels() {
  const names = Object.keys(R.models);
  const short = (n) => ({
    "Gradient boosting (personal baseline)": "Main model, calibrated",
    "Logistic regression (personal baseline)": "Simpler model, calibrated",
    "Gradient boosting (no baseline)": "Main model, not calibrated",
  })[n] || n;
  hbar("c-models", names.map(short), names.map((n) => R.models[n].auroc.mean), { max: 1, fmt: (v) => Number(v).toFixed(2),
    tip: (it) => `Score ${it.raw.toFixed(2)} · ${pct(R.models[names[it.dataIndex]].bal_acc.mean)} told apart correctly` });
}

function renderTasks() {
  const labels = [...Object.keys(R.by_task), "False alarms during rest"];
  const values = [...Object.values(R.by_task), R.false_alarm_rate];
  hbar("c-task", labels, values, { max: 1, fmt: (v) => pct(v) });
}
