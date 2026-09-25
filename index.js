// Burgers FNO web demo — no bundler, no framework. Talks to the wasm module
// built by build-for-web.sh (pkg/demo_web.js) and does its own plotting on a
// plain <canvas> — no charting library.

import init, { Fno1d, project_to_k_modes } from "./pkg/demo_web.js";

export function $(id) {
  return document.getElementById(id);
}

// Resolutions the slider can pick, as powers of two.
const RESOLUTIONS = [256, 512, 1024, 2048, 4096, 8192];
const TRAINING_RESOLUTION = 256;
// Base length ICs are constructed/sampled at before predict() resamples them
// to whatever resolution is selected. Matches training resolution, but the
// choice is arbitrary — predict() works from any power-of-two length.
const IC_BASE_LENGTH = 256;

// Fixed y-range covering the observed range of a(x)/u(x,T) in presets.json,
// with headroom. Used for the draw canvas and its axis.
const DRAW_Y_MIN = -1.5;
const DRAW_Y_MAX = 1.5;

let fno = null;
let presets = null;
let mode = "preset";
let resolutionIndex = 0; // index into RESOLUTIONS
let currentIc = null; // Float32Array, length IC_BASE_LENGTH
let currentPresetId = null; // set only in preset mode

const els = {};

function relativeL2(pred, truth) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < pred.length; i++) {
    const d = pred[i] - truth[i];
    num += d * d;
    den += truth[i] * truth[i];
  }
  return Math.sqrt(num) / Math.sqrt(den);
}

// --- Plain-canvas line plotting -------------------------------------------

/**
 * Draws one or more series on a canvas. Each series is `{ values, color,
 * label, style }` where `style` is "line" or "dots". X is always the series'
 * own uniform grid on [0, 1] (index / (length - 1)) — series of different
 * lengths still align correctly, since they share the same physical domain.
 * Y auto-scales to fit every series, with padding.
 */
function plot(canvas, series, { xLabel = "x", yLabel } = {}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = { left: 48, right: 16, top: 16, bottom: 32 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  let yMin = Infinity;
  let yMax = -Infinity;
  for (const s of series) {
    for (const v of s.values) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (!isFinite(yMin) || !isFinite(yMax)) {
    yMin = -1;
    yMax = 1;
  }
  if (yMax - yMin < 1e-6) {
    yMin -= 0.5;
    yMax += 0.5;
  }
  const yPad = (yMax - yMin) * 0.1;
  yMin -= yPad;
  yMax += yPad;

  const xToPx = (x) => pad.left + x * plotW;
  const yToPx = (y) => pad.top + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  // Axes + gridlines.
  ctx.strokeStyle = "#e2e2e2";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#888888";
  ctx.font = "11px system-ui, sans-serif";
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const y = yMin + ((yMax - yMin) * i) / yTicks;
    const py = yToPx(y);
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(pad.left + plotW, py);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(y.toFixed(2), pad.left - 6, py);
  }
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const x = i / xTicks;
    const px = xToPx(x);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(x.toFixed(1), px, pad.top + plotH + 6);
  }

  ctx.strokeStyle = "#333333";
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Zero line, if in range.
  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = "#cccccc";
    ctx.beginPath();
    ctx.moveTo(pad.left, yToPx(0));
    ctx.lineTo(pad.left + plotW, yToPx(0));
    ctx.stroke();
  }

  // Dots first, lines on top - a line series matching a dense dot series
  // near-exactly (e.g. prediction vs. ground truth) would otherwise vanish
  // completely under the dots, drawn last, rather than merely being close.
  const byStyle = (style) => series.filter((s) => s.style === style);

  for (const s of byStyle("dots")) {
    const n = s.values.length;
    ctx.fillStyle = s.color;
    // Thin to ~60 markers regardless of length - readable as "sampled
    // ground truth" rather than a solid swath at high resolution.
    const stride = Math.max(1, Math.round(n / 60));
    for (let i = 0; i < n; i += stride) {
      const px = xToPx(i / (n - 1));
      const py = yToPx(s.values[i]);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  for (const s of byStyle("line")) {
    const n = s.values.length;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = xToPx(i / (n - 1));
      const py = yToPx(s.values[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  if (yLabel) {
    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#888888";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
}

function renderLegend(entries) {
  els.legend.innerHTML = "";
  for (const { color, label } of entries) {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = color;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(label));
    els.legend.appendChild(item);
  }
}

// --- Prediction -------------------------------------------------------------

let predictSeq = 0;

async function runPrediction() {
  if (!fno || !currentIc) return;
  const seq = ++predictSeq;
  const n_eval = RESOLUTIONS[resolutionIndex];

  els.status.textContent = "computing…";
  const t0 = performance.now();
  const pred = await fno.predict(currentIc, n_eval);
  const dt = performance.now() - t0;
  if (seq !== predictSeq) return; // a newer prediction superseded this one

  const series = [
    { values: Array.from(currentIc), color: "#4a90d9", label: "a(x) — input", style: "line" },
    {
      values: Array.from(pred),
      color: "#d94a4a",
      label: `u(x, T) — prediction (n=${n_eval})`,
      style: "line",
    },
  ];

  let l2Text = "";
  if (mode === "preset" && currentPresetId !== null) {
    const truth = presets.presets[currentPresetId].u_true;
    series.push({
      values: truth,
      color: "#2ea36c",
      label: "u_true(x) — ground truth",
      style: "dots",
    });
    if (n_eval === TRAINING_RESOLUTION) {
      const l2 = relativeL2(Array.from(pred), truth);
      l2Text = ` — relative L2 vs ground truth: ${l2.toFixed(5)}`;
    } else {
      l2Text = ` — relative L2 only computable at training resolution (${TRAINING_RESOLUTION}); ground truth shown for shape comparison`;
    }
  }

  plot(els.plot, series);
  renderLegend(series.map((s) => ({ color: s.color, label: s.label })));
  els.status.textContent = `${dt.toFixed(1)} ms${l2Text}`;
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
const debouncedPredict = debounce(runPrediction, 60);

// --- Mode: preset -------------------------------------------------------------

function setupPresetMode() {
  presets.presets.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `Preset ${p.id}`;
    els.presetSelect.appendChild(opt);
  });

  els.presetSelect.addEventListener("change", () => {
    loadPreset(Number(els.presetSelect.value));
  });
}

function loadPreset(id) {
  currentPresetId = id;
  currentIc = Float32Array.from(presets.presets[id].a);
  runPrediction();
}

// --- Mode: sliders -------------------------------------------------------------

function sliderIc() {
  const k = Number(els.wavenumber.value);
  const amp = Number(els.amplitude.value);
  const ic = new Float32Array(IC_BASE_LENGTH);
  for (let i = 0; i < IC_BASE_LENGTH; i++) {
    const x = i / IC_BASE_LENGTH;
    ic[i] = amp * Math.sin(2 * Math.PI * k * x);
  }
  return ic;
}

function setupSliderMode() {
  const update = () => {
    els.wavenumberLabel.textContent = els.wavenumber.value;
    els.amplitudeLabel.textContent = Number(els.amplitude.value).toFixed(2);
    currentPresetId = null;
    currentIc = sliderIc();
    debouncedPredict();
  };
  els.wavenumber.addEventListener("input", update);
  els.amplitude.addEventListener("input", update);
}

// --- Mode: draw -------------------------------------------------------------

function setupDrawMode() {
  const canvas = els.drawCanvas;
  const ctx = canvas.getContext("2d");
  let samples = null; // Float32Array(IC_BASE_LENGTH) | null, sparse while drawing
  let drawing = false;

  function resetSamples() {
    samples = new Array(IC_BASE_LENGTH).fill(null);
  }
  resetSamples();

  function drawGrid() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#e2e2e2";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    ctx.strokeStyle = "#cfcfcf";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  }
  drawGrid();

  function pointerToSample(evt) {
    const rect = canvas.getBoundingClientRect();
    const px = evt.clientX - rect.left;
    const py = evt.clientY - rect.top;
    const x = Math.min(1, Math.max(0, px / canvas.width));
    const yFrac = Math.min(1, Math.max(0, py / canvas.height));
    const value = DRAW_Y_MAX - yFrac * (DRAW_Y_MAX - DRAW_Y_MIN);
    const bin = Math.min(IC_BASE_LENGTH - 1, Math.round(x * (IC_BASE_LENGTH - 1)));
    return { bin, value, px, py };
  }

  function redrawStroke() {
    drawGrid();
    ctx.strokeStyle = "#4a90d9";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < IC_BASE_LENGTH; i++) {
      if (samples[i] === null) continue;
      const px = (i / (IC_BASE_LENGTH - 1)) * canvas.width;
      const py = ((DRAW_Y_MAX - samples[i]) / (DRAW_Y_MAX - DRAW_Y_MIN)) * canvas.height;
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();
  }

  // Forward-fills each gap from the last drawn sample, then backward-fills
  // the leading gap (before the first stroke touched the canvas) from the
  // first drawn sample - so an incomplete drawing still yields a full-length,
  // gap-free array rather than silently padding with zeros.
  function fillGaps(sparse) {
    const filled = sparse.slice();
    let last = null;
    for (let i = 0; i < filled.length; i++) {
      if (filled[i] !== null) last = filled[i];
      else if (last !== null) filled[i] = last;
    }
    const first = filled.find((v) => v !== null);
    for (let i = 0; i < filled.length && filled[i] === null; i++) {
      filled[i] = first;
    }
    return filled;
  }

  async function commitDrawing() {
    if (samples.every((v) => v === null)) return;
    const filled = fillGaps(samples);
    const ic = Float32Array.from(filled);
    els.status.textContent = "projecting onto training-distribution modes…";
    const projected = await project_to_k_modes(ic);
    currentPresetId = null;
    currentIc = Float32Array.from(projected);
    runPrediction();
  }

  function handleMove(evt) {
    if (!drawing) return;
    const { bin, value } = pointerToSample(evt);
    samples[bin] = value;
    redrawStroke();
  }

  canvas.addEventListener("pointerdown", (evt) => {
    drawing = true;
    canvas.setPointerCapture(evt.pointerId);
    handleMove(evt);
  });
  canvas.addEventListener("pointermove", handleMove);
  canvas.addEventListener("pointerup", () => {
    drawing = false;
    commitDrawing();
  });

  els.clearDraw.addEventListener("click", () => {
    resetSamples();
    drawGrid();
  });
}

// --- Mode switching + resolution slider --------------------------------------

function setMode(next) {
  mode = next;
  for (const name of ["preset", "sliders", "draw"]) {
    els.panels[name].classList.toggle("hidden", name !== next);
    els.tabs[name].classList.toggle("active", name === next);
  }
  if (next === "preset" && els.presetSelect.value !== "") {
    loadPreset(Number(els.presetSelect.value));
  } else if (next === "sliders") {
    currentPresetId = null;
    currentIc = sliderIc();
    runPrediction();
  }
  // "draw" mode: leave the canvas as-is; predicts on stroke release.
}

function setupResolutionSlider() {
  els.resolution.min = 0;
  els.resolution.max = RESOLUTIONS.length - 1;
  els.resolution.value = resolutionIndex;
  els.resolutionLabel.textContent = RESOLUTIONS[resolutionIndex];

  els.resolution.addEventListener("input", () => {
    resolutionIndex = Number(els.resolution.value);
    els.resolutionLabel.textContent = RESOLUTIONS[resolutionIndex];
    debouncedPredict();
  });
}

// --- Boot -------------------------------------------------------------------

async function boot() {
  els.status = $("status");
  els.plot = $("plot");
  els.legend = $("legend");
  els.presetSelect = $("preset-select");
  els.wavenumber = $("wavenumber");
  els.wavenumberLabel = $("wavenumber-label");
  els.amplitude = $("amplitude");
  els.amplitudeLabel = $("amplitude-label");
  els.drawCanvas = $("draw-canvas");
  els.clearDraw = $("clear-draw");
  els.resolution = $("resolution");
  els.resolutionLabel = $("resolution-label");
  els.tabs = { preset: $("tab-preset"), sliders: $("tab-sliders"), draw: $("tab-draw") };
  els.panels = { preset: $("panel-preset"), sliders: $("panel-sliders"), draw: $("panel-draw") };

  for (const name of Object.keys(els.tabs)) {
    els.tabs[name].addEventListener("click", () => setMode(name));
  }

  setupResolutionSlider();
  setupSliderMode();
  setupDrawMode();

  els.status.textContent = "loading model…";
  const [_wasm, presetsResp] = await Promise.all([init(), fetch("./presets.json")]);
  presets = await presetsResp.json();
  fno = new Fno1d();

  setupPresetMode();
  els.presetSelect.value = "0";
  setMode("preset");
}

boot();
