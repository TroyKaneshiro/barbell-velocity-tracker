'use strict';

// ── Element refs ────────────────────────────────────────────────
const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const browseBtn    = document.getElementById('browseBtn');
const dropContent  = document.getElementById('dropContent');
const fileChosen   = document.getElementById('fileChosen');
const fileNameEl   = document.getElementById('fileName');
const clearFileBtn = document.getElementById('clearFile');
const analyzeBtn   = document.getElementById('analyzeBtn');
const progressWrap = document.getElementById('progressWrap');
const progressLabel= document.getElementById('progressLabel');
const errorBox     = document.getElementById('errorBox');
const errorMsg     = document.getElementById('errorMsg');
const results      = document.getElementById('results');

const plateSelect = document.getElementById('plateSelect');

// Result fields
const rpeValue  = document.getElementById('rpeValue');
const rpeDesc   = document.getElementById('rpeDesc');
const rpeNote   = document.getElementById('rpeNote');
const mcvValue  = document.getElementById('mcvValue');
const peakValue = document.getElementById('peakValue');
const liftValue = document.getElementById('liftValue');
const plateValue= document.getElementById('plateValue');
const calibValue= document.getElementById('calibValue');

// ── Populate plate dropdown from backend ─────────────────────────
fetch('/plates')
  .then((r) => r.json())
  .then(({ plates, default: def }) => {
    plateSelect.innerHTML = plates
      .map((p) => `<option value="${p}"${p === def ? ' selected' : ''}>${p}</option>`)
      .join('');
  })
  .catch(() => { /* keep the static fallback option already in the HTML */ });

let selectedFile = null;
let velocityChart = null;

// ── File selection ───────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
  if (e.target !== browseBtn && e.target !== clearFileBtn) fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) setFile(file);
});

clearFileBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});

function setFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  dropContent.classList.add('hidden');
  fileChosen.classList.remove('hidden');
  analyzeBtn.disabled = false;
  hideError();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  dropContent.classList.remove('hidden');
  fileChosen.classList.add('hidden');
  analyzeBtn.disabled = true;
  hideError();
}

// ── Analyse ──────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) return;

  const liftType = document.querySelector('input[name="lift"]:checked').value;

  hideError();
  results.classList.add('hidden');
  progressWrap.classList.remove('hidden');
  progressLabel.textContent = 'Tracking bar & computing velocity…';
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append('video', selectedFile);
  formData.append('lift_type', liftType);
  formData.append('plate', plateSelect.value);

  fetch('/analyze', { method: 'POST', body: formData })
    .then((res) =>
      res.text().then((text) => {
        let body;
        try { body = JSON.parse(text); } catch { body = { detail: text }; }
        if (!res.ok) throw new Error(body.detail || `Server error ${res.status}`);
        return body;
      })
    )
    .then(showResults)
    .catch((err) => showError(err.message))
    .finally(() => {
      progressWrap.classList.add('hidden');
      analyzeBtn.disabled = false;
    });
});

// ── Render results ───────────────────────────────────────────────
function showResults(data) {
  rpeValue.textContent  = data.rpe;
  rpeDesc.textContent   = data.rpe_description;

  if (data.rpe_note) {
    rpeNote.textContent = data.rpe_note;
    rpeNote.classList.remove('hidden');
  } else {
    rpeNote.classList.add('hidden');
  }

  mcvValue.textContent  = data.mean_concentric_velocity.toFixed(3);
  peakValue.textContent = data.peak_concentric_velocity.toFixed(3);
  liftValue.textContent  = data.lift_type.charAt(0).toUpperCase() + data.lift_type.slice(1);
  plateValue.textContent = `${data.calibration.plate} (${(data.calibration.plate_diameter_m * 1000).toFixed(0)} mm)`;
  calibValue.textContent = `${data.calibration.fps.toFixed(0)} fps · ${data.calibration.m_per_px.toFixed(4)} m/px`;

  drawChart(data);

  results.classList.remove('hidden');
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function drawChart(data) {
  const ctx = document.getElementById('velocityChart').getContext('2d');

  if (velocityChart) velocityChart.destroy();

  const labels = data.time.map((t) => t.toFixed(2));
  const vel    = data.velocity;
  const eStart = data.eccentric_start;
  const cStart = data.concentric_start;
  const cEnd   = data.concentric_end;
  const mcv    = data.mean_concentric_velocity;

  // Background highlights: blue = eccentric, red = concentric
  const phasePlugin = {
    id: 'phaseRegions',
    beforeDraw(chart) {
      const { ctx: c, chartArea, scales } = chart;
      if (!chartArea) return;
      const xScale = scales.x;

      c.save();

      // Eccentric phase (bar going down)
      const ex1 = xScale.getPixelForValue(eStart);
      const ex2 = xScale.getPixelForValue(cStart);
      c.fillStyle = 'rgba(100,149,237,0.10)';
      c.fillRect(ex1, chartArea.top, ex2 - ex1, chartArea.bottom - chartArea.top);

      // Concentric phase (bar going up)
      const cx1 = xScale.getPixelForValue(cStart);
      const cx2 = xScale.getPixelForValue(cEnd - 1);
      c.fillStyle = 'rgba(230,57,70,0.12)';
      c.fillRect(cx1, chartArea.top, cx2 - cx1, chartArea.bottom - chartArea.top);

      c.restore();
    },
  };

  velocityChart = new Chart(ctx, {
    type: 'line',
    plugins: [phasePlugin],
    data: {
      labels,
      datasets: [
        {
          label: 'Velocity (m/s)',
          data: vel,
          borderColor: '#e8eaf0',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: `MCV ${mcv} m/s`,
          data: Array(vel.length).fill(mcv),
          borderColor: '#e63946',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          label: 'Zero',
          data: Array(vel.length).fill(0),
          borderColor: '#2a2d3e',
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#7a7f99',
            font: { size: 12 },
            boxWidth: 20,
          },
        },
        tooltip: {
          backgroundColor: '#1a1d27',
          borderColor: '#2a2d3e',
          borderWidth: 1,
          titleColor: '#e8eaf0',
          bodyColor: '#7a7f99',
          callbacks: {
            title: (items) => `t = ${items[0].label}s`,
            label: (item) => ` ${item.dataset.label.split(' ')[0]}: ${Number(item.raw).toFixed(3)} m/s`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#7a7f99',
            maxTicksLimit: 10,
            callback: (v, i) => (i % Math.ceil(labels.length / 10) === 0 ? labels[i] + 's' : ''),
          },
          grid: { color: '#2a2d3e' },
          title: { display: true, text: 'Time (s)', color: '#7a7f99' },
        },
        y: {
          ticks: { color: '#7a7f99' },
          grid: { color: '#2a2d3e' },
          title: { display: true, text: 'Velocity (m/s)', color: '#7a7f99' },
        },
      },
    },
  });
}

// ── Helpers ──────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  errorBox.classList.add('hidden');
}
