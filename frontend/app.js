'use strict';

// ── Element refs ────────────────────────────────────────────────
const dropZone          = document.getElementById('dropZone');
const fileInput         = document.getElementById('fileInput');
const browseBtn         = document.getElementById('browseBtn');
const dropContent       = document.getElementById('dropContent');
const fileChosen        = document.getElementById('fileChosen');
const fileNameEl        = document.getElementById('fileName');
const clearFileBtn      = document.getElementById('clearFile');
const analyzeBtn        = document.getElementById('analyzeBtn');
const progressWrap      = document.getElementById('progressWrap');
const progressLabel     = document.getElementById('progressLabel');
const errorBox          = document.getElementById('errorBox');
const errorMsg          = document.getElementById('errorMsg');
const results           = document.getElementById('results');
const clickWrap         = document.getElementById('clickWrap');
const previewVideo      = document.getElementById('previewVideo');
const clickCanvas       = document.getElementById('clickCanvas');
const clickStatus       = document.getElementById('clickStatus');

const plateSelect = document.getElementById('plateSelect');
const barWeight   = document.getElementById('barWeight');

// ── Click state ─────────────────────────────────────────────────
let plateClick     = null;   // { normX, normY } in 0-1 coords
let detectedCircle = null;   // { cx_norm, cy_norm, r_norm } from /detect-plate, or null
let manualEdge     = null;   // { normX, normY } right-click edge point, or null

// Result fields
const rpeValue  = document.getElementById('rpeValue');
const rpeDesc   = document.getElementById('rpeDesc');
const rpeNote   = document.getElementById('rpeNote');
const mcvValue  = document.getElementById('mcvValue');
const peakValue = document.getElementById('peakValue');
const liftValue = document.getElementById('liftValue');
const plateValue= document.getElementById('plateValue');
const calibValue= document.getElementById('calibValue');
const platePxValue = document.getElementById('platePxValue');
const rmRow      = document.getElementById('rmRow');
const rmValue    = document.getElementById('rmValue');
const rmUnit     = document.getElementById('rmUnit');
const pctRow     = document.getElementById('pctRow');
const pctValue   = document.getElementById('pctValue');
const debugCard  = document.getElementById('debugCard');
const debugVideo = document.getElementById('debugVideo');

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
  hideError();

  // Show preview and ask user to click the plate
  plateClick     = null;
  detectedCircle = null;
  manualEdge     = null;
  clickStatus.textContent = 'No plate selected';
  analyzeBtn.disabled = true;
  previewVideo.src = URL.createObjectURL(file);
  previewVideo.currentTime = 0;
  clickWrap.classList.remove('hidden');

  // Size canvas to match video once metadata loads
  previewVideo.addEventListener('loadedmetadata', syncCanvas, { once: true });
}

function syncCanvas() {
  clickCanvas.width  = previewVideo.videoWidth;
  clickCanvas.height = previewVideo.videoHeight;
  redrawCanvas();
}

function _drawCircle(ctx, cx, cy, r, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth   = Math.max(2, r * 0.05);
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(3, r * 0.06), 0, Math.PI * 2);
  ctx.fill();
}

function _manualRadius() {
  if (!plateClick || !manualEdge) return null;
  const cx = plateClick.normX * clickCanvas.width;
  const cy = plateClick.normY * clickCanvas.height;
  const ex = manualEdge.normX * clickCanvas.width;
  const ey = manualEdge.normY * clickCanvas.height;
  return Math.hypot(ex - cx, ey - cy);
}

function redrawCanvas() {
  const ctx = clickCanvas.getContext('2d');
  ctx.clearRect(0, 0, clickCanvas.width, clickCanvas.height);
  if (!plateClick) return;

  const manualR = _manualRadius();

  if (manualR !== null) {
    // Manual circle defined by left-click center + right-click edge — draw in orange
    const cx = plateClick.normX * clickCanvas.width;
    const cy = plateClick.normY * clickCanvas.height;
    _drawCircle(ctx, cx, cy, manualR, '#ff8c42');

    // Edge point dot
    ctx.fillStyle = '#ff8c42';
    ctx.beginPath();
    ctx.arc(manualEdge.normX * clickCanvas.width, manualEdge.normY * clickCanvas.height, 5, 0, Math.PI * 2);
    ctx.fill();
  } else if (detectedCircle) {
    // Hough-detected circle — draw in green
    const cx = detectedCircle.cx_norm * clickCanvas.width;
    const cy = detectedCircle.cy_norm * clickCanvas.height;
    const r  = detectedCircle.r_norm  * Math.min(clickCanvas.width, clickCanvas.height);
    _drawCircle(ctx, cx, cy, r, '#43d982');
  } else {
    // Fallback: crosshair at click while detection in flight or failed
    const x = plateClick.normX * clickCanvas.width;
    const y = plateClick.normY * clickCanvas.height;
    const r = Math.min(clickCanvas.width, clickCanvas.height) * 0.04;
    ctx.strokeStyle = '#f9c74f';
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - r * 1.5, y); ctx.lineTo(x + r * 1.5, y);
    ctx.moveTo(x, y - r * 1.5); ctx.lineTo(x, y + r * 1.5);
    ctx.stroke();
  }
}

clickCanvas.addEventListener('click', (e) => {
  const rect   = clickCanvas.getBoundingClientRect();
  const scaleX = clickCanvas.width  / rect.width;
  const scaleY = clickCanvas.height / rect.height;
  const px     = (e.clientX - rect.left)  * scaleX;
  const py     = (e.clientY - rect.top)   * scaleY;

  plateClick     = { normX: px / clickCanvas.width, normY: py / clickCanvas.height };
  detectedCircle = null;
  manualEdge     = null;
  analyzeBtn.disabled = false;
  clickStatus.textContent = 'Detecting plate…';
  redrawCanvas();

  const fd = new FormData();
  fd.append('video',   selectedFile);
  fd.append('click_x', plateClick.normX);
  fd.append('click_y', plateClick.normY);

  fetch('/detect-plate', { method: 'POST', body: fd })
    .then((r) => r.json())
    .then((data) => {
      if (data.found) {
        detectedCircle = data;
        clickStatus.textContent = 'Plate detected — right-click the outer edge to correct, or left-click to reposition';
      } else {
        clickStatus.textContent = 'No circle found — right-click the outer edge of the plate to set it manually';
      }
      redrawCanvas();
    })
    .catch(() => {
      clickStatus.textContent = 'Detection failed — right-click the outer edge to set manually';
      redrawCanvas();
    });
});

clickCanvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (!plateClick) return;

  const rect   = clickCanvas.getBoundingClientRect();
  const scaleX = clickCanvas.width  / rect.width;
  const scaleY = clickCanvas.height / rect.height;
  const px     = (e.clientX - rect.left)  * scaleX;
  const py     = (e.clientY - rect.top)   * scaleY;

  manualEdge     = { normX: px / clickCanvas.width, normY: py / clickCanvas.height };
  detectedCircle = null;

  const r = _manualRadius();
  clickStatus.textContent = `Manual circle set (r=${r.toFixed(0)}px) — right-click again to adjust`;
  redrawCanvas();
});

function clearFile() {
  selectedFile   = null;
  plateClick     = null;
  detectedCircle = null;
  manualEdge     = null;
  fileInput.value = '';
  dropContent.classList.remove('hidden');
  fileChosen.classList.add('hidden');
  clickWrap.classList.add('hidden');
  previewVideo.src = '';
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

  if (!plateClick) {
    progressWrap.classList.add('hidden');
    analyzeBtn.disabled = false;
    showError('Please click on the weight plate in the preview first.');
    return;
  }

  const formData = new FormData();
  formData.append('video', selectedFile);
  formData.append('lift_type', liftType);
  formData.append('plate', plateSelect.value);
  formData.append('click_x', plateClick.normX);
  formData.append('click_y', plateClick.normY);

  const manualR = _manualRadius();
  if (manualR !== null) {
    formData.append('plate_r_norm', manualR / Math.min(clickCanvas.width, clickCanvas.height));
  }

  const w = parseFloat(barWeight.value);
  if (!isNaN(w) && w > 0) formData.append('bar_weight', w);

  fetch('/analyze', { method: 'POST', body: formData })
    .then((res) =>
      res.text().then((text) => {
        if (!res.ok) {
          // Show the raw server response so we can see exactly what failed
          throw new Error(`Server ${res.status}: ${text}`);
        }
        let body;
        try { body = JSON.parse(text); } catch { throw new Error(text); }
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

  // 1RM projection
  if (data.projected_1rm != null) {
    const unit = barWeight.value ? (barWeight.placeholder.includes('kg') ? 'kg' : '') : '';
    rmValue.textContent = data.projected_1rm;
    rmUnit.textContent  = unit || (plateSelect.value.includes('kg') ? 'kg' : 'lb');
    pctValue.textContent = (data.percent_1rm * 100).toFixed(1);
    rmRow.classList.remove('hidden');
    pctRow.classList.remove('hidden');
  } else {
    rmRow.classList.add('hidden');
    pctRow.classList.add('hidden');
  }

  mcvValue.textContent  = data.mean_concentric_velocity.toFixed(3);
  peakValue.textContent = data.peak_concentric_velocity.toFixed(3);
  liftValue.textContent  = data.lift_type.charAt(0).toUpperCase() + data.lift_type.slice(1);
  plateValue.textContent = `${data.calibration.plate} (${(data.calibration.plate_diameter_m * 1000).toFixed(0)} mm)`;
  calibValue.textContent = `${data.calibration.fps.toFixed(0)} fps · ${data.calibration.m_per_px.toFixed(4)} m/px`;
  platePxValue.textContent = data.calibration.plate_diameter_px.toFixed(1);

  drawChart(data);

  // Debug video — conversion runs in background; poll until the file is ready
  if (data.debug_video_url) {
    debugCard.style.display = '';
    debugVideo.innerHTML = '';
    debugVideo.removeAttribute('src');

    const pollInterval = 1500;  // ms between attempts
    const maxAttempts  = 40;    // ~60s total
    let   attempts     = 0;

    function tryLoadDebugVideo() {
      const url = data.debug_video_url + '?t=' + Date.now();
      fetch(url, { method: 'HEAD' })
        .then((r) => {
          if (r.ok) {
            debugVideo.innerHTML = '';
            const src = document.createElement('source');
            src.src  = url;
            src.type = 'video/mp4';
            debugVideo.appendChild(src);
            debugVideo.load();
          } else if (++attempts < maxAttempts) {
            setTimeout(tryLoadDebugVideo, pollInterval);
          }
        })
        .catch(() => { if (++attempts < maxAttempts) setTimeout(tryLoadDebugVideo, pollInterval); });
    }
    tryLoadDebugVideo();
  }

  results.classList.remove('hidden');
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function drawChart(data) {
  const ctx = document.getElementById('velocityChart').getContext('2d');

  if (velocityChart) velocityChart.destroy();

  const labels = data.time.map((t) => t.toFixed(2));
  const vel    = data.velocity;
  const eStart  = data.eccentric_start;
  const cStart  = data.concentric_start;
  const cEnd    = data.concentric_end;
  const bStart  = data.burst_start;
  const bEnd    = data.burst_end;
  const bThresh = data.burst_threshold;
  const mcv     = data.mean_concentric_velocity;

  // Background highlights: blue = eccentric, red = concentric, green = burst (MCV window)
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
      c.fillStyle = 'rgba(230,57,70,0.10)';
      c.fillRect(cx1, chartArea.top, cx2 - cx1, chartArea.bottom - chartArea.top);

      // Burst window — frames used for MCV (green)
      const bx1 = xScale.getPixelForValue(bStart);
      const bx2 = xScale.getPixelForValue(bEnd - 1);
      c.fillStyle = 'rgba(67,217,130,0.18)';
      c.fillRect(bx1, chartArea.top, bx2 - bx1, chartArea.bottom - chartArea.top);

      // Burst threshold line
      const yScale = scales.y;
      const ty = yScale.getPixelForValue(bThresh);
      c.strokeStyle = 'rgba(67,217,130,0.50)';
      c.lineWidth = 1;
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(bx1, ty);
      c.lineTo(bx2, ty);
      c.stroke();
      c.setLineDash([]);

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
          label: `Burst threshold ${bThresh.toFixed(3)} m/s`,
          data: Array(vel.length).fill(bThresh),
          borderColor: 'rgba(67,217,130,0.6)',
          borderWidth: 1,
          borderDash: [3, 3],
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
