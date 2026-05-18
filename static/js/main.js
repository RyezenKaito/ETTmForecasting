/* ETTm1 Demo – main.js */

// ── Tab switcher ──────────────────────────────────────────────────────────────
function initTabs(containerSel) {
  const container = document.querySelector(containerSel);
  if (!container) return;
  const btns   = container.querySelectorAll('.tab-btn');
  const panels = container.querySelectorAll('.tab-panel');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.target);
      if (target) target.classList.add('active');
    });
  });
}

// ── Visualize page ────────────────────────────────────────────────────────────
function initVizPage() {
  const modelSel  = document.getElementById('modelSelect');
  const idxSlider = document.getElementById('idxSlider');
  const idxLabel  = document.getElementById('idxLabel');
  const runBtn    = document.getElementById('runBtn');
  const chartDiv  = document.getElementById('chartContainer');
  const metricsDiv= document.getElementById('metricsContainer');
  const spinner   = document.getElementById('spinner');

  if (!modelSel) return;

  idxSlider.addEventListener('input', () => {
    idxLabel.textContent = idxSlider.value;
  });

  async function runPrediction() {
    const model = modelSel.value;
    const idx   = parseInt(idxSlider.value);
    spinner.style.display  = 'block';
    chartDiv.innerHTML     = '';
    metricsDiv.innerHTML   = '';

    try {
      const res  = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model, index: idx}),
      });
      const data = await res.json();
      spinner.style.display = 'none';

      if (data.error) {
        chartDiv.innerHTML = `<p style="color:var(--danger)">${data.error}</p>`;
        return;
      }

      // chart
      chartDiv.innerHTML = `<img src="data:image/png;base64,${data.chart}" alt="prediction chart" class="fade-in">`;

      // metrics
      const m = data.metrics;
      metricsDiv.innerHTML = `
        <div class="metric-mini-grid fade-in">
          <div class="metric-mini"><div class="val">${m.MSE.toFixed(4)}</div><div class="lbl">MSE °C²</div></div>
          <div class="metric-mini"><div class="val">${m.RMSE.toFixed(4)}</div><div class="lbl">RMSE °C</div></div>
          <div class="metric-mini"><div class="val">${m.MAE.toFixed(4)}</div><div class="lbl">MAE °C</div></div>
          <div class="metric-mini"><div class="val">${m['sMAPE%'].toFixed(2)}%</div><div class="lbl">sMAPE</div></div>
        </div>`;
    } catch(e) {
      spinner.style.display = 'none';
      chartDiv.innerHTML = `<p style="color:var(--danger)">Error: ${e.message}</p>`;
    }
  }

  runBtn.addEventListener('click', runPrediction);
  // auto-run first prediction
  runPrediction();
}

// ── Weight table collapsible ──────────────────────────────────────────────────
function initWeightTables() {
  document.querySelectorAll('.weight-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const tbody = document.getElementById(btn.dataset.target);
      if (!tbody) return;
      const isHidden = tbody.style.display === 'none';
      tbody.style.display = isHidden ? '' : 'none';
      btn.textContent     = isHidden ? '▲ Collapse' : '▼ Expand';
    });
  });
}

// ── Numbers counter animation ─────────────────────────────────────────────────
function animateCounters() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count);
    const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 4;
    let start = null;
    const duration = 800;
    const step = ts => {
      if (!start) start = ts;
      const prog = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - prog, 3);
      el.textContent = (target * ease).toFixed(decimals);
      if (prog < 1) requestAnimationFrame(step);
      else el.textContent = target.toFixed(decimals);
    };
    requestAnimationFrame(step);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs('.tabs-container');
  initVizPage();
  initWeightTables();
  animateCounters();

  // Mark active nav link
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === path ||
        (path !== '/' && a.getAttribute('href') !== '/' && path.startsWith(a.getAttribute('href')))) {
      a.classList.add('active');
    }
  });
});
