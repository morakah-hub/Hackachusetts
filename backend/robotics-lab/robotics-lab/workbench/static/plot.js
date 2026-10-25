// Minimal canvas line plots (no external libraries). NaN/null values break a line.
"use strict";
const Plot = (() => {
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= step0) || mag * 10;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function fmt(v) { const a = Math.abs(v); return a >= 1000 || (a < 0.01 && a > 0) ? v.toExponential(1) : +v.toPrecision(3) + ""; }

  // opts: {title, series:[{x, y, color, width, dash, label}], xlabel, ylabel, cursor, bands:[{x0, x1, color}],
  //        ymin, ymax, onClick(x)}
  function draw(canvas, opts) {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return;
    if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const L = 46, R = 8, T = 20, B = 22, pw = W - L - R, ph = H - T - B;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of opts.series) for (let i = 0; i < s.x.length; i++) {
      const x = s.x[i], y = s.y[i]; if (x == null || y == null || !isFinite(y)) continue;
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    ctx.fillStyle = "#8b9aab"; ctx.font = "11px Segoe UI, sans-serif";
    ctx.fillText(opts.title || "", L, 13);
    if (!isFinite(xmin)) { ctx.fillText("no data", L + 10, T + ph / 2); return; }
    if (opts.xmin != null) xmin = opts.xmin; if (opts.xmax != null) xmax = opts.xmax;
    if (opts.ymin != null) ymin = Math.min(ymin, opts.ymin); if (opts.ymax != null) ymax = Math.max(ymax, opts.ymax);
    if (ymax - ymin < 1e-12) { ymin -= 1; ymax += 1; }
    const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad; if (xmax === xmin) xmax = xmin + 1;
    const X = x => L + (x - xmin) / (xmax - xmin) * pw, Y = y => T + (1 - (y - ymin) / (ymax - ymin)) * ph;
    canvas._xform = { X, L, pw, xmin, xmax };
    for (const b of opts.bands || []) { ctx.fillStyle = b.color; ctx.fillRect(X(b.x0), T, Math.max(1, X(b.x1) - X(b.x0)), ph); }
    ctx.strokeStyle = "#2a3542"; ctx.lineWidth = 1; ctx.fillStyle = "#8b9aab";
    for (const v of niceTicks(ymin, ymax, 4)) { const y = Y(v); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke(); ctx.fillText(fmt(v), 2, y + 3); }
    for (const v of niceTicks(xmin, xmax, 6)) { const x = X(v); ctx.fillText(fmt(v), x - 8, H - 6); }
    if (opts.ylabel) { ctx.fillText(opts.ylabel, L + pw - ctx.measureText(opts.ylabel).width, 13); }
    for (const s of opts.series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.setLineDash(s.dash || []); ctx.beginPath();
      let pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i], y = s.y[i];
        if (x == null || y == null || !isFinite(y)) { pen = false; continue; }
        if (!pen) { ctx.moveTo(X(x), Y(y)); pen = true; } else ctx.lineTo(X(x), Y(y));
      }
      ctx.stroke(); ctx.setLineDash([]);
      if (s.points) { ctx.fillStyle = s.color; for (let i = 0; i < s.x.length; i++) if (s.y[i] != null) { ctx.beginPath(); ctx.arc(X(s.x[i]), Y(s.y[i]), 2, 0, 7); ctx.fill(); } }
    }
    for (const c of [].concat(opts.cursor == null ? [] : opts.cursor)) {
      const x = typeof c === "object" ? c.x : c; if (x < xmin || x > xmax) continue;
      ctx.strokeStyle = (typeof c === "object" && c.color) || "#e69a2e"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(x), T); ctx.lineTo(X(x), T + ph); ctx.stroke();
    }
    let lx = L + 4; const ly = T + 10;
    for (const s of opts.series) if (s.label) {
      ctx.fillStyle = s.color; ctx.fillRect(lx, ly - 4, 12, 3); ctx.fillStyle = "#b8c4d0"; ctx.fillText(s.label, lx + 15, ly); lx += 22 + ctx.measureText(s.label).width;
    }
    if (opts.onClick && !canvas._clickBound) {
      canvas._clickBound = true;
      canvas.addEventListener("click", ev => {
        const r = canvas.getBoundingClientRect(), f = canvas._xform; if (!f) return;
        const x = f.xmin + (ev.clientX - r.left - f.L) / f.pw * (f.xmax - f.xmin);
        canvas._onClick && canvas._onClick(x);
      });
    }
    canvas._onClick = opts.onClick;
  }
  return { draw };
})();
