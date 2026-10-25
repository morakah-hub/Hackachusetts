// Robotics Lab Workbench UI (v0.7). Talks only to the local JSON API; all simulation happens in Python.
"use strict";
const $ = id => document.getElementById(id);
const S = { caps: null, status: null, req: null, tele: null, teleKey: "", tab: "result", joint: 1, selT: null,
            msgSince: 0, frameId: -1, scrubbing: false, lastState: "", inspectKey: "" };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const act = (action, extra = {}) => api("/api/action", { action, ...extra }).catch(e => consoleLine(e.message, "err"));
const fmt = (v, d = 3) => v == null || !isFinite(v) ? "–" : (+v).toFixed(d);

// ------------------------------------------------------------------ config form
function fillSelect(sel, entries) { sel.innerHTML = ""; for (const [v, label] of entries) { const o = document.createElement("option"); o.value = v; o.textContent = label; sel.appendChild(o); } }

function scenarioOptions() { return S.caps.scenarios[$("f-scenario").value]; }

function updateVisibility() {
  const sc = scenarioOptions();
  document.querySelectorAll(".opt").forEach(el => el.classList.toggle("hidden", !sc.options.includes(el.dataset.opt)));
  $("scenario-desc").textContent = sc.description;
  const cur = $("f-controller").value;
  fillSelect($("f-controller"), sc.controllers.map(c => [c, c + " — " + S.caps.controllers[c].split(" (")[0]]));
  $("f-controller").value = sc.controllers.includes(cur) ? cur : sc.controllers[0];
  $("f-observer").disabled = !["torque_mpc", "torque_ctc"].includes($("f-controller").value);
  drawWorkspace();
}

function fillForm(req) {
  S.req = req;
  $("f-scenario").value = req.scenario; updateVisibility();
  $("f-controller").value = req.controller; $("f-observer").checked = req.observer;
  $("f-observer").disabled = !["torque_mpc", "torque_ctc"].includes(req.controller);
  $("f-payload").value = req.payload_kg;
  [$("f-ox").value, $("f-oy").value] = req.object_xy; [$("f-px").value, $("f-py").value] = req.place_xy;
  const sc = S.caps.scenarios[req.scenario];
  $("f-targets").value = (req.targets || sc.default_targets || []).map(t => t.join(" ")).join("\n");
  const ob = req.obstacle || sc.default_obstacle;
  if (ob) { [$("f-obx").value, $("f-oby").value, $("f-obz").value] = ob.pos; [$("f-obsx").value, $("f-obsy").value, $("f-obsz").value] = ob.size; }
  $("f-contact").value = req.contact_preset; $("f-grip").value = req.grip_preset; $("f-seed").value = req.plan_seed;
  $("f-force-on").checked = !!req.force_pulse;
  if (req.force_pulse) { const f = req.force_pulse; $("f-force-body").value = f.body; $("f-force-mag").value = f.magnitude_n;
    [$("f-fdx").value, $("f-fdy").value, $("f-fdz").value] = f.direction; $("f-force-t").value = f.start_s; $("f-force-d").value = f.duration_s; }
  $("f-noise-on").checked = !!req.noise;
  if (req.noise) { $("f-noise-q").value = req.noise.position_std; $("f-noise-v").value = req.noise.velocity_std; $("f-noise-seed").value = req.noise.seed || 0; }
  drawWorkspace();
}

function readForm() {
  const sc = scenarioOptions(), num = id => parseFloat($(id).value);
  const req = { ...S.req, scenario: $("f-scenario").value, controller: $("f-controller").value,
    observer: $("f-observer").checked && !$("f-observer").disabled, payload_kg: num("f-payload"),
    object_xy: [num("f-ox"), num("f-oy")], place_xy: [num("f-px"), num("f-py")],
    contact_preset: $("f-contact").value, grip_preset: $("f-grip").value, plan_seed: parseInt($("f-seed").value || "0") };
  if (sc.options.includes("targets")) {
    const t = $("f-targets").value.trim().split(/\n+/).filter(Boolean).map(l => l.trim().split(/[\s,]+/).map(Number));
    const def = JSON.stringify(sc.default_targets);
    req.targets = JSON.stringify(t) === def ? null : t;
  } else req.targets = null;
  if (sc.options.includes("obstacle")) {
    const ob = { pos: [num("f-obx"), num("f-oby"), num("f-obz")], size: [num("f-obsx"), num("f-obsy"), num("f-obsz")] };
    req.obstacle = JSON.stringify(ob) === JSON.stringify(sc.default_obstacle) ? null : ob;
  } else req.obstacle = null;
  req.force_pulse = $("f-force-on").checked && sc.options.includes("force_pulse") ? { body: $("f-force-body").value, magnitude_n: num("f-force-mag"),
    direction: [num("f-fdx"), num("f-fdy"), num("f-fdz")], start_s: num("f-force-t"), duration_s: num("f-force-d") } : null;
  req.noise = $("f-noise-on").checked ? { position_std: num("f-noise-q"), velocity_std: num("f-noise-v"), seed: parseInt($("f-noise-seed").value || "0") } : null;
  return req;
}

async function applySettings() {
  try {
    const r = await api("/api/request", readForm());
    showResolved(r.summary); $("apply-msg").className = "msg ok"; $("apply-msg").textContent = "settings applied (used by the next Run / Reset)";
    S.req = r.request; return true;
  } catch (e) { $("apply-msg").className = "msg err"; $("apply-msg").textContent = e.message; return false; }
}
function showResolved(lines) { $("resolved").innerHTML = lines.map(l => `<li>${esc(l)}</li>`).join(""); }
const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// ------------------------------------------------------------------ workspace editor (top view)
const WS = { xr: [-0.20, 0.90], yr: [-0.50, 0.50], drag: null };
function wsXY(x, y) { // world (x forward = up, y left = left) -> canvas
  const c = $("workspace"), s = c.width / (WS.xr[1] - WS.xr[0]);
  return [c.width / 2 - y * s, c.height - (x - WS.xr[0]) * s];
}
function wsWorld(px, py) { const c = $("workspace"), s = c.width / (WS.xr[1] - WS.xr[0]); return [WS.xr[0] + (c.height - py) / s, (c.width / 2 - px) / s]; }
function drawWorkspace() {
  const c = $("workspace"); if (!c || !S.caps) return; const ctx = c.getContext("2d"), s = c.width / (WS.xr[1] - WS.xr[0]);
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#3a2a1c"; const [tx0, ty0] = wsXY(0.90, 0.50), [tx1, ty1] = wsXY(-0.20, -0.50); ctx.fillRect(tx0, ty0, tx1 - tx0, ty1 - ty0);
  const b = S.caps.bounds.table_xy; const [bx0, by0] = wsXY(b.x[1], b.y[1]), [bx1, by1] = wsXY(b.x[0], b.y[0]);
  ctx.setLineDash([4, 3]); ctx.strokeStyle = "#8b9aab"; ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0); ctx.setLineDash([]);
  const [rx, ry] = wsXY(0, 0); ctx.fillStyle = "#c9d1d9"; ctx.beginPath(); ctx.arc(rx, ry, 0.07 * s, 0, 7); ctx.fill();
  ctx.fillStyle = "#11171d"; ctx.font = "10px sans-serif"; ctx.fillText("base", rx - 11, ry + 3);
  const sc = S.caps.scenarios[$("f-scenario").value];
  if (sc.options.includes("obstacle") || sc.default_obstacle) {
    const ob = sc.options.includes("obstacle") ? { pos: [+$("f-obx").value, +$("f-oby").value], size: [+$("f-obsx").value, +$("f-obsy").value] } : null;
    if (ob && isFinite(ob.pos[0])) { const [a, b2] = wsXY(ob.pos[0] + ob.size[0], ob.pos[1] + ob.size[1]), [c2, d] = wsXY(ob.pos[0] - ob.size[0], ob.pos[1] - ob.size[1]);
      ctx.fillStyle = "#3383d6"; ctx.fillRect(a, b2, c2 - a, d - b2); }
  }
  if (sc.kind === "pick_place") {
    const h = S.caps.object_half_size[0] * s;
    const [ox, oy] = wsXY(+$("f-ox").value, +$("f-oy").value); ctx.fillStyle = "#f28c28"; ctx.fillRect(ox - h, oy - h, 2 * h, 2 * h);
    const [px, py] = wsXY(+$("f-px").value, +$("f-py").value); ctx.strokeStyle = "#3ccf7a"; ctx.lineWidth = 2; ctx.strokeRect(px - h, py - h, 2 * h, 2 * h); ctx.lineWidth = 1;
  } else {
    ctx.fillStyle = "#f2555a";
    for (const l of $("f-targets").value.trim().split(/\n+/)) { const t = l.trim().split(/[\s,]+/).map(Number); if (t.length === 3) { const [x, y] = wsXY(t[0], t[1]); ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill(); } }
  }
  if (S.scan && S.scan.key === $("f-scenario").value + JSON.stringify(S.req && S.req.obstacle)) {
    const colors = { found: "rgba(60,207,122,.55)", blocked: "rgba(242,85,90,.75)", not_found: "rgba(139,154,171,.6)" };
    for (const p of S.scan.points) { const [x, y] = wsXY(p.x, p.y); ctx.fillStyle = colors[p.class]; ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 7); ctx.fill(); }
  }
  ctx.fillStyle = "#8b9aab"; ctx.fillText("+x", c.width / 2 + 4, 10); ctx.fillText("+y ←", 4, c.height - 6);
}
function wsInit() {
  const c = $("workspace");
  c.addEventListener("mousedown", ev => {
    if (scenarioOptions().kind !== "pick_place") return;
    const r = c.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
    const d = (ix, iy) => { const [x, y] = wsXY(+$(ix).value, +$(iy).value); return Math.hypot(px - x, py - y); };
    WS.drag = d("f-ox", "f-oy") < d("f-px", "f-py") ? ["f-ox", "f-oy"] : ["f-px", "f-py"];
  });
  window.addEventListener("mousemove", ev => {
    if (!WS.drag) return; const r = c.getBoundingClientRect();
    const [x, y] = wsWorld(ev.clientX - r.left, ev.clientY - r.top);
    $(WS.drag[0]).value = x.toFixed(3); $(WS.drag[1]).value = y.toFixed(3); drawWorkspace();
  });
  window.addEventListener("mouseup", () => { if (WS.drag) { WS.drag = null; applySettings(); } });
}

// ------------------------------------------------------------------ viewport camera
function camInit() {
  const img = $("frame"); let last = null, pan = false, pending = null, timer = null;
  const send = () => { timer = null; if (pending) { act("camera", pending); pending = null; } };
  const queue = c => { pending = { ...(pending || {}), ...c }; if (!timer) timer = setTimeout(send, 40); };
  img.addEventListener("contextmenu", e => e.preventDefault());
  img.addEventListener("mousedown", e => { last = [e.clientX, e.clientY]; pan = e.shiftKey || e.button === 2; img.style.cursor = "grabbing"; });
  window.addEventListener("mouseup", () => { last = null; img.style.cursor = "grab"; });
  window.addEventListener("mousemove", e => {
    if (!last || !S.status) return; const cam = S.camLocal || (S.camLocal = { ...S.status.camera });
    const dx = e.clientX - last[0], dy = e.clientY - last[1]; last = [e.clientX, e.clientY];
    if (pan) {
      const az = cam.azimuth * Math.PI / 180, k = cam.distance * 0.0016;
      cam.lookat = [cam.lookat[0] + k * (dx * Math.sin(az) - dy * Math.cos(az) * 0), cam.lookat[1] - k * dx * Math.cos(az), cam.lookat[2] + k * dy];
      queue({ lookat: cam.lookat });
    } else { cam.azimuth -= dx * 0.4; cam.elevation = Math.max(-89, Math.min(89, cam.elevation - dy * 0.3)); queue({ azimuth: cam.azimuth, elevation: cam.elevation }); }
  });
  img.addEventListener("wheel", e => { e.preventDefault(); const cam = S.camLocal || (S.camLocal = { ...S.status.camera });
    cam.distance = Math.max(0.3, Math.min(6, cam.distance * (e.deltaY > 0 ? 1.08 : 0.93))); queue({ distance: cam.distance }); }, { passive: false });
  $("btn-cam-reset").onclick = () => { S.camLocal = null; act("camera_reset"); };
}

// ------------------------------------------------------------------ polling
async function pollStatus() {
  try {
    const st = await api("/api/status?since=" + S.msgSince); S.status = st;
    for (const m of st.messages) { S.msgSince = Math.max(S.msgSince, m.t); if (m.kind === "error" || m.kind === "result") consoleLine(m.text, m.kind === "error" ? "err" : ""); }
    const pill = $("st-state"); pill.textContent = st.state; pill.className = "pill " + st.state;
    $("st-text").textContent = st.view === "replay"
      ? `REPLAY of recorded run ${st.replay_run} · t = ${fmt(st.replay_time, 2)} s · live state: ${st.message}`
      : `t = ${fmt(st.sim_time, 2)} s · ${st.phase || "–"} · ${st.active_controller || ""} · ${st.message}`;
    $("btn-pause").textContent = st.state === "paused" ? "▶ Resume" : "❚❚ Pause";
    if (document.activeElement !== $("sel-speed")) $("sel-speed").value = String(st.speed);
    if (document.activeElement !== $("sel-quality")) $("sel-quality").value = st.quality;
    $("busy").classList.toggle("hidden", !(st.busy || st.state === "building"));
    $("badge").className = "badge " + (st.view === "replay" ? "replay" : "live");
    $("badge").textContent = st.view === "replay" ? "RECORDED DATA (replay)" : "LIVE SIMULATION";
    if (st.frame_id !== S.frameId) { S.frameId = st.frame_id; $("frame").src = "/api/frame?id=" + st.frame_id; }
    if (st.view === "replay") {
      if (!S.scrubbing) { $("scrub").max = st.replay_duration || 1; $("scrub").value = st.replay_time; }
      S.selT = st.replay_time; $("btn-rplay").textContent = st.replay_playing ? "❚❚" : "▶";
    } else S.selT = st.sim_time;
    $("scrub-t").textContent = "t = " + fmt(S.selT, 3) + " s";
    if (!S.req || (st.state !== S.lastState && ["ready", "finished", "failed"].includes(st.state))) {
      if (!document.querySelector("#config :focus")) fillForm(st.request); showResolved(st.pending_summary); }
    const key = (st.run_id || "") + "|" + st.view + "|" + (st.replay_run || "") + "|" + st.state;
    const due = st.state === "running" ? Date.now() - (S.teleAt || 0) > 900 : key !== S.teleKey;
    if (due) { S.teleKey = key; S.teleAt = Date.now(); loadTelemetry(); }
    if (st.state !== S.lastState && ["finished", "failed"].includes(st.state) && S.tab === "result") renderTab();
    S.lastState = st.state;
    if (st.view === "replay" && !st.replay_playing && S.tab !== "runs" && S.inspectKey !== S.tab + fmt(S.selT, 3)) { S.inspectKey = S.tab + fmt(S.selT, 3); if (["motion", "mpc", "dynamics", "planning", "explain"].includes(S.tab)) renderTab(); }
  } catch (e) { $("st-text").textContent = "server not reachable: " + e.message; }
}

async function loadTelemetry() { try { S.tele = await api("/api/telemetry"); drawPlots(); } catch (e) { /* ignore */ } }

// ------------------------------------------------------------------ plots
const PHASE_COLORS = { approach: "rgba(74,163,255,.07)", descend: "rgba(74,163,255,.14)", close: "rgba(242,179,61,.16)", lift: "rgba(60,207,122,.12)",
  hold: "rgba(60,207,122,.2)", transport: "rgba(198,120,221,.12)", lower: "rgba(60,207,122,.12)", release: "rgba(242,179,61,.16)",
  retreat: "rgba(74,163,255,.1)", verify: "rgba(255,255,255,.06)", move: "rgba(74,163,255,.08)", hold_: "rgba(60,207,122,.12)" };
function bands(T) {
  const out = []; if (!T || !T.t.length) return out; let s = 0;
  for (let i = 1; i <= T.t.length; i++) if (i === T.t.length || T.phase[i] !== T.phase[s]) { out.push({ x0: T.t[s], x1: T.t[Math.min(i, T.t.length - 1)], color: PHASE_COLORS[T.phase[s]] || "rgba(255,255,255,.04)" }); s = i; }
  return out;
}
function col(arr, j) { return arr.map(r => r == null ? null : r[j]); }
function drawPlots() {
  const T = S.tele; if (!T || !T.available) { for (const id of ["plot1", "plot2", "plot3"]) Plot.draw($(id), { title: "no run yet", series: [] }); return; }
  const j = S.joint, b = bands(T), cur = S.selT, seek = x => { act("seek", { t: Math.max(0, x) }); };
  Plot.draw($("plot1"), { title: `joint ${j + 1}: measured q vs reference (rad)`, bands: b, cursor: cur, onClick: seek,
    series: [{ x: T.t, y: col(T.q_ref, j), color: "#c9d1d9", dash: [4, 3], label: "reference" }, { x: T.t, y: col(T.q, j), color: "#4aa3ff", label: "measured" }] });
  Plot.draw($("plot2"), { title: "end-effector tracking error |FK(q) − FK(q_ref)| (mm)", bands: b, cursor: cur, onClick: seek, ymin: 0,
    series: [{ x: T.t, y: T.ee_err_mm, color: "#f2b33d", label: "measured − reference" }] });
  const s3 = [{ x: T.t, y: col(T.tau, j), color: "#4aa3ff", label: "applied τ" }];
  if (T.mpc_cmd.some(r => r && r[j] != null)) s3.push({ x: T.t, y: col(T.mpc_cmd, j), color: "#c678dd", dash: [3, 2], label: "MPC optimizer τ0" });
  if (T.d_hat.some(r => r && r[j] != null)) s3.push({ x: T.t, y: col(T.d_hat, j), color: "#f2b33d", label: "observer d̂" });
  Plot.draw($("plot3"), { title: `joint ${j + 1}: torque (Nm)`, bands: b, cursor: cur, onClick: seek, series: s3 });
}

// ------------------------------------------------------------------ console
function consoleLine(text, cls = "") { const d = document.createElement("div"); d.className = cls; d.textContent = text; $("console-out").appendChild(d); $("console-out").scrollTop = 1e9; }
const HIST = []; let histI = 0;
async function runCommand(text) {
  if (!text.trim()) return; HIST.push(text); histI = HIST.length; consoleLine("› " + text, "cmd");
  try {
    const r = await api("/api/command", { text });
    for (const l of r.output) consoleLine(l, r.ok ? "" : "err");
    showResolved(r.settings);
    const st = await api("/api/status"); fillForm(st.request);
    if (r.ui && r.ui.panel) selectTab(r.ui.panel);
    if (r.ui && r.ui.explain_t != null) { act("seek", { t: r.ui.explain_t }); }
  } catch (e) { consoleLine(e.message, "err"); }
}

// ------------------------------------------------------------------ inspector tabs
function selectTab(name) { S.tab = name; document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name)); renderTab(); }

function renderSections(sections) {
  const body = $("tab-body"); body.innerHTML = ""; const plots = [];
  for (const s of sections) {
    const div = document.createElement("div"); div.className = "card " + (s.type || "");
    let h = s.type ? `<span class="tag">${esc({ intuition: "physical intuition", equation: "equation", values: "actual values", consequence: "observed consequence", warn: "note", detail: "details" }[s.type] || s.type)}</span>` : "";
    if (s.title) h += `<h4>${esc(s.title)}</h4>`;
    if (s.eq) h += `<div class="eq">${s.eq}</div>`;
    if (s.text) h += (Array.isArray(s.text) ? s.text : [s.text]).map(t => `<p>${esc(t)}</p>`).join("");
    if (s.table) h += `<table class="kv">${s.header ? "<tr>" + s.header.map(x => `<th>${esc(x)}</th>`).join("") + "</tr>" : ""}${s.table.map(r => "<tr>" + r.map((c, i) => `<td class="${i && typeof c === "number" ? "num" : ""}">${esc(typeof c === "number" ? fmtNum(c) : c ?? "unavailable")}</td>`).join("") + "</tr>").join("")}</table>`;
    if (s.plot) { h += `<canvas class="inline-plot"></canvas>`; }
    if (s.details) h += `<details><summary>${esc(s.details.summary)}</summary><pre class="mono">${esc(s.details.text)}</pre></details>`;
    if (s.buttons) h += s.buttons.map(bt => `<button data-cmd="${esc(bt.cmd)}">${esc(bt.label)}</button>`).join(" ");
    div.innerHTML = h; body.appendChild(div);
    if (s.plot) plots.push([div.querySelector("canvas"), s.plot]);
  }
  body.querySelectorAll("button[data-cmd]").forEach(b => b.onclick = () => runCommand(b.dataset.cmd));
  for (const [c, p] of plots) Plot.draw(c, p);
}
function fmtNum(v) { const a = Math.abs(v); if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2); return (+v.toPrecision(4)).toString(); }

async function renderTab() {
  const t = S.selT, body = $("tab-body");
  try {
    if (S.tab === "result") return renderResult();
    if (S.tab === "runs") return renderRuns();
    if (S.tab === "experiments") return renderExperiments();
    if (S.tab === "sandbox") return renderSandbox();
    const r = S.tab === "explain" ? await api("/api/explain?t=" + (t ?? "")) : await api(`/api/inspect?kind=${S.tab}&t=${t ?? ""}&joint=${S.joint}`);
    renderSections(r.sections);
  } catch (e) { body.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

function renderResult() {
  const st = S.status || {}, r = st.last_result, T = S.tele, sec = [];
  if (r) {
    const ok = r.outcome === "success";
    sec.push({ type: ok ? "" : "warn", title: `Run ${r.run_id}: ${r.outcome.toUpperCase()}`, text: [r.failure_reason ? `stage ${r.failure_stage}: ${r.failure_reason}` : "all task checks passed",
      `simulated ${fmt(r.sim_time_s, 2)} s in ${fmt(r.wall_time_s, 1)} s of wall time; saved to ${r.directory}`] });
    const m = r.metrics || {}, rows = [];
    const keys = [["placement_error_mm", "placement error", "mm"], ["lift_height_mm", "lift height", "mm"], ["hold_slip_mm", "slip while holding", "mm"],
      ["carry_slip_mm", "slip while carrying", "mm"], ["ee_err_max_mm", "max EE tracking error", "mm"], ["joint_err_max_rad", "max joint tracking error", "rad"],
      ["forbidden_contact_steps", "physics steps with forbidden contacts", ""], ["min_clearance_robot_env_mm", "min robot–environment distance (sampled every 10 ms)", "mm"],
      ["min_clearance_object_env_mm", "min carried-object–environment distance (10 ms)", "mm"], ["torque_saturation_steps", "steps with torque saturation", ""],
      ["robot_contact_steps", "physics steps with robot contacts", ""], ["max_joint_tracking_error_rad", "max joint tracking error", "rad"]];
    for (const [k, l, u] of keys) if (m[k] != null) rows.push([l, m[k], u]);
    if (m.targets) for (const tg of m.targets) rows.push([`${tg.name} [${tg.position.map(v => v.toFixed(2)).join(", ")}]`, tg.planned ? tg.error_at_end_of_hold_mm : "rejected: " + tg.reason, tg.planned ? "mm at end of hold" : ""]);
    if (m.mpc) rows.push(["MPC fallback updates", m.mpc.fallback_updates ?? m.mpc.fallbacks, ""], ["MPC solve time (mean)", m.mpc.solve_ms_mean ?? m.mpc.solve_time_mean_ms, "ms"]);
    sec.push({ type: "values", title: "Measured outcome", table: rows });
    if (m.criterion) sec.push({ type: "", text: m.criterion });
  } else sec.push({ text: st.state === "running" ? "Run in progress …" : "No finished run yet. Choose a scenario and press Run (or type 'help' in the console)." });
  if (r && T && T.available) sec.push({ type: "", title: "Settings of the recorded run", table: T.summary.map(l => [l]) });
  else if (st.pending_summary) sec.push({ type: "", title: "Pending settings (used by the next Run)", table: st.pending_summary.map(l => [l]) });
  renderSections(sec);
}

async function renderRuns() {
  const runs = await api("/api/runs"); const body = $("tab-body");
  body.innerHTML = `<p class="hint">Saved runs (results/workbench/runs). Opening one shows recorded data only.</p>` +
    `<table class="kv"><tr><th>run</th><th>scenario</th><th>controller</th><th>outcome</th><th></th></tr>` +
    runs.slice(0, 60).map(r => `<tr><td class="mono">${esc(r.run_id)}${r.label ? "<br><span class='muted'>" + esc(r.label) + "</span>" : ""}</td><td>${esc(r.scenario)}</td><td>${esc(r.controller)}${r.observer ? " + obs" : ""}</td><td class="${r.outcome === "success" ? "ok" : "bad"}">${esc(r.outcome || "?")}</td><td><button data-run="${esc(r.run_id)}">Replay</button></td></tr>`).join("") + "</table>";
  body.querySelectorAll("button[data-run]").forEach(b => b.onclick = () => act("replay", { run_id: b.dataset.run }));
}

async function renderExperiments() {
  try { const r = await api("/api/inspect?kind=experiments"); renderSections(r.sections); }
  catch (e) { $("tab-body").innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

async function renderSandbox() {
  const body = $("tab-body");
  if (!$("sb-form")) {
    body.innerHTML = `<div class="card warn"><b>Educational model, not a Panda experiment.</b> One rotary joint, inertia I, no gravity, solved by the project's own TorqueMPCController with a fixed linear model.</div>
    <div id="sb-form" class="card"><div class="xy">I <input id="sb-I" type="number" value="0.5" step="0.1"> kg·m² τmax <input id="sb-tau" type="number" value="1" step="0.5"> Nm</div>
    <div class="xy">horizon N <input id="sb-N" type="number" value="20" step="1"> position weight <input id="sb-q" type="number" value="10" step="1"></div>
    <div class="xy">velocity weight <input id="sb-v" type="number" value="1"> effort weight <input id="sb-r" type="number" value="0.1" step="0.05"></div>
    <div class="xy">target θ <input id="sb-ref" type="number" value="1" step="0.1"> rad, Ts 0.1 s, 3 s</div><button id="sb-go" class="primary">Simulate</button></div><div id="sb-out"></div>`;
    $("sb-go").onclick = renderSandbox;
  }
  const v = id => parseFloat($(id).value);
  try {
    const r = await api("/api/sandbox", { inertia: v("sb-I"), torque_limit: v("sb-tau"), horizon: v("sb-N"), w_position: v("sb-q"), w_velocity: v("sb-v"), w_effort: v("sb-r"), target: v("sb-ref") });
    const keep = $("sb-form").outerHTML; renderSections(r.sections); $("tab-body").insertAdjacentHTML("afterbegin", `<div class="card warn"><b>Educational model, not a Panda experiment.</b> One rotary joint, inertia I, no gravity, solved by the project's own TorqueMPCController with a fixed linear model.</div>` + keep);
    $("sb-go").onclick = renderSandbox;
  } catch (e) { $("tab-body").insertAdjacentHTML("beforeend", `<p class="bad">${esc(e.message)}</p>`); }
}

// ------------------------------------------------------------------ wiring
async function init() {
  S.caps = await api("/api/capabilities");
  fillSelect($("f-scenario"), Object.entries(S.caps.scenarios).map(([k, v]) => [k, v.title]));
  fillSelect($("f-contact"), Object.entries(S.caps.contact_presets).map(([k, v]) => [k, k]));
  fillSelect($("f-grip"), Object.entries(S.caps.grip_presets).map(([k, v]) => [k, k]));
  fillSelect($("f-force-body"), S.caps.force_bodies.map(b => [b, b]));
  fillSelect($("sel-joint"), [0, 1, 2, 3, 4, 5, 6].map(j => [j, "joint " + (j + 1)])); $("sel-joint").value = "1";
  const st = await api("/api/status"); S.status = st; fillForm(st.request); showResolved(st.pending_summary);
  $("f-scenario").onchange = () => { const sc = scenarioOptions(); const ob = sc.default_obstacle;
    $("f-targets").value = (sc.default_targets || []).map(t => t.join(" ")).join("\n");
    if (ob) { [$("f-obx").value, $("f-oby").value, $("f-obz").value] = ob.pos; [$("f-obsx").value, $("f-obsy").value, $("f-obsz").value] = ob.size; }
    updateVisibility(); applySettings(); };
  $("f-controller").onchange = () => { $("f-observer").disabled = !["torque_mpc", "torque_ctc"].includes($("f-controller").value); applySettings(); };
  for (const id of ["f-observer", "f-contact", "f-grip", "f-seed", "f-payload", "f-ox", "f-oy", "f-px", "f-py", "f-obx", "f-oby", "f-obz", "f-obsx", "f-obsy", "f-obsz", "f-targets"])
    $(id).addEventListener("change", () => { drawWorkspace(); applySettings(); });
  document.querySelectorAll(".presets button").forEach(b => b.onclick = () => { $("f-payload").value = b.dataset.mass; applySettings(); });
  $("btn-apply").onclick = applySettings;
  $("btn-scan").onclick = async () => {
    if (!(await applySettings())) return;
    $("scan-legend").textContent = "scanning ...";
    try { const r = await api("/api/scan", {}); r.key = $("f-scenario").value + JSON.stringify(S.req && S.req.obstacle); S.scan = r;
      $("scan-legend").innerHTML = `<span style="color:#3ccf7a">● found ${r.counts.found}</span> <span style="color:#f2555a">● blocked ${r.counts.blocked}</span> <span style="color:#8b9aab">● not found ${r.counts.not_found}</span>`;
      $("scan-legend").title = r.note; drawWorkspace();
    } catch (e) { $("scan-legend").textContent = e.message; } };
  $("btn-check").onclick = async () => {
    if (!(await applySettings())) return;
    $("apply-msg").className = "msg"; $("apply-msg").textContent = "checking (IK search on a private model) ...";
    try { const r = await api("/api/preflight");
      $("apply-msg").innerHTML = r.checks.map(c => `<div class="${c.ok ? "ok" : "bad"}">${c.ok ? "✓" : "✗"} ${esc(c.check)}: ${esc(c.detail)}</div>`).join("");
    } catch (e) { $("apply-msg").className = "msg err"; $("apply-msg").textContent = e.message; } };
  $("btn-defaults").onclick = async () => { const r = await api("/api/request", { scenario: $("f-scenario").value, controller: scenarioOptions().controllers[0] }); fillForm(r.request); showResolved(r.summary); };
  $("btn-run").onclick = async () => { if (await applySettings()) act("run"); };
  $("btn-pause").onclick = () => act(S.status && S.status.state === "paused" ? "resume" : "pause");
  $("btn-step").onclick = () => act("step", { physics_steps: 1 });
  $("btn-step-ctrl").onclick = () => act("step", { unit: "control" });
  $("btn-stop").onclick = () => act("stop");
  $("btn-reset").onclick = async () => { if (await applySettings()) act("reset"); };
  $("sel-speed").onchange = e => act("speed", { value: parseFloat(e.target.value) });
  $("sel-quality").onchange = e => act("quality", { level: e.target.value });
  $("btn-live").onclick = () => act("live");
  $("btn-quit").onclick = async () => { if (!confirm("Close the workbench (the server stops)?")) return;
    await api("/api/quit", {}).catch(() => {}); document.body.innerHTML = "<p style='padding:40px'>Workbench closed. You can close this tab.</p>"; };
  $("btn-replay").onclick = () => act("replay", {});
  $("btn-rplay").onclick = () => act(S.status && S.status.replay_playing ? "replay_pause" : "replay_play", { speed: parseFloat($("sel-rspeed").value) });
  $("sel-rspeed").onchange = () => { if (S.status && S.status.replay_playing) act("replay_play", { speed: parseFloat($("sel-rspeed").value) }); };
  let seekT = null;
  $("scrub").addEventListener("input", e => { S.scrubbing = true; const t = parseFloat(e.target.value); if (!seekT) seekT = setTimeout(() => { seekT = null; act("seek", { t: parseFloat($("scrub").value) }); }, 60); $("scrub-t").textContent = "t = " + t.toFixed(3) + " s"; });
  $("scrub").addEventListener("change", e => { S.scrubbing = false; act("seek", { t: parseFloat(e.target.value) }); });
  $("sel-joint").onchange = e => { S.joint = parseInt(e.target.value); drawPlots(); if (["motion", "mpc", "dynamics"].includes(S.tab)) renderTab(); };
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => selectTab(b.dataset.tab));
  const inp = $("console-in");
  inp.addEventListener("keydown", e => {
    if (e.key === "Enter") { runCommand(inp.value); inp.value = ""; }
    else if (e.key === "ArrowUp") { histI = Math.max(0, histI - 1); inp.value = HIST[histI] || ""; e.preventDefault(); }
    else if (e.key === "ArrowDown") { histI = Math.min(HIST.length, histI + 1); inp.value = HIST[histI] || ""; e.preventDefault(); }
  });
  consoleLine("Robotics Lab Workbench v0.7 — type 'help' for commands, e.g. 'load pick; payload 1.15; run'.", "set");
  wsInit(); camInit(); selectTab("result");
  setInterval(pollStatus, 200); setInterval(() => { if (S.tab === "experiments") renderExperiments(); }, 2000);
  window.addEventListener("resize", drawPlots);
  pollStatus();
}
init();
