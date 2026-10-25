import type { SceneObject, TaskDef, TaskParam } from "../api/types";
import { isRunActive, useStudio } from "../store";
import { TaskIcon } from "./Icons";
import { RunPanel } from "./RunPanel";

function ObjectField({ p, value, objects, active }: { p: TaskParam; value: unknown; objects: SceneObject[]; active: boolean }) {
  const st = useStudio.getState();
  return (
    <div className={`param${active ? " param-active" : ""}`}>
      <label className="field">
        <span>{p.label}</span>
        <div className="param-row">
          <select value={(value as string) ?? ""} onChange={(e) => st.setParam(p.key, e.target.value || undefined)}>
            <option value="">Choose…</option>
            {objects.map((o) => (
              <option key={o.id} value={o.id} disabled={!o.can_lift && p.key !== "base"}>
                {o.name}
                {!o.can_lift && p.key !== "base" ? " (cannot be lifted)" : ""}
              </option>
            ))}
          </select>
          <button
            className={`btn btn-sm${active ? " btn-primary" : ""}`}
            onClick={() => st.setActiveParam(active ? null : p.key)}
            aria-pressed={active}
            title="Pick the object in the 3D view"
          >
            {active ? "Picking…" : "Pick in 3D"}
          </button>
        </div>
      </label>
    </div>
  );
}

function PointField({ p, value, active }: { p: TaskParam; value: unknown; active: boolean }) {
  const st = useStudio.getState();
  const pt = Array.isArray(value) ? (value as [number, number]) : null;
  const typed = (axis: 0 | 1, text: string) => {
    const v = parseFloat(text) / 1000;
    if (!Number.isFinite(v)) return;
    const next: [number, number] = pt ? [...pt] : [0.5, 0];
    next[axis] = v;
    st.setParam(p.key, next);
  };
  return (
    <div className={`param${active ? " param-active" : ""}`}>
      <div className="field">
        <span>{p.label}</span>
        <div className="param-row">
          <span className="param-value">
            {pt ? `${Math.round(Math.hypot(pt[0], pt[1]) * 1000)} mm from the base, ${Math.round((Math.atan2(pt[1], pt[0]) * 180) / Math.PI)}°` : "Not chosen yet"}
          </span>
          <button className={`btn btn-sm${active ? " btn-primary" : ""}`} onClick={() => st.setActiveParam(active ? null : p.key)} aria-pressed={active}>
            {active ? "Click the table…" : pt ? "Change" : "Choose on table"}
          </button>
        </div>
        <details className="coords">
          <summary>Enter the position instead</summary>
          <div className="field-row" key={pt ? `${pt[0]}:${pt[1]}` : "none"}>
            <label className="field">
              <span>Forward x (mm)</span>
              <input type="number" step={10} defaultValue={pt ? Math.round(pt[0] * 1000) : ""} onBlur={(e) => typed(0, e.target.value)} />
            </label>
            <label className="field">
              <span>Sideways y (mm)</span>
              <input type="number" step={10} defaultValue={pt ? Math.round(pt[1] * 1000) : ""} onBlur={(e) => typed(1, e.target.value)} />
            </label>
          </div>
        </details>
      </div>
    </div>
  );
}

function ObjectsField({ p, value, objects }: { p: TaskParam; value: unknown; objects: SceneObject[] }) {
  const st = useStudio.getState();
  const chosen = Array.isArray(value) ? (value as string[]) : [];
  const liftable = objects.filter((o) => o.can_lift && o.state === "on_table");
  const toggle = (id: string) => {
    const next = chosen.includes(id) ? chosen.filter((x) => x !== id) : [...chosen, id];
    st.setParam(p.key, next.length ? next : undefined);
  };
  return (
    <fieldset className="param field">
      <legend>{p.label}</legend>
      {liftable.length === 0 && <p className="muted small">No objects on the table can be lifted.</p>}
      <div className="checklist">
        {liftable.map((o) => (
          <label key={o.id} className="check-item">
            <input type="checkbox" checked={chosen.length === 0 || chosen.includes(o.id)} onChange={() => {
              if (chosen.length === 0) st.setParam(p.key, liftable.filter((x) => x.id !== o.id).map((x) => x.id));
              else toggle(o.id);
            }} />
            <span className="swatch" style={{ background: o.color }} aria-hidden="true" />
            {o.name}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function ChoiceField({ p, value }: { p: TaskParam; value: unknown }) {
  const st = useStudio.getState();
  return (
    <fieldset className="param field">
      <legend>{p.label}</legend>
      <div className="segmented" role="radiogroup">
        {(p.choices ?? []).map(([v, label]) => (
          <button key={v} role="radio" aria-checked={value === v} className={value === v ? "seg seg-on" : "seg"} onClick={() => st.setParam(p.key, v)}>
            {label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function TaskForm({ def }: { def: TaskDef }) {
  const task = useStudio((s) => s.task);
  const objects = useStudio((s) => s.scene?.objects ?? []);
  const plan = useStudio((s) => s.plan);
  const planning = useStudio((s) => s.planning);
  const starting = useStudio((s) => s.starting);
  const st = useStudio.getState();
  const missing = def.params.filter((p) => (p.type === "object" || p.type === "point") && task.params[p.key] === undefined);

  return (
    <div className="task-form">
      <p className="muted">{def.summary}</p>
      {def.params.map((p) =>
        p.type === "object" ? (
          <ObjectField key={p.key} p={p} value={task.params[p.key]} objects={objects} active={task.active === p.key} />
        ) : p.type === "point" ? (
          <PointField key={p.key} p={p} value={task.params[p.key]} active={task.active === p.key} />
        ) : p.type === "objects" ? (
          <ObjectsField key={p.key} p={p} value={task.params[p.key]} objects={objects} />
        ) : (
          <ChoiceField key={p.key} p={p} value={task.params[p.key]} />
        ),
      )}
      {plan && plan.task_id === def.id && (
        <div className={`plan ${plan.ok ? "plan-ok" : "plan-bad"}`} aria-live="polite">
          {plan.ok ? (
            <>
              <p className="plan-title">
                Ready: {plan.legs.length} {plan.legs.length === 1 ? "move" : "moves"}, about {Math.round(plan.estimated_s)} s
              </p>
              <ol className="plan-steps">
                {plan.legs.map((l, i) => (
                  <li key={i} className={l.purpose !== "move" ? "plan-step-aux" : ""}>
                    {l.label}
                  </li>
                ))}
              </ol>
            </>
          ) : (
            <>
              <p className="plan-title">This cannot run yet</p>
              <ul className="issues">
                {plan.problems.map((p, i) => (
                  <li key={i}>
                    <strong>{p.message}</strong>
                    {p.hint && <span> {p.hint}</span>}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
      <div className="task-actions">
        <button className="btn" disabled={missing.length > 0 || planning || starting} onClick={() => st.checkTask()}>
          {planning ? "Checking…" : "Check"}
        </button>
        <button
          className="btn btn-primary btn-run"
          disabled={missing.length > 0 || planning || starting || (plan !== null && plan.task_id === def.id && !plan.ok)}
          onClick={() => st.runTask()}
        >
          {starting ? "Starting…" : "Run"}
        </button>
      </div>
      {missing.length > 0 && <p className="muted small">Still needed: {missing.map((m) => m.label.toLowerCase()).join(", ")}.</p>}
    </div>
  );
}

function RobotCard() {
  const config = useStudio((s) => s.config);
  const atHome = useStudio((s) => s.live?.robot_at_home ?? true);
  const running = useStudio((s) => isRunActive(s));
  if (!config) return null;
  const r = config.robot;
  const w = config.world;
  return (
    <section className="panel-section robot-card" aria-label="Robot">
      <div className="section-head">
        <h2>{r.name}</h2>
        <span className="chip chip-ok">Simulated</span>
      </div>
      <dl className="facts facts-grid">
        <dt>Payload</dt>
        <dd>{r.payload_kg.toFixed(1)} kg</dd>
        <dt>Work area</dt>
        <dd>
          {Math.round(w.r_in * 1000)}–{Math.round(w.r_out * 1000)} mm, ±{Math.round(w.theta_max_deg)}°
        </dd>
        <dt>Gripper</dt>
        <dd>
          {Math.round(r.limits.min_grasp_width_m * 1000)}–{Math.round(r.limits.max_grasp_width_m * 1000)} mm wide objects
        </dd>
        <dt>Table</dt>
        <dd>⌀ {(2 * w.table_radius).toFixed(2)} m, sized to the reach</dd>
      </dl>
      {!atHome && !running && (
        <button className="btn btn-sm" onClick={() => useStudio.getState().resetRobot()}>
          Move robot to rest position
        </button>
      )}
    </section>
  );
}

export function TaskPanel() {
  const config = useStudio((s) => s.config);
  const task = useStudio((s) => s.task);
  const run = useStudio((s) => s.live?.run ?? null);
  const st = useStudio.getState();
  const def = config?.tasks.find((t) => t.id === task.id) ?? null;

  return (
    <aside className="panel panel-right" aria-label="Tasks">
      <RobotCard />
      {run ? (
        <RunPanel run={run} />
      ) : (
        <section className="panel-section panel-grow">
          <div className="section-head">
            <h2>Tasks</h2>
          </div>
          <div className="task-grid" role="list">
            {config?.tasks.map((t) => (
              <button
                key={t.id}
                role="listitem"
                className={`task-card${t.id === task.id ? " task-card-on" : ""}`}
                onClick={() => st.setTask(t.id === task.id ? null : t.id)}
                aria-pressed={t.id === task.id}
              >
                <span className="task-icon">
                  <TaskIcon id={t.id} />
                </span>
                <span className="task-title">{t.title}</span>
              </button>
            ))}
          </div>
          {def ? <TaskForm def={def} /> : <p className="muted">Choose a task. The robot only does what it can safely reach and lift.</p>}
        </section>
      )}
    </aside>
  );
}
