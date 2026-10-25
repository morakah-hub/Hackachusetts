import { useEffect, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { isRunActive, useStudio } from "../store";
import { dims, kg } from "../lib/format";

const STATE_TEXT: Record<string, string> = {
  on_table: "On the table",
  stacked: "Stacked",
  tipped: "Tipped over",
  fallen: "Fell off the table",
  unsupported: "Not supported",
};

/** Details and edits of the selected object, floating over the 3D view. */
export function ObjectInspector() {
  const config = useStudio((s) => s.config);
  const obj = useStudio((s) => s.scene?.objects.find((o) => o.id === s.selectedId) ?? null);
  const running = useStudio((s) => isRunActive(s));
  const [mass, setMass] = useState("");
  const [name, setName] = useState("");
  const [px, setPx] = useState("");
  const [py, setPy] = useState("");

  useEffect(() => {
    if (obj) {
      setMass(obj.mass.toString());
      setName(obj.name);
    }
  }, [obj?.id, obj?.mass, obj?.name]);

  useEffect(() => {
    if (obj) {
      setPx(Math.round(obj.pos[0] * 1000).toString());
      setPy(Math.round(obj.pos[1] * 1000).toString());
    }
  }, [obj?.id, obj?.pos[0], obj?.pos[1]]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = useStudio.getState();
      if (!st.selectedId || isRunActive(st)) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        st.deleteObject(st.selectedId);
      } else if (e.key === "q" || e.key === "Q") st.rotateObject(st.selectedId, Math.PI / 12);
      else if (e.key === "e" || e.key === "E") st.rotateObject(st.selectedId, -Math.PI / 12);
      else if (e.key === "Escape") st.select(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!obj || !config) return null;
  const st = useStudio.getState();
  const payload = config.robot.payload_kg;
  const frac = Math.min(1, obj.mass / payload);
  const r = Math.hypot(obj.pos[0], obj.pos[1]);
  const commitMass = () => {
    const v = parseFloat(mass);
    if (!Number.isFinite(v) || v === obj.mass) return setMass(obj.mass.toString());
    st.updateObject(obj.id, { mass: v });
  };
  const commitName = () => {
    const v = name.trim();
    if (!v || v === obj.name) return setName(obj.name);
    st.updateObject(obj.id, { name: v });
  };
  const commitPosition = () => {
    const x = parseFloat(px) / 1000;
    const y = parseFloat(py) / 1000;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      setPx(Math.round(obj.pos[0] * 1000).toString());
      setPy(Math.round(obj.pos[1] * 1000).toString());
      return;
    }
    if (Math.abs(x - obj.pos[0]) < 0.0005 && Math.abs(y - obj.pos[1]) < 0.0005) return;
    st.moveObject(obj.id, x, y);
  };
  const onPosKey = (e: ReactKeyboardEvent<HTMLInputElement>) => e.key === "Enter" && commitPosition();

  return (
    <aside className="inspector" aria-label={`Selected object: ${obj.name}`}>
      <div className="inspector-head">
        <span className="swatch" style={{ background: obj.color }} aria-hidden="true" />
        <input
          className="inspector-name"
          value={name}
          aria-label="Object name"
          disabled={running}
          onChange={(e) => setName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        />
        <button className="icon-btn" aria-label="Close" onClick={() => st.select(null)}>
          ×
        </button>
      </div>
      <dl className="facts">
        <dt>Size</dt>
        <dd>{dims(obj.size)}{obj.asset_id ? " (box around the model)" : ""}</dd>
        <dt>State</dt>
        <dd className={obj.state === "tipped" || obj.state === "fallen" ? "text-bad" : ""}>{STATE_TEXT[obj.state] ?? obj.state}</dd>
        <dt>Position</dt>
        <dd>
          {Math.round(r * 1000)} mm from the base, {Math.round((Math.atan2(obj.pos[1], obj.pos[0]) * 180) / Math.PI)}°
        </dd>
      </dl>
      <div className="field-row pos-row">
        <label className="field">
          <span>Forward x (mm)</span>
          <input type="number" step={5} value={px} disabled={running} onChange={(e) => setPx(e.target.value)} onBlur={commitPosition} onKeyDown={onPosKey} />
        </label>
        <label className="field">
          <span>Sideways y (mm)</span>
          <input type="number" step={5} value={py} disabled={running} onChange={(e) => setPy(e.target.value)} onBlur={commitPosition} onKeyDown={onPosKey} />
        </label>
      </div>
      <label className="field">
        <span>Weight (kg)</span>
        <input
          type="number"
          min={0.01}
          step={0.05}
          value={mass}
          disabled={running}
          onChange={(e) => setMass(e.target.value)}
          onBlur={commitMass}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        />
      </label>
      <div className="payload" aria-label={`Weight ${kg(obj.mass)} of ${payload} kg payload`}>
        <div className={`payload-bar ${obj.mass > payload ? "payload-over" : frac > 0.8 ? "payload-high" : ""}`}>
          <div style={{ width: `${frac * 100}%` }} />
        </div>
        <span>
          {kg(obj.mass)} of {payload.toFixed(1)} kg payload
        </span>
      </div>
      {obj.problems.length > 0 || obj.notes.length > 0 ? (
        <ul className="issues">
          {obj.problems.map((p) => (
            <li key={p.code}>
              <strong>{p.message}</strong>
              {p.hint && <span> {p.hint}</span>}
            </li>
          ))}
          {obj.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
          {obj.problems.length > 0 && <li className="muted">It can stay on the table as an obstacle.</li>}
        </ul>
      ) : (
        <p className="ok-line">✓ The arm can pick this up.</p>
      )}
      <div className="inspector-actions">
        <button className="btn btn-sm" disabled={running} onClick={() => st.rotateObject(obj.id, Math.PI / 12)} title="Rotate left (Q)">
          ⟲ 15°
        </button>
        <button className="btn btn-sm" disabled={running} onClick={() => st.rotateObject(obj.id, -Math.PI / 12)} title="Rotate right (E)">
          ⟳ 15°
        </button>
        <button className="btn btn-sm btn-danger" disabled={running} onClick={() => st.deleteObject(obj.id)} title="Remove (Delete)">
          Remove
        </button>
      </div>
      <p className="muted small">Drag the object to move it. Q / E rotate, Delete removes.</p>
    </aside>
  );
}
