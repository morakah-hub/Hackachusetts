import type { Run } from "../api/types";
import { secs } from "../lib/format";
import { useStudio } from "../store";

const SPEEDS: [number, string][] = [
  [0.5, "½×"],
  [1, "1×"],
  [2, "2×"],
  [4, "4×"],
  [0, "Max"],
];

const STATE_TEXT: Record<string, string> = {
  starting: "Starting",
  running: "Running",
  paused: "Paused",
  stopping: "Stopping",
  settling: "Letting go",
  succeeded: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

export function RunPanel({ run }: { run: Run }) {
  const config = useStudio((s) => s.config);
  const speed = useStudio((s) => s.live?.speed ?? 1);
  const thinking = useStudio((s) => s.live?.thinking ?? false);
  const st = useStudio.getState();
  const active = ["starting", "running", "paused", "stopping", "settling"].includes(run.state);
  const phases = config?.phases ?? [];
  const leg = run.legs[run.leg_index];
  const phaseIdx = phases.findIndex((p) => p.id === run.phase);
  const done = run.legs.filter((l) => l.status === "done").length;
  const progress = run.legs.length ? (done + (active && phaseIdx >= 0 ? (phaseIdx + 1) / (phases.length + 1) : 0)) / run.legs.length : 0;
  const errors = run.legs.map((l) => l.result?.placement_error_mm).filter((v): v is number => typeof v === "number");

  return (
    <section className="panel-section panel-grow run-panel" aria-label="Task progress" aria-live="polite">
      <div className="section-head">
        <h2>{run.title}</h2>
        <span className={`chip chip-${run.state}`}>{STATE_TEXT[run.state] ?? run.state}</span>
      </div>
      <p className="muted">{run.summary}</p>
      <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
        <div style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <div className="run-meta">
        <span>
          Move {Math.min(run.leg_index + 1, run.legs.length)} of {run.legs.length}
        </span>
        <span>Simulated {secs(run.sim_time)}</span>
      </div>
      {active && (
        <div className="phase-now">
          <span className="pulse" aria-hidden="true" />
          {thinking ? "Planning a collision-free path…" : config?.phase_labels[run.phase] ?? run.phase}
          {leg && <span className="muted"> · {leg.object_name}</span>}
        </div>
      )}
      {active && (
        <ol className="phase-chips" aria-label="Steps of the current move">
          {phases.map((p, i) => (
            <li key={p.id} className={i < phaseIdx ? "pc-done" : i === phaseIdx ? "pc-now" : ""}>
              {p.label}
            </li>
          ))}
        </ol>
      )}
      <ol className="legs">
        {run.legs.map((l, i) => (
          <li key={i} className={`leg leg-${l.status}`}>
            <span className="leg-status" aria-hidden="true">
              {l.status === "done" ? "✓" : l.status === "failed" ? "✕" : l.status === "active" ? "●" : l.status === "stopped" ? "■" : "○"}
            </span>
            <span className="leg-text">
              {l.label}
              {l.result?.success && typeof l.result.placement_error_mm === "number" && (
                <span className="muted small"> · placed within {l.result.placement_error_mm.toFixed(1)} mm</span>
              )}
            </span>
          </li>
        ))}
      </ol>
      {run.failure && (
        <div className="plan plan-bad" role="alert">
          <p className="plan-title">{run.failure.message}</p>
          {run.failure.hint && <p>{run.failure.hint}</p>}
          <details>
            <summary>Details from the simulation</summary>
            <p className="mono small">
              {run.failure.stage}: {run.failure.reason}
            </p>
          </details>
        </div>
      )}
      {run.notices.map((n) => (
        <p key={n} className="notice">
          {n}
        </p>
      ))}
      {run.state === "succeeded" && (
        <div className="plan plan-ok">
          <p className="plan-title">Task completed in the simulation.</p>
          {errors.length > 0 && <p>Largest placement error: {Math.max(...errors).toFixed(1)} mm (limit 10 mm).</p>}
        </div>
      )}
      {active ? (
        <>
          <div className="task-actions">
            {run.state === "paused" ? (
              <button className="btn btn-primary" onClick={() => st.resume()}>
                Resume
              </button>
            ) : (
              <button className="btn" onClick={() => st.pause()} disabled={run.state !== "running"}>
                Pause
              </button>
            )}
            <button className="btn btn-danger" onClick={() => st.stop()} disabled={run.state === "stopping" || run.state === "settling"}>
              Stop
            </button>
          </div>
          <div className="field">
            <span>Speed</span>
            <div className="segmented" role="radiogroup" aria-label="Simulation speed">
              {SPEEDS.map(([v, l]) => (
                <button key={l} role="radio" aria-checked={speed === v} className={speed === v ? "seg seg-on" : "seg"} onClick={() => st.setSpeed(v)}>
                  {l}
                </button>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="task-actions">
          <button className="btn" onClick={() => st.restoreScene().then(() => st.dismissRun())}>
            Undo: put objects back
          </button>
          <button className="btn btn-primary" onClick={() => st.dismissRun()}>
            Done
          </button>
        </div>
      )}
      <details className="dev">
        <summary>Simulation details</summary>
        <ul className="small mono">
          <li>Physics: {config?.physics.engine}, {(config?.physics.timestep_s ?? 0) * 1000} ms steps</li>
          <li>Arm controller: {config?.controller === "torque_ctc" ? "computed torque + disturbance observer" : config?.controller}</li>
          {run.legs.map((l, i) =>
            l.result ? (
              <li key={i}>
                Move {i + 1}: {l.result.success ? "ok" : `failed at ${l.result.stage}`}, sim {l.result.sim_time_s} s, wall {l.result.wall_time_s} s,
                carried along {l.result.transport ?? "–"}
                {l.grasp ? `, grip ${Math.round(l.grasp.width_mm)} mm at ${Math.round((l.grasp.yaw * 180) / Math.PI)}°` : ""}
              </li>
            ) : null,
          )}
        </ul>
      </details>
    </section>
  );
}
