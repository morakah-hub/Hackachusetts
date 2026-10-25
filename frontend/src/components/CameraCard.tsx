import { useEffect, useState } from "react";
import type { SceneObject, SimEvent } from "../api/types";
import { isRunActive, useStudio } from "../store";

const NO_EVENTS: SimEvent[] = [];
const NO_OBJECTS: SceneObject[] = [];

/** Live RGB-D camera feed, how objects are located (camera or simulator), the last localization and the
 *  developer tools (fault injection, event log). */
export function CameraCard() {
  const camera = useStudio((s) => s.live?.camera ?? null);
  const mode = useStudio((s) => s.live?.perception_mode ?? "camera_rgbd");
  const estimate = useStudio((s) => s.live?.estimate ?? null);
  const faults = useStudio((s) => s.live?.faults);
  const events = useStudio((s) => s.live?.events ?? NO_EVENTS);
  const objects = useStudio((s) => s.scene?.objects ?? NO_OBJECTS);
  const running = useStudio((s) => isRunActive(s));
  const st = useStudio.getState();
  const [view, setView] = useState<"rgb" | "depth">("rgb");
  const [open, setOpen] = useState(true);
  const [src, setSrc] = useState("");
  const frameId = camera?.frame_id;

  useEffect(() => {
    if (!open || frameId === undefined) return;
    setSrc(`/api/camera/${view}?f=${frameId}`);
  }, [frameId, view, open]);

  const name = estimate ? objects.find((o) => o.id === estimate.id)?.name ?? estimate.id : null;

  return (
    <section className="panel-section camera-card" aria-label="Camera">
      <div className="section-head">
        <h2>Camera</h2>
        <button className="btn btn-ghost btn-sm" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? "Hide" : "Show"}
        </button>
      </div>
      {open && (
        <>
          <div className="cam-frame">
            {src ? <img src={src} alt={view === "rgb" ? "Live colour image from the simulated camera" : "Live depth image from the simulated camera"} /> : <span className="muted small">Waiting for the camera…</span>}
            <div className="segmented cam-toggle" role="radiogroup" aria-label="Image">
              {(["rgb", "depth"] as const).map((v) => (
                <button key={v} role="radio" aria-checked={view === v} className={view === v ? "seg seg-on" : "seg"} onClick={() => setView(v)}>
                  {v === "rgb" ? "Colour" : "Depth"}
                </button>
              ))}
            </div>
          </div>
          {camera && (
            <p className="muted small mono">
              frame {camera.frame_id} · {camera.width}×{camera.height}
              {camera.masked ? " · MASKED" : ""}
            </p>
          )}
        </>
      )}
      <div className="field">
        <span>Locate objects with</span>
        <div className="segmented" role="radiogroup" aria-label="How the robot finds objects">
          <button role="radio" aria-checked={mode === "camera_rgbd"} className={mode === "camera_rgbd" ? "seg seg-on" : "seg"} disabled={running} onClick={() => st.setPerception("camera_rgbd")}>
            Camera
          </button>
          <button role="radio" aria-checked={mode === "known_world"} className={mode === "known_world" ? "seg seg-on" : "seg"} disabled={running} onClick={() => st.setPerception("known_world")}>
            Simulator (dev)
          </button>
        </div>
      </div>
      {estimate && (
        <p className={`small ${estimate.valid ? "" : "text-bad"}`} aria-live="polite">
          {estimate.valid && estimate.position_m
            ? `Camera found ${name} at ${Math.round(estimate.position_m[0] * 1000)}, ${Math.round(estimate.position_m[1] * 1000)} mm (${estimate.pixels} px).`
            : `Camera could not find ${name}: ${estimate.reasons[0] ?? "unknown reason"}`}
        </p>
      )}
      <details className="dev">
        <summary>Developer tools</summary>
        <label className="check-item">
          <input type="checkbox" checked={!!faults?.camera_masked} disabled={running} onChange={(e) => st.setFaults({ camera_masked: e.target.checked })} />
          Cover the camera (test: the robot must not move blind)
        </label>
        <label className="check-item">
          <input type="checkbox" checked={!!faults?.gripper_disabled} disabled={running} onChange={(e) => st.setFaults({ gripper_disabled: e.target.checked })} />
          Disable the gripper (test: the task must fail safely)
        </label>
        <ol className="event-log small mono" aria-label="Event log">
          {events
            .slice()
            .reverse()
            .map((e, i) => (
              <li key={`${e.t_wall}-${i}`} className={`ev ev-${e.level}`}>
                <span className="muted">{e.t_sim.toFixed(1)} s</span> {e.msg}
              </li>
            ))}
        </ol>
      </details>
    </section>
  );
}
