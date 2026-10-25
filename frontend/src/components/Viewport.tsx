import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { PlanLeg } from "../api/types";
import { isRunActive, useStudio } from "../store";
import { SceneView } from "../three/SceneView";
import { ObjectInspector } from "./ObjectInspector";

export function Viewport() {
  const host = useRef<HTMLDivElement>(null);
  const viewRef = useRef<SceneView | null>(null);
  const [loading, setLoading] = useState("Loading the robot model…");
  const config = useStudio((s) => s.config);
  const scene = useStudio((s) => s.scene);
  const frame = useStudio((s) => s.live?.frame);
  const run = useStudio((s) => s.live?.run ?? null);
  const selectedId = useStudio((s) => s.selectedId);
  const hoverId = useStudio((s) => s.hoverId);
  const plan = useStudio((s) => s.plan);
  const task = useStudio((s) => s.task);
  const viewRequest = useStudio((s) => s.viewRequest);
  const running = useStudio((s) => isRunActive(s));

  // Create the 3D view once.
  useEffect(() => {
    if (!host.current) return;
    const view = new SceneView(host.current, {
      canEdit: () => !isRunActive(useStudio.getState()),
      pickMode: () => {
        const st = useStudio.getState();
        if (!st.task.active || isRunActive(st)) return null;
        const def = st.config?.tasks.find((t) => t.id === st.task.id);
        const p = def?.params.find((x) => x.key === st.task.active);
        return p?.type === "object" || p?.type === "point" ? p.type : null;
      },
      pickSubject: () => {
        const st = useStudio.getState();
        const def = st.config?.tasks.find((t) => t.id === st.task.id);
        const key = def?.params.find((p) => p.type === "object")?.key;
        const id = key ? st.task.params[key] : undefined;
        return st.scene?.objects.find((o) => o.id === id) ?? null;
      },
      onSelect: (id) => useStudio.getState().select(id),
      onHover: (id) => useStudio.getState().setHover(id),
      onPickObject: (id) => {
        const st = useStudio.getState();
        const o = st.scene?.objects.find((x) => x.id === id);
        if (o && !o.can_lift) {
          st.toast("error", `${o.name} cannot be picked up.`, o.problems[0]?.message ?? o.notes[0]);
          return;
        }
        st.pickObject(id);
      },
      onPickPoint: (p) => useStudio.getState().pickPoint(p),
      onMove: (id, x, y) => useStudio.getState().moveObject(id, x, y),
      onDropTemplate: (item, p) => useStudio.getState().addFromLibrary(item.id, p),
      onIssue: (msg) => useStudio.getState().toast("error", msg),
    });
    viewRef.current = view;
    let cancelled = false;
    Promise.all([api.geometry(), api.geometryBlob()])
      .then(([manifest, blob]) => {
        if (cancelled) return;
        view.setRobot(manifest, blob);
        const fr = useStudio.getState().live?.frame;
        if (fr) view.setFrame(fr);
        setLoading("");
      })
      .catch((e) => setLoading(`Could not load the robot model: ${(e as Error).message}`));
    return () => {
      cancelled = true;
      view.dispose();
      viewRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (config) viewRef.current?.setWorld(config.world, config.gap_m);
  }, [config]);

  useEffect(() => {
    if (scene) viewRef.current?.setObjects(scene.objects);
  }, [scene]);

  useEffect(() => {
    if (frame) viewRef.current?.setFrame(frame);
  }, [frame]);

  useEffect(() => viewRef.current?.setSelection(selectedId), [selectedId]);
  useEffect(() => viewRef.current?.setHover(hoverId), [hoverId]);

  // Planned legs: the run's legs while a run is shown, otherwise the last check.
  useEffect(() => {
    const legs: PlanLeg[] | null = run ? run.legs : plan ? plan.legs : null;
    viewRef.current?.setPlan(legs, scene?.objects ?? [], run ? run.leg_index : -1);
  }, [run?.id, run?.leg_index, run?.state, plan, scene, run]);

  // Destination pins of point parameters.
  useEffect(() => {
    const def = config?.tasks.find((t) => t.id === task.id);
    const pts = (def?.params ?? [])
      .filter((p) => p.type === "point" && Array.isArray(task.params[p.key]))
      .map((p) => ({ p: task.params[p.key] as [number, number], label: p.label }));
    viewRef.current?.setMarkers(run ? [] : pts);
  }, [task, config, run]);

  useEffect(() => {
    if (viewRequest.n === 0) return;
    if (viewRequest.kind === "top") viewRef.current?.topView();
    else viewRef.current?.resetView();
  }, [viewRequest]);

  const activeParam = (() => {
    const def = config?.tasks.find((t) => t.id === task.id);
    return def?.params.find((p) => p.key === task.active) ?? null;
  })();

  return (
    <section className="viewport" aria-label="Workspace">
      <div className="viewport-host" ref={host} />
      {loading && (
        <div className="viewport-loading" role="status">
          {loading.startsWith("Could") ? null : <span className="spinner" aria-hidden="true" />}
          <span>{loading}</span>
        </div>
      )}
      <div className="viewport-tools" role="toolbar" aria-label="Camera">
        <button className="btn btn-ghost btn-sm" onClick={() => useStudio.getState().requestView("reset")}>
          Reset view
        </button>
        <button className="btn btn-ghost btn-sm" onClick={() => useStudio.getState().requestView("top")}>
          Top view
        </button>
      </div>
      {activeParam && !running && (
        <div className="viewport-hint viewport-hint-pick" role="status">
          {activeParam.type === "point"
            ? `Click on the table to choose: ${activeParam.label.toLowerCase()}`
            : `Click an object in the view to choose: ${activeParam.label.toLowerCase()}`}
          <button className="btn btn-ghost btn-sm" onClick={() => useStudio.getState().setActiveParam(null)}>
            Cancel
          </button>
        </div>
      )}
      {!activeParam && !running && scene && scene.objects.length === 0 && (
        <div className="viewport-hint" role="status">Drag an object from the library onto the table to get started.</div>
      )}
      {running && <div className="viewport-hint viewport-hint-live" role="status">Simulation running · editing is paused</div>}
      <div className="viewport-legend" aria-hidden="true">
        <span className="legend-swatch legend-zone" /> Reachable area
        <span className="legend-swatch legend-clear" /> Not usable
      </div>
      <ObjectInspector />
    </section>
  );
}
