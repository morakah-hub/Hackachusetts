import { create } from "zustand";
import { api, ApiError } from "./api/client";
import type { LibraryItem, LiveState, ObjectModel, PerceptionMode, Scene, StudioConfig, TaskDef, TaskParam, TaskPlan } from "./api/types";
import { findFreeSpot } from "./lib/geom";

export interface Toast {
  id: number;
  kind: "info" | "error" | "success";
  message: string;
  hint?: string;
}

type Connection = "connecting" | "online" | "offline";

interface TaskForm {
  id: string | null;
  params: Record<string, unknown>;
  /** Parameter currently filled by clicking in the 3D view. */
  active: string | null;
}

interface StudioState {
  config: StudioConfig | null;
  scene: Scene | null;
  library: LibraryItem[];
  live: LiveState | null;
  connection: Connection;
  loadError: string | null;

  selectedId: string | null;
  hoverId: string | null;
  dialog: null | "upload" | "custom";
  task: TaskForm;
  plan: TaskPlan | null;
  planning: boolean;
  starting: boolean;
  toasts: Toast[];
  /** Bumped to ask the 3D view to frame the table again. */
  viewRequest: { kind: "reset" | "top"; n: number };
  /** The "How to use" guide. */
  helpOpen: boolean;

  init(): Promise<void>;
  refreshScene(): Promise<void>;
  refreshLibrary(): Promise<void>;
  select(id: string | null): void;
  setHover(id: string | null): void;
  openDialog(d: StudioState["dialog"]): void;

  addFromLibrary(templateId: string, at?: [number, number]): Promise<void>;
  moveObject(id: string, x: number, y: number, yaw?: number): Promise<boolean>;
  rotateObject(id: string, delta: number): Promise<void>;
  updateObject(id: string, patch: Partial<{ name: string; mass: number; color: string; x: number; y: number }>): Promise<void>;
  deleteObject(id: string): Promise<void>;
  addLibraryItem(model: ObjectModel): Promise<LibraryItem | null>;
  deleteLibraryItem(id: string): Promise<void>;

  setTask(id: string | null): void;
  setParam(key: string, value: unknown): void;
  setActiveParam(key: string | null): void;
  pickObject(id: string): boolean;
  pickPoint(p: [number, number]): boolean;
  checkTask(): Promise<TaskPlan | null>;
  runTask(): Promise<void>;

  pause(): Promise<void>;
  resume(): Promise<void>;
  stop(): Promise<void>;
  dismissRun(): Promise<void>;
  setSpeed(v: number): Promise<void>;
  resetRobot(): Promise<void>;
  resetScene(mode: "default" | "empty" | "objects"): Promise<void>;
  setPerception(mode: PerceptionMode): Promise<void>;
  setFaults(body: { camera_masked?: boolean; gripper_disabled?: boolean }): Promise<void>;
  setHelp(open: boolean): void;
  restoreScene(): Promise<void>;
  requestView(kind: "reset" | "top"): void;

  toast(kind: Toast["kind"], message: string, hint?: string): void;
  dismissToast(id: number): void;
}

let toastId = 0;
let pollTimer: number | undefined;
let initStarted = false;

function errorToast(get: () => StudioState, err: unknown, fallback = "Something went wrong") {
  if (err instanceof ApiError) {
    const p = err.problems[0];
    get().toast("error", err.message || fallback, p?.hint || undefined);
  } else {
    get().toast("error", fallback, String(err));
  }
}

/** Required parameters that are filled by clicking in the 3D view (object or table spot). */
export const needsPick = (p: TaskParam) => (p.type === "object" || p.type === "point") && !p.optional;

export const isRunActive = (s: Pick<StudioState, "live">) => {
  const st = s.live?.run?.state;
  return st === "starting" || st === "running" || st === "paused" || st === "stopping" || st === "settling";
};

export const useStudio = create<StudioState>((set, get) => ({
  config: null,
  scene: null,
  library: [],
  live: null,
  connection: "connecting",
  loadError: null,
  selectedId: null,
  hoverId: null,
  dialog: null,
  task: { id: null, params: {}, active: null },
  plan: null,
  planning: false,
  starting: false,
  toasts: [],
  viewRequest: { kind: "reset", n: 0 },
  helpOpen: (() => {
    try {
      if (new URLSearchParams(window.location.search).has("nohelp")) return false;
      return window.localStorage.getItem("armstrong.helpSeen") !== "1";
    } catch {
      return true;
    }
  })(),

  async init() {
    if (initStarted) return;
    initStarted = true;
    for (;;) {
      try {
        const [config, scene, library] = await Promise.all([api.config(), api.scene(), api.library()]);
        set({ config, scene, library, connection: "online", loadError: null });
        break;
      } catch (err) {
        set({ connection: "offline", loadError: (err as Error).message });
        await new Promise((r) => window.setTimeout(r, 2000));
      }
    }
    const poll = async () => {
      let next = 400;
      try {
        const live = await api.state();
        const prev = get();
        const changed = !prev.scene || live.revision !== prev.scene.revision;
        set({ live, connection: "online" });
        if (changed) await get().refreshScene();
        if (live.frame.live || isRunActive({ live })) next = 33;
      } catch {
        set({ connection: "offline" });
        next = 1500;
      }
      pollTimer = window.setTimeout(poll, next);
    };
    window.clearTimeout(pollTimer);
    poll();
  },

  async refreshScene() {
    try {
      const scene = await api.scene();
      const { selectedId, plan } = get();
      set({
        scene,
        selectedId: selectedId && scene.objects.some((o) => o.id === selectedId) ? selectedId : null,
        plan: plan && plan.revision === scene.revision ? plan : null,
      });
    } catch (err) {
      errorToast(get, err, "Could not load the scene");
    }
  },

  async refreshLibrary() {
    try {
      set({ library: await api.library() });
    } catch (err) {
      errorToast(get, err, "Could not load the library");
    }
  },

  select: (id) => set({ selectedId: id }),
  setHover: (id) => set({ hoverId: id }),
  openDialog: (d) => set({ dialog: d }),

  async addFromLibrary(templateId, at) {
    const { config, scene, library } = get();
    if (!config || !scene) return;
    const item = library.find((i) => i.id === templateId);
    let spot = at ?? null;
    if (!spot && item) spot = findFreeSpot(config.world, config.gap_m, item.size, scene.objects);
    if (!spot) {
      get().toast("error", "There is no free spot on the table for this object.", "Move or remove some objects first.");
      return;
    }
    try {
      const obj = await api.addObject({ template_id: templateId, x: spot[0], y: spot[1], yaw: 0 });
      await get().refreshScene();
      set({ selectedId: obj.id });
    } catch (err) {
      errorToast(get, err, "Could not place the object");
    }
  },

  async moveObject(id, x, y, yaw) {
    try {
      await api.updateObject(id, yaw === undefined ? { x, y } : { x, y, yaw });
      await get().refreshScene();
      return true;
    } catch (err) {
      errorToast(get, err, "Could not move the object");
      await get().refreshScene();
      return false;
    }
  },

  async rotateObject(id, delta) {
    const o = get().scene?.objects.find((s) => s.id === id);
    if (!o) return;
    try {
      await api.updateObject(id, { x: o.pos[0], y: o.pos[1], yaw: o.yaw + delta });
      await get().refreshScene();
    } catch (err) {
      errorToast(get, err, "Could not rotate the object");
    }
  },

  async updateObject(id, patch) {
    try {
      await api.updateObject(id, patch);
      await get().refreshScene();
    } catch (err) {
      errorToast(get, err, "Could not change the object");
    }
  },

  async deleteObject(id) {
    try {
      await api.deleteObject(id);
      const params = { ...get().task.params };
      for (const [k, v] of Object.entries(params)) {
        if (v === id) delete params[k];
        if (Array.isArray(v)) params[k] = v.filter((x) => x !== id);
      }
      set({ selectedId: null, task: { ...get().task, params } });
      await get().refreshScene();
    } catch (err) {
      errorToast(get, err, "Could not remove the object");
    }
  },

  async addLibraryItem(model) {
    try {
      const item = await api.addLibraryItem(model);
      await get().refreshLibrary();
      return item;
    } catch (err) {
      errorToast(get, err, "Could not add the object to the library");
      return null;
    }
  },

  async deleteLibraryItem(id) {
    try {
      await api.deleteLibraryItem(id);
      await get().refreshLibrary();
    } catch (err) {
      errorToast(get, err, "Could not delete the library item");
    }
  },

  setTask(id) {
    const def = get().config?.tasks.find((t) => t.id === id) as TaskDef | undefined;
    const params: Record<string, unknown> = {};
    def?.params.forEach((p) => {
      if (p.type === "choice" && p.default) params[p.key] = p.default;
    });
    const sel = get().selectedId;
    const first = def?.params.find((p) => p.type === "object");
    if (first && sel) {
      const o = get().scene?.objects.find((s) => s.id === sel);
      if (o?.can_lift) params[first.key] = sel;
    }
    const active = def?.params.find((p) => needsPick(p) && params[p.key] === undefined)?.key ?? null;
    set({ task: { id, params, active }, plan: null });
  },

  setParam(key, value) {
    const t = get().task;
    const params = { ...t.params, [key]: value };
    const def = get().config?.tasks.find((d) => d.id === t.id);
    const next = def?.params.find((p) => needsPick(p) && params[p.key] === undefined)?.key ?? null;
    set({ task: { ...t, params, active: t.active === key ? next : t.active }, plan: null });
  },

  setActiveParam(key) {
    set({ task: { ...get().task, active: key } });
  },

  pickObject(id) {
    const { task, config } = get();
    if (!task.active) return false;
    const def = config?.tasks.find((d) => d.id === task.id);
    const p = def?.params.find((x) => x.key === task.active);
    if (!p || p.type !== "object") return false;
    get().setParam(p.key, id);
    return true;
  },

  pickPoint(pt) {
    const { task, config } = get();
    if (!task.active) return false;
    const def = config?.tasks.find((d) => d.id === task.id);
    const p = def?.params.find((x) => x.key === task.active);
    if (!p || p.type !== "point") return false;
    get().setParam(p.key, pt);
    return true;
  },

  async checkTask() {
    const { task } = get();
    if (!task.id) return null;
    set({ planning: true });
    try {
      const plan = await api.checkTask(task.id, task.params);
      set({ plan });
      return plan;
    } catch (err) {
      set({ plan: null });
      errorToast(get, err, "Could not check the task");
      return null;
    } finally {
      set({ planning: false });
    }
  },

  async runTask() {
    const { task } = get();
    if (!task.id) return;
    set({ starting: true });
    try {
      const res = await api.runTask(task.id, task.params);
      set({ plan: res.plan, selectedId: null, task: { ...get().task, active: null } });
    } catch (err) {
      if (err instanceof ApiError && err.body?.plan) set({ plan: err.body.plan });
      errorToast(get, err, "Could not start the task");
    } finally {
      set({ starting: false });
    }
  },

  async pause() {
    await api.pause().catch((e) => errorToast(get, e));
  },
  async resume() {
    await api.resume().catch((e) => errorToast(get, e));
  },
  async stop() {
    await api.stop().catch((e) => errorToast(get, e));
  },
  async dismissRun() {
    await api.dismiss().catch((e) => errorToast(get, e));
    set({ plan: null });
  },
  async setSpeed(v) {
    await api.speed(v).catch((e) => errorToast(get, e));
  },
  async resetRobot() {
    try {
      await api.resetRobot();
      get().toast("info", "The robot is back at its rest position.");
    } catch (err) {
      errorToast(get, err);
    }
  },
  async resetScene(mode) {
    try {
      await api.resetScene(mode);
      set({ selectedId: null, plan: null, task: { id: get().task.id, params: {}, active: null } });
      await get().refreshScene();
    } catch (err) {
      errorToast(get, err, "Could not reset the table");
    }
  },
  async restoreScene() {
    try {
      await api.restoreScene();
      await get().refreshScene();
      get().toast("success", "Objects are back where they were before the task.");
    } catch (err) {
      errorToast(get, err, "Could not restore the layout");
    }
  },
  async setPerception(mode) {
    try {
      await api.setPerception(mode);
      get().toast("info", mode === "camera_rgbd" ? "Objects are now located with the RGB-D camera." : "Objects are now located from the simulator (development mode).");
    } catch (err) {
      errorToast(get, err, "Could not change the setting");
    }
  },
  async setFaults(body) {
    await api.setFaults(body).catch((e) => errorToast(get, e));
  },
  setHelp(open) {
    if (!open) {
      try {
        window.localStorage.setItem("armstrong.helpSeen", "1");
      } catch {
        /* private mode: the guide just shows again next time */
      }
    }
    set({ helpOpen: open });
  },
  requestView(kind) {
    set({ viewRequest: { kind, n: get().viewRequest.n + 1 } });
  },

  toast(kind, message, hint) {
    const id = ++toastId;
    set({ toasts: [...get().toasts, { id, kind, message, hint }].slice(-4) });
    window.setTimeout(() => get().dismissToast(id), kind === "error" ? 7000 : 4000);
  },
  dismissToast(id) {
    set({ toasts: get().toasts.filter((t) => t.id !== id) });
  },
}));
