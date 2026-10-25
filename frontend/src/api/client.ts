import type {
  ApiErrorBody,
  GeometryManifest,
  LibraryItem,
  LiveState,
  ObjectModel,
  PerceptionMode,
  Problem,
  Run,
  Scene,
  SceneObject,
  StudioConfig,
  TaskPlan,
} from "./types";

/** An API call that the server rejected, with its explanation. */
export class ApiError extends Error {
  status: number;
  problems: Problem[];
  body: ApiErrorBody | null;

  constructor(status: number, body: ApiErrorBody | null, fallback: string) {
    super(body?.error ?? fallback);
    this.status = status;
    this.problems = body?.problems ?? [];
    this.body = body;
  }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      signal,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, null, "Cannot reach the simulation server. Is the Armstrong Studio window still open?");
  }
  if (!res.ok) {
    let parsed: ApiErrorBody | null = null;
    try {
      parsed = (await res.json()) as ApiErrorBody;
      if (typeof parsed?.error !== "string" && typeof (parsed as { detail?: unknown })?.detail === "string") {
        parsed = { error: String((parsed as { detail?: unknown }).detail) };
      }
    } catch {
      parsed = null;
    }
    throw new ApiError(res.status, parsed, `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export const api = {
  config: () => request<StudioConfig>("GET", "/api/config"),
  geometry: () => request<GeometryManifest>("GET", "/api/geometry"),
  async geometryBlob(): Promise<ArrayBuffer> {
    const res = await fetch("/api/geometry.bin");
    if (!res.ok) throw new ApiError(res.status, null, "Could not load the robot model");
    return res.arrayBuffer();
  },
  scene: () => request<Scene>("GET", "/api/scene"),
  state: (signal?: AbortSignal) => request<LiveState>("GET", "/api/state", undefined, signal),
  library: () => request<LibraryItem[]>("GET", "/api/library"),

  addObject: (body: { template_id?: string; x?: number; y?: number; yaw?: number } & Partial<ObjectModel>) =>
    request<SceneObject>("POST", "/api/scene/objects", body),
  updateObject: (id: string, body: Partial<{ x: number; y: number; yaw: number; name: string; mass: number; color: string }>) =>
    request<SceneObject>("PATCH", `/api/scene/objects/${encodeURIComponent(id)}`, body),
  deleteObject: (id: string) => request<{ ok: boolean }>("DELETE", `/api/scene/objects/${encodeURIComponent(id)}`),
  resetScene: (mode: "default" | "empty" | "objects") => request<Scene>("POST", "/api/scene/reset", { mode }),
  restoreScene: () => request<Scene>("POST", "/api/scene/restore"),

  addLibraryItem: (body: ObjectModel) => request<LibraryItem>("POST", "/api/library", body),
  deleteLibraryItem: (id: string) => request<{ ok: boolean }>("DELETE", `/api/library/${encodeURIComponent(id)}`),
  async uploadAsset(file: File): Promise<{ id: string; filename: string; bytes: number }> {
    let res: Response;
    try {
      res = await fetch(`/api/assets?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file,
      });
    } catch {
      throw new ApiError(0, null, "Cannot reach the studio server.");
    }
    if (!res.ok) {
      let parsed: ApiErrorBody | null = null;
      try {
        parsed = (await res.json()) as ApiErrorBody;
      } catch {
        parsed = null;
      }
      throw new ApiError(res.status, parsed, "Upload failed");
    }
    return res.json();
  },

  checkTask: (task_id: string, params: Record<string, unknown>) =>
    request<TaskPlan>("POST", "/api/tasks/check", { task_id, params }),
  runTask: (task_id: string, params: Record<string, unknown>) =>
    request<{ plan: TaskPlan; run: Run }>("POST", "/api/tasks/run", { task_id, params }),
  pause: () => request("POST", "/api/run/pause"),
  resume: () => request("POST", "/api/run/resume"),
  stop: () => request("POST", "/api/run/stop"),
  dismiss: () => request("POST", "/api/run/dismiss"),
  speed: (value: number) => request("POST", "/api/run/speed", { value }),
  resetRobot: () => request("POST", "/api/robot/reset"),
  setPerception: (mode: PerceptionMode) => request<{ perception_mode: PerceptionMode }>("POST", "/api/settings", { perception_mode: mode }),
  setFaults: (body: { camera_masked?: boolean; gripper_disabled?: boolean }) => request("POST", "/api/faults", body),
  exportUrl: (runId: string) => `/api/runs/${encodeURIComponent(runId)}/export`,
};
