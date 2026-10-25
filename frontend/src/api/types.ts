// Types of the ArmStrong API (backend/robotics-lab/robotics-lab/studio/server.py).
// World frame: right-handed, z up, metres. Quaternions from the server are MuJoCo order (w, x, y, z).

export type Vec3 = [number, number, number];
export type QuatWXYZ = [number, number, number, number];
/** [x, y, z, qw, qx, qy, qz] */
export type Pose7 = [number, number, number, number, number, number, number];

export interface Problem {
  code: string;
  message: string;
  hint: string;
  object_id: string | null;
}

export interface World {
  table_height: number;
  table_radius: number;
  r_in: number;
  r_out: number;
  theta_max_deg: number;
}

export interface ObjectLimits {
  min_grasp_width_m: number;
  max_grasp_width_m: number;
  min_height_m: number;
  max_height_m: number;
  max_footprint_m: number;
  min_mass_kg: number;
  max_tilt_deg: number;
  gap_m: number;
}

export interface RobotProfile {
  id: string;
  name: string;
  payload_kg: number;
  datasheet_reach_m: number;
  gripper_stroke_m: number;
  dof: number;
  limits: ObjectLimits;
}

export type ParamType = "object" | "objects" | "point" | "choice";

export interface TaskParam {
  key: string;
  type: ParamType;
  label: string;
  choices?: [string, string][];
  default?: string;
}

export interface TaskDef {
  id: string;
  title: string;
  verb: string;
  summary: string;
  params: TaskParam[];
}

export interface StudioConfig {
  version: string;
  robot: RobotProfile;
  world: World;
  gap_m: number;
  tasks: TaskDef[];
  phases: { id: string; label: string }[];
  phase_labels: Record<string, string>;
  controller: string;
  physics: { timestep_s: number; engine: string };
  workspace: {
    measured_at: string;
    wall_s: number;
    profile: { theta_deg: number; r_min: number | null; r_max: number | null }[];
    probe_heights_m: number[];
    measured_theta_max_deg: number;
  };
  limits: { max_asset_mb: number };
}

/** How an uploaded mesh maps into the object's box (set by the upload dialog). */
export interface MeshPlacement {
  format: "glb" | "gltf" | "obj" | "stl";
  scale: number;
  up: "y" | "z";
  offset: Vec3;
}

export interface ObjectModel {
  name: string;
  size: Vec3; // w, d, h (m)
  mass: number; // kg
  color: string;
  asset_id: string | null;
  mesh: MeshPlacement | null;
}

export interface LibraryItem extends ObjectModel {
  id: string;
  builtin: boolean;
  created_at: number;
  problems: Problem[];
  can_lift: boolean;
}

export type ObjectState = "on_table" | "stacked" | "tipped" | "fallen" | "unsupported";

export interface SceneObject extends ObjectModel {
  id: string;
  pos: Vec3;
  quat: QuatWXYZ;
  template_id: string | null;
  yaw: number;
  tilt_deg: number;
  state: ObjectState;
  notes: string[];
  supports: string[];
  supporting: string[];
  problems: Problem[];
  can_lift: boolean;
}

export interface Scene {
  revision: number;
  objects: SceneObject[];
}

export interface PlanLeg {
  object_id: string;
  object_name: string;
  purpose: "move" | "buffer" | "via";
  label: string;
  from: Vec3;
  to: Vec3;
  ok: boolean;
  problems: Problem[];
  grasp: { yaw: number; width_mm: number; depth_mm: number; lift_mm: number } | null;
}

export interface TaskPlan {
  task_id: string;
  title: string;
  summary: string;
  params: Record<string, unknown>;
  ok: boolean;
  revision: number;
  legs: PlanLeg[];
  problems: Problem[];
  check_s: number;
  estimated_s: number;
}

export interface LegResult {
  success: boolean;
  stage: string | null;
  reason: string | null;
  message: string;
  hint: string;
  placement_error_mm: number | null;
  tilt_deg: number | null;
  sim_time_s: number;
  wall_time_s: number;
  controller: string;
  phases: { phase: string; t_start: number; t_end: number | null }[];
  transport: "arc" | "joint path" | null;
}

export type RunState = "starting" | "running" | "paused" | "stopping" | "settling" | "succeeded" | "failed" | "stopped";

export interface RunLeg extends PlanLeg {
  status: "pending" | "active" | "done" | "failed" | "stopped" | "skipped";
  result: LegResult | null;
}

export interface Run {
  id: string;
  task_id: string;
  title: string;
  summary: string;
  state: RunState;
  started_at: number;
  finished_at: number | null;
  leg_index: number;
  phase: string;
  sim_time: number;
  message: string;
  failure: { leg: number; stage: string; reason: string; message: string; hint: string } | null;
  notices: string[];
  legs: RunLeg[];
}

export interface Frame {
  seq: number;
  bodies: Record<string, Pose7>;
  objects: Record<string, Pose7>;
  gripper_width: number;
  live: boolean;
}

export interface LiveState {
  time: number;
  revision: number;
  run: Run | null;
  speed: number;
  thinking: boolean;
  robot_at_home: boolean;
  error: string | null;
  frame: Frame;
}

export interface GeometryGeom {
  body: string;
  name: string;
  type: "mesh" | "box" | "cylinder" | "sphere" | "capsule" | "ellipsoid" | "plane" | "unknown";
  size: Vec3;
  pos: Vec3;
  quat: QuatWXYZ;
  rgba: [number, number, number, number];
  material: string | null;
  specular: number | null;
  shininess: number | null;
  mesh?: number;
}

export interface GeometryMesh {
  name: string;
  vertex_count: number;
  index_count: number;
  positions: number;
  normals: number;
  indices: number;
  index_type: "uint16" | "uint32";
}

export interface GeometryManifest {
  bodies: string[];
  geoms: GeometryGeom[];
  meshes: GeometryMesh[];
  byte_length: number;
}

export interface ApiErrorBody {
  error: string;
  problems?: Problem[];
  plan?: TaskPlan;
}
