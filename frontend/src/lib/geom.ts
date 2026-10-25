// Client-side copies of the server's placement rules (studio/scene.py), used only for instant feedback while
// dragging. The server re-checks every edit and its answer is the one that counts.
import type { ObjectLimits, QuatWXYZ, SceneObject, Vec3, World } from "../api/types";

export type Pt = [number, number];

export function footprint(x: number, y: number, yaw: number, size: Vec3, grow = 0): Pt[] {
  const hw = size[0] / 2 + grow;
  const hd = size[1] / 2 + grow;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [
    [hw, hd],
    [-hw, hd],
    [-hw, -hd],
    [hw, -hd],
  ].map(([lx, ly]) => [x + lx * c - ly * s, y + lx * s + ly * c] as Pt);
}

export function polygonsOverlap(a: Pt[], b: Pt[]): boolean {
  for (const poly of [a, b]) {
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i];
      const q = poly[(i + 1) % poly.length];
      const ax = -(q[1] - p[1]);
      const ay = q[0] - p[0];
      let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
      for (const v of a) {
        const d = v[0] * ax + v[1] * ay;
        aMin = Math.min(aMin, d);
        aMax = Math.max(aMax, d);
      }
      for (const v of b) {
        const d = v[0] * ax + v[1] * ay;
        bMin = Math.min(bMin, d);
        bMax = Math.max(bMax, d);
      }
      if (aMax < bMin || bMax < aMin) return false;
    }
  }
  return true;
}

export function yawOfQuat(q: QuatWXYZ): number {
  const [w, x, y, z] = q;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

/** Why an upright object of this size cannot stand at (x, y); null if it can. */
export function placementIssue(
  world: World,
  gap: number,
  x: number,
  y: number,
  yaw: number,
  size: Vec3,
  others: SceneObject[],
  excludeId?: string,
): string | null {
  const r = Math.hypot(x, y);
  const th = (Math.atan2(y, x) * 180) / Math.PI;
  if (r < world.r_in) return "Too close to the robot base";
  if (r > world.r_out) return "Out of the arm's reach";
  if (Math.abs(th) > world.theta_max_deg + 1e-6) return "Behind the robot (kept clear)";
  const corners = footprint(x, y, yaw, size);
  if (corners.some(([cx, cy]) => Math.hypot(cx, cy) > world.table_radius - 0.005)) return "Over the table edge";
  const fp = footprint(x, y, yaw, size, gap / 2);
  const bottom = world.table_height;
  const top = bottom + size[2];
  for (const o of others) {
    if (o.id === excludeId) continue;
    const oh = o.size[2] / 2;
    if (o.pos[2] - oh > top || o.pos[2] + oh < bottom) continue;
    if (polygonsOverlap(fp, footprint(o.pos[0], o.pos[1], o.yaw, o.size, gap / 2))) return `Too close to ${o.name}`;
  }
  return null;
}

/** Can the gripper close on this box at all? Mirrors RobotRules.lift_problems. */
export function liftIssues(size: Vec3, mass: number, payload: number, lim: ObjectLimits): string[] {
  const out: string[] = [];
  if (mass > payload) out.push(`Too heavy: ${mass.toFixed(2)} kg is over the ${payload.toFixed(1)} kg payload`);
  const [w, d, h] = size;
  const lo = lim.min_grasp_width_m;
  const hi = lim.max_grasp_width_m;
  if (!(w >= lo && w <= hi) && !(d >= lo && d <= hi)) {
    const narrow = Math.min(w, d);
    if (narrow > hi) out.push(`Too wide for the gripper: ${Math.round(narrow * 1000)} mm at its narrowest side (max ${Math.round(hi * 1000)} mm)`);
    else if (Math.max(w, d) < lo) out.push(`Too thin for the gripper (under ${Math.round(lo * 1000)} mm)`);
    else out.push("No side fits between the fingers");
  }
  if (h < lim.min_height_m) out.push(`Too short: under ${Math.round(lim.min_height_m * 1000)} mm tall`);
  if (h > lim.max_height_m) out.push(`Too tall: over ${Math.round(lim.max_height_m * 1000)} mm`);
  return out;
}

/** A free spot for an object, searched in rings from the front of the table. */
export function findFreeSpot(world: World, gap: number, size: Vec3, others: SceneObject[]): Pt | null {
  const mid = (world.r_in + world.r_out) / 2;
  const radii: number[] = [];
  for (let dr = 0; dr <= world.r_out - world.r_in; dr += 0.035) {
    radii.push(mid - dr, mid + dr);
  }
  for (let dth = 0; dth <= world.theta_max_deg; dth += 7.5) {
    for (const sgn of dth === 0 ? [1] : [1, -1]) {
      const th = (sgn * dth * Math.PI) / 180;
      for (const r of radii) {
        if (r < world.r_in || r > world.r_out) continue;
        const x = r * Math.cos(th);
        const y = r * Math.sin(th);
        if (!placementIssue(world, gap, x, y, 0, size, others)) return [x, y];
      }
    }
  }
  return null;
}
