import type { Vec3 } from "../api/types";

export const mm = (m: number) => `${Math.round(m * 1000)} mm`;
export const dims = (s: Vec3) => `${Math.round(s[0] * 1000)} × ${Math.round(s[1] * 1000)} × ${Math.round(s[2] * 1000)} mm`;
export const kg = (v: number) => (v < 1 ? `${Math.round(v * 1000)} g` : `${v.toFixed(2)} kg`);
export const deg = (rad: number) => `${Math.round((rad * 180) / Math.PI)}°`;
export const secs = (s: number) => (s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

export function wrapAngle(a: number): number {
  return ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
}
