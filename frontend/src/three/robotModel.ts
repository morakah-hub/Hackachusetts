import * as THREE from "three";
import type { GeometryGeom, GeometryManifest, QuatWXYZ } from "../api/types";

/** MuJoCo (w, x, y, z) -> three.js quaternion. */
export function quatFromWXYZ(q: QuatWXYZ | number[], out = new THREE.Quaternion()): THREE.Quaternion {
  return out.set(q[1], q[2], q[3], q[0]);
}

// The Menagerie materials, given physically based parameters for the browser.
const ROBOT_MATERIALS: Record<string, { color: string; roughness: number; metalness: number; emissive?: string }> = {
  white: { color: "#f3f3f1", roughness: 0.34, metalness: 0.0 },
  off_white: { color: "#e2e5e7", roughness: 0.42, metalness: 0.0 },
  black: { color: "#2e3033", roughness: 0.55, metalness: 0.05 },
  light_blue: { color: "#0f86c4", roughness: 0.4, metalness: 0.05 },
  green: { color: "#37c96f", roughness: 0.45, metalness: 0.0, emissive: "#0c3d1f" },
};

const TABLE_MATERIALS: Record<string, { color: string; roughness: number; metalness: number }> = {
  tabletop: { color: "#b9b4ab", roughness: 0.7, metalness: 0.0 },
  pedestal: { color: "#3a3d42", roughness: 0.5, metalness: 0.35 },
};

function materialFor(g: GeometryGeom): THREE.Material {
  const preset = (g.material && (ROBOT_MATERIALS[g.material] ?? TABLE_MATERIALS[g.material])) || null;
  if (preset) {
    return new THREE.MeshStandardMaterial({
      color: preset.color,
      roughness: preset.roughness,
      metalness: preset.metalness,
      emissive: "emissive" in preset && preset.emissive ? preset.emissive : "#000000",
    });
  }
  const [r, gg, b, a] = g.rgba;
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(r, gg, b, THREE.SRGBColorSpace),
    roughness: 0.5,
    transparent: a < 1,
    opacity: a,
  });
}

function primitive(g: GeometryGeom): THREE.BufferGeometry | null {
  const [a, b, c] = g.size;
  switch (g.type) {
    case "box":
      return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "sphere":
      return new THREE.SphereGeometry(a, 32, 16);
    case "cylinder":
      return new THREE.CylinderGeometry(a, a, 2 * b, 96, 1).rotateX(Math.PI / 2); // MuJoCo: along local z
    case "capsule":
      return new THREE.CapsuleGeometry(a, 2 * b, 8, 24).rotateX(Math.PI / 2);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 32, 16).scale(a, b, c);
    default:
      return null;
  }
}

export interface RobotModel {
  root: THREE.Group;
  bodies: Map<string, THREE.Group>;
  triangles: number;
}

/** Build one group per MuJoCo body with its visual geoms; the caller sets the body poses every frame. */
export function buildRobotModel(manifest: GeometryManifest, blob: ArrayBuffer): RobotModel {
  const root = new THREE.Group();
  root.name = "simulation-bodies";
  const bodies = new Map<string, THREE.Group>();
  for (const name of manifest.bodies) {
    const g = new THREE.Group();
    g.name = name;
    g.matrixAutoUpdate = true;
    bodies.set(name, g);
    root.add(g);
  }
  const geoms: THREE.BufferGeometry[] = manifest.meshes.map((m) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(blob, m.positions, m.vertex_count * 3), 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(blob, m.normals, m.vertex_count * 3), 3, true));
    const idx = m.index_type === "uint16"
      ? new Uint16Array(blob, m.indices, m.index_count)
      : new Uint32Array(blob, m.indices, m.index_count);
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeBoundingSphere();
    return geo;
  });
  const materials = new Map<string, THREE.Material>();
  let triangles = 0;
  for (const g of manifest.geoms) {
    const body = bodies.get(g.body);
    if (!body) continue;
    const geometry = g.type === "mesh" && g.mesh !== undefined ? geoms[g.mesh] : primitive(g);
    if (!geometry) continue;
    const key = g.material ?? `rgba:${g.rgba.join(",")}`;
    let mat = materials.get(key);
    if (!mat) {
      mat = materialFor(g);
      materials.set(key, mat);
    }
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
    quatFromWXYZ(g.quat, mesh.quaternion);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.name = g.name || `${g.body}-geom`;
    mesh.userData.body = g.body;
    if (g.body === "table") mesh.userData.table = true;
    body.add(mesh);
    triangles += (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
  }
  return { root, bodies, triangles };
}
