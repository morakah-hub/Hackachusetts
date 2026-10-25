// Loading user meshes (GLB/glTF, OBJ, STL) and fitting them into an object's box proxy.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import type { MeshPlacement, Vec3 } from "../api/types";

export type MeshFormat = MeshPlacement["format"];

export function formatOf(filename: string): MeshFormat | null {
  const ext = filename.toLowerCase().split(".").pop();
  return ext === "glb" || ext === "gltf" || ext === "obj" || ext === "stl" ? ext : null;
}

function plainMaterial(color: string): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
}

/** Parse a mesh file into an Object3D in the file's own units and axes. */
export async function parseMesh(data: ArrayBuffer, format: MeshFormat, color = "#b8bec6"): Promise<THREE.Object3D> {
  if (format === "glb" || format === "gltf") {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const payload = format === "gltf" ? new TextDecoder().decode(data) : data;
    const gltf = await new Promise<{ scene: THREE.Object3D }>((resolve, reject) => {
      loader.parse(payload, "", (g) => resolve(g), (e) => reject(e));
    }).catch((e: unknown) => {
      const msg = String((e as Error)?.message ?? e);
      if (/draco|ktx|basis/i.test(msg)) {
        throw new Error("This file uses compression (Draco/KTX2) that the studio does not decode. Export it without compression.");
      }
      if (format === "gltf") throw new Error("Could not read the .gltf file. Use a self-contained .glb (external files are not supported).");
      throw new Error(`Could not read the file: ${msg}`);
    });
    return gltf.scene;
  }
  if (format === "obj") {
    const obj = new OBJLoader().parse(new TextDecoder().decode(data));
    obj.traverse((c) => {
      if ((c as THREE.Mesh).isMesh) {
        const m = c as THREE.Mesh;
        m.material = plainMaterial(color);
        if (!m.geometry.attributes.normal) m.geometry.computeVertexNormals();
      }
    });
    return obj;
  }
  const geo = new STLLoader().parse(data);
  if (!geo.attributes.normal) geo.computeVertexNormals();
  return new THREE.Mesh(geo, plainMaterial(color));
}

/** Units a file is probably written in, from its size. */
export function guessUnits(maxExtent: number): number {
  if (maxExtent > 20) return 0.001; // millimetres
  if (maxExtent > 2) return 0.01; // centimetres
  return 1; // metres
}

/** Rotation that turns the file's up axis into world z. */
export function upRotation(up: "y" | "z"): THREE.Quaternion {
  return up === "y" ? new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2) : new THREE.Quaternion();
}

/** Box size and centring offset of a mesh after scaling and turning it upright. */
export function fitBox(obj: THREE.Object3D, scale: number, up: "y" | "z"): { size: Vec3; offset: Vec3 } {
  const holder = new THREE.Group();
  const inner = obj.clone(true);
  inner.scale.setScalar(scale);
  inner.quaternion.copy(upRotation(up));
  holder.add(inner);
  holder.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(holder, true);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  return { size: [size.x, size.y, size.z], offset: [-center.x, -center.y, -center.z] };
}

/** Put a parsed mesh into an object group so that it fills the object's box (centred at the origin). */
export function placeMesh(obj: THREE.Object3D, placement: MeshPlacement): THREE.Object3D {
  const inner = obj.clone(true);
  inner.scale.setScalar(placement.scale);
  inner.quaternion.copy(upRotation(placement.up));
  const outer = new THREE.Group();
  outer.position.set(...placement.offset);
  outer.add(inner);
  outer.traverse((c) => {
    if ((c as THREE.Mesh).isMesh) {
      c.castShadow = true;
      c.receiveShadow = true;
    }
  });
  return outer;
}

const cache = new Map<string, Promise<THREE.Object3D>>();

/** Fetch and parse an uploaded asset once; callers clone the result. */
export function loadAsset(assetId: string, format: MeshFormat, color: string): Promise<THREE.Object3D> {
  const key = `${assetId}:${format === "obj" || format === "stl" ? color : ""}`;
  let p = cache.get(key);
  if (!p) {
    p = fetch(`/api/assets/${encodeURIComponent(assetId)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`asset ${assetId}: HTTP ${r.status}`);
        return r.arrayBuffer();
      })
      .then((buf) => parseMesh(buf, format, color));
    cache.set(key, p);
    p.catch(() => cache.delete(key));
  }
  return p;
}
