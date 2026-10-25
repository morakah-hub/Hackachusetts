import { useEffect, useRef } from "react";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { MeshPlacement, Vec3 } from "../api/types";
import { placeMesh } from "../three/meshAssets";

/** Small turntable preview of an uploaded mesh inside its box proxy. */
export function MeshPreview({ object, placement, size }: { object: THREE.Object3D | null; placement: MeshPlacement | null; size: Vec3 }) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<{ renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; holder: THREE.Group; raf: number } | null>(null);

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.setSize(el.clientWidth, el.clientHeight, false);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    scene.add(new THREE.HemisphereLight("#ffffff", "#334", 1.2));
    const dir = new THREE.DirectionalLight("#ffffff", 1.6);
    dir.position.set(2, -2, 3);
    scene.add(dir);
    const camera = new THREE.PerspectiveCamera(35, el.clientWidth / el.clientHeight, 0.001, 100);
    camera.up.set(0, 0, 1);
    const holder = new THREE.Group();
    scene.add(holder);
    const s = { renderer, scene, camera, holder, raf: 0 };
    state.current = s;
    const loop = () => {
      s.raf = requestAnimationFrame(loop);
      holder.rotation.z += 0.01;
      renderer.render(scene, camera);
    };
    loop();
    return () => {
      cancelAnimationFrame(s.raf);
      renderer.dispose();
      renderer.domElement.remove();
      state.current = null;
    };
  }, []);

  useEffect(() => {
    const s = state.current;
    if (!s) return;
    s.holder.clear();
    const [w, d, h] = size;
    if (object && placement) s.holder.add(placeMesh(object, placement));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, d, h)), new THREE.LineBasicMaterial({ color: "#2ec4b6" }));
    s.holder.add(edges);
    const base = new THREE.Mesh(new THREE.CircleGeometry(Math.max(w, d) * 0.9, 48), new THREE.MeshBasicMaterial({ color: "#2b3036" }));
    base.position.z = -h / 2 - 0.0005;
    s.holder.add(base);
    const r = Math.max(w, d, h) || 0.1;
    s.camera.position.set(r * 1.9, -r * 1.9, r * 1.35);
    s.camera.near = r / 100;
    s.camera.far = r * 50;
    s.camera.updateProjectionMatrix();
    s.camera.lookAt(0, 0, 0);
  }, [object, placement, size]);

  return <div className="mesh-preview" ref={host} aria-label="Preview of the uploaded model" role="img" />;
}
