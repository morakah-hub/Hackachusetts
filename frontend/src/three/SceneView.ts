// The 3D workspace: renders the simulation's robot and table geometry with the poses the server streams, the
// objects of the scene, the reach-sized work area, and handles selection, dragging and picking on the table.
//
// Rendering is display only: positions shown between two server frames are smoothed for the eye, while every
// measurement and decision comes from the simulation.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import type { Frame, GeometryManifest, LibraryItem, PlanLeg, SceneObject, Vec3, World } from "../api/types";
import { footprint, placementIssue } from "../lib/geom";
import { libraryDrag } from "../lib/dnd";
import { loadAsset, placeMesh } from "./meshAssets";
import { buildRobotModel, quatFromWXYZ } from "./robotModel";

export interface ViewCallbacks {
  canEdit(): boolean;
  pickMode(): "object" | "point" | null;
  /** The object a point being picked is for (e.g. the one to move), to preview and check its footprint. */
  pickSubject(): SceneObject | null;
  onSelect(id: string | null): void;
  onHover(id: string | null): void;
  onPickObject(id: string): void;
  onPickPoint(p: [number, number]): void;
  onMove(id: string, x: number, y: number): Promise<boolean>;
  onDropTemplate(item: LibraryItem, p: [number, number]): void;
  onIssue(message: string): void;
}

interface ObjView {
  id: string;
  group: THREE.Group;
  box: THREE.Mesh;
  mesh: THREE.Object3D | null;
  outline: THREE.LineSegments;
  label: CSS2DObject;
  key: string;
  data: SceneObject;
  target: { p: THREE.Vector3; q: THREE.Quaternion };
  override: { x: number; y: number } | null;
  pending: boolean;
}

const ACCENT = new THREE.Color("#2ec4b6");
const WARN = new THREE.Color("#f2a541");
const BAD = new THREE.Color("#ef5b5b");
const HOVER = new THREE.Color("#e9ecef");
const UP = new THREE.Vector3(0, 0, 1);
const HOME_VIEW = { pos: new THREE.Vector3(2.05, 1.2, 1.9), target: new THREE.Vector3(0.18, 0, 0.8) };

export class SceneView {
  private renderer: THREE.WebGLRenderer;
  private labels: CSS2DRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private plane = new THREE.Plane(UP.clone(), -0.75);
  private world: World | null = null;
  private gap = 0.015;
  private bodies = new Map<string, { group: THREE.Group; target: { p: THREE.Vector3; q: THREE.Quaternion } }>();
  private objects = new Map<string, ObjView>();
  private zone = new THREE.Group();
  private planGroup = new THREE.Group();
  private markers = new THREE.Group();
  private footprintMesh: THREE.Mesh;
  private footprintLine: THREE.LineLoop;
  private ghost: THREE.Mesh;
  private ghostItem: LibraryItem | null = null;
  private dragTag: CSS2DObject;
  private selected: string | null = null;
  private hovered: string | null = null;
  private drag: { id: string; offset: THREE.Vector2; start: THREE.Vector2; moved: boolean } | null = null;
  private down: { x: number; y: number; onObject: boolean } | null = null;
  private timer = new THREE.Timer();
  private raf = 0;
  private resizeObs: ResizeObserver;
  private disposed = false;
  private live = false;

  constructor(private container: HTMLElement, private cb: ViewCallbacks) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap; // soft edges come from shadow.radius
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.domElement.className = "viewport-canvas";
    this.renderer.domElement.setAttribute("aria-label", "3D view of the robot, the table and the objects");
    this.renderer.domElement.setAttribute("role", "img");
    container.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = "viewport-labels";
    container.appendChild(this.labels.domElement);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.02, 60);
    this.camera.up.set(0, 0, 1);
    this.camera.position.copy(HOME_VIEW.pos);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(HOME_VIEW.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 6.5;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.update();

    this.setupLights();
    this.setupFloor();
    this.scene.add(this.zone, this.planGroup, this.markers);

    const fpGeo = new THREE.PlaneGeometry(1, 1);
    this.footprintMesh = new THREE.Mesh(fpGeo, new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.22, depthWrite: false }));
    this.footprintMesh.visible = false;
    this.footprintMesh.renderOrder = 3;
    this.footprintLine = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.9 }));
    this.footprintLine.visible = false;
    this.footprintLine.renderOrder = 4;
    this.scene.add(this.footprintMesh, this.footprintLine);

    this.ghost = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: ACCENT, transparent: true, opacity: 0.45, depthWrite: false }));
    this.ghost.visible = false;
    this.scene.add(this.ghost);
    this.dragTag = this.makeLabel("", "label drag-tag");
    this.dragTag.visible = false;
    this.scene.add(this.dragTag);

    const el = this.renderer.domElement;
    // Capture phase: decide before OrbitControls whether this press drags an object or orbits the camera.
    el.addEventListener("pointerdown", this.onPointerDown, { capture: true });
    el.addEventListener("pointermove", this.onPointerMove);
    el.addEventListener("pointerup", this.onPointerUp);
    el.addEventListener("pointerleave", this.onPointerLeave);
    container.addEventListener("dragover", this.onDragOver);
    container.addEventListener("dragleave", this.onDragLeave);
    container.addEventListener("drop", this.onDrop);

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.resize();
    this.loop();
  }

  // ------------------------------------------------------------------ setup

  private setupLights() {
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.55;
    pmrem.dispose();
    this.scene.add(new THREE.HemisphereLight("#e6eef5", "#2a2e33", 0.65));
    const key = new THREE.DirectionalLight("#ffffff", 2.0);
    key.position.set(1.4, -1.0, 3.2);
    key.target.position.set(0, 0, 0.75);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const s = key.shadow.camera as THREE.OrthographicCamera;
    s.left = -1.3;
    s.right = 1.3;
    s.top = 1.3;
    s.bottom = -1.3;
    s.near = 0.5;
    s.far = 6;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.015;
    key.shadow.radius = 3;
    this.scene.add(key, key.target);
    const fill = new THREE.DirectionalLight("#cfe3ff", 0.55);
    fill.position.set(-2.0, 1.6, 1.8);
    this.scene.add(fill);
  }

  private setupFloor() {
    const size = 512;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    grad.addColorStop(0, "rgba(88,94,102,0.95)");
    grad.addColorStop(0.55, "rgba(60,65,72,0.75)");
    grad.addColorStop(1, "rgba(40,44,49,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const floor = new THREE.Mesh(new THREE.CircleGeometry(5, 96), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    floor.renderOrder = -2;
    this.scene.add(floor);
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(3.5, 64), new THREE.ShadowMaterial({ opacity: 0.32 }));
    shadow.position.z = 0.001;
    shadow.receiveShadow = true;
    shadow.renderOrder = -1;
    this.scene.add(shadow);
    const grid = new THREE.PolarGridHelper(3.5, 12, 7, 96, "#5b626b", "#4a5058");
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.002;
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.28;
    this.scene.add(grid);
  }

  /** Table height and work area (from the measured reach). */
  setWorld(world: World, gap: number) {
    this.world = world;
    this.gap = gap;
    this.plane.constant = -world.table_height;
    this.zone.clear();
    const z = world.table_height + 0.0012;
    const th = (world.theta_max_deg * Math.PI) / 180;
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(world.r_in, world.r_out, 160, 1, -th, 2 * th),
      new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.11, depthWrite: false }),
    );
    ring.position.z = z;
    ring.renderOrder = 1;
    this.zone.add(ring);
    const edge = (r: number) => {
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 160; i++) {
        const a = -th + (2 * th * i) / 160;
        pts.push(new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), z + 0.0004));
      }
      return pts;
    };
    const outline = [...edge(world.r_out), ...edge(world.r_in).reverse()];
    outline.push(outline[0].clone());
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(outline), new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.75 }));
    line.renderOrder = 2;
    this.zone.add(line);
    if (world.theta_max_deg < 180) {
      const back = new THREE.Mesh(
        new THREE.RingGeometry(0.16, world.table_radius - 0.004, 96, 1, th, 2 * Math.PI - 2 * th),
        new THREE.MeshBasicMaterial({ color: "#1d2024", transparent: true, opacity: 0.32, depthWrite: false }),
      );
      back.position.z = z;
      back.renderOrder = 1;
      this.zone.add(back);
      const tag = this.makeLabel("Kept clear behind the robot", "label label-muted");
      tag.position.set(-(world.r_in + world.table_radius) / 2 - 0.05, 0, z + 0.01);
      this.zone.add(tag);
    }
    const inner = new THREE.Mesh(
      new THREE.RingGeometry(0.16, world.r_in, 96, 1, -th, 2 * th),
      new THREE.MeshBasicMaterial({ color: "#1d2024", transparent: true, opacity: 0.16, depthWrite: false }),
    );
    inner.position.z = z;
    inner.renderOrder = 1;
    this.zone.add(inner);
  }

  setRobot(manifest: GeometryManifest, blob: ArrayBuffer) {
    const model = buildRobotModel(manifest, blob);
    for (const [name, group] of model.bodies) {
      this.bodies.set(name, { group, target: { p: new THREE.Vector3(), q: new THREE.Quaternion() } });
    }
    this.scene.add(model.root);
  }

  // ------------------------------------------------------------------ data

  setFrame(frame: Frame) {
    this.live = frame.live;
    for (const [name, pose] of Object.entries(frame.bodies)) {
      const b = this.bodies.get(name);
      if (!b) continue;
      b.target.p.set(pose[0], pose[1], pose[2]);
      quatFromWXYZ([pose[3], pose[4], pose[5], pose[6]], b.target.q);
      if (b.group.userData.init !== true) {
        b.group.position.copy(b.target.p);
        b.group.quaternion.copy(b.target.q);
        b.group.userData.init = true;
      }
    }
    for (const [id, pose] of Object.entries(frame.objects)) {
      const v = this.objects.get(id);
      if (!v) continue;
      v.target.p.set(pose[0], pose[1], pose[2]);
      quatFromWXYZ([pose[3], pose[4], pose[5], pose[6]], v.target.q);
    }
  }

  setObjects(list: SceneObject[]) {
    const seen = new Set<string>();
    for (const o of list) {
      seen.add(o.id);
      const key = `${o.size.join(",")}|${o.color}|${o.asset_id}|${JSON.stringify(o.mesh)}`;
      let v = this.objects.get(o.id);
      if (v && v.key !== key) {
        this.removeObject(o.id);
        v = undefined;
      }
      if (!v) v = this.createObject(o, key);
      v.data = o;
      v.label.element.textContent = o.name;
      if (!this.live && !v.override) {
        v.target.p.set(o.pos[0], o.pos[1], o.pos[2]);
        quatFromWXYZ(o.quat, v.target.q);
      }
      v.pending = false;
    }
    for (const id of [...this.objects.keys()]) if (!seen.has(id)) this.removeObject(id);
    this.refreshHighlights();
  }

  private createObject(o: SceneObject, key: string): ObjView {
    const group = new THREE.Group();
    group.name = `object-${o.id}`;
    const [w, d, h] = o.size;
    const r = Math.min(0.004, Math.min(w, d, h) * 0.12);
    const box = new THREE.Mesh(
      new RoundedBoxGeometry(w, d, h, 2, r),
      new THREE.MeshStandardMaterial({ color: o.color, roughness: 0.5, metalness: o.name.toLowerCase().includes("steel") || o.name.toLowerCase().includes("alu") ? 0.55 : 0.04 }),
    );
    box.castShadow = true;
    box.receiveShadow = true;
    box.userData.objectId = o.id;
    group.add(box);
    // The outline is the collision box the simulation uses (for uploaded models it encloses the mesh).
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(w + 0.004, d + 0.004, h + 0.004)),
      new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.95 }),
    );
    outline.visible = false;
    outline.renderOrder = 10;
    group.add(outline);
    const label = this.makeLabel(o.name, "label");
    label.position.set(0, 0, h / 2 + 0.03);
    label.visible = false;
    group.add(label);
    group.position.set(o.pos[0], o.pos[1], o.pos[2]);
    quatFromWXYZ(o.quat, group.quaternion);
    this.scene.add(group);
    const v: ObjView = {
      id: o.id, group, box, mesh: null, outline, label, key, data: o, override: null, pending: false,
      target: { p: group.position.clone(), q: group.quaternion.clone() },
    };
    this.objects.set(o.id, v);
    if (o.asset_id && o.mesh) {
      const placement = o.mesh;
      (box.material as THREE.MeshStandardMaterial).transparent = true;
      (box.material as THREE.MeshStandardMaterial).opacity = 0.35;
      loadAsset(o.asset_id, placement.format, o.color)
        .then((obj) => {
          if (this.objects.get(o.id) !== v) return;
          const mesh = placeMesh(obj, placement);
          mesh.traverse((c) => (c.userData.objectId = o.id));
          group.add(mesh);
          v.mesh = mesh;
          box.visible = false;
          box.userData.proxyOnly = true;
        })
        .catch(() => {
          (box.material as THREE.MeshStandardMaterial).opacity = 1;
          (box.material as THREE.MeshStandardMaterial).transparent = false;
          this.cb.onIssue(`Could not load the 3D model of ${o.name}; showing its box instead.`);
        });
    }
    return v;
  }

  private removeObject(id: string) {
    const v = this.objects.get(id);
    if (!v) return;
    this.scene.remove(v.group);
    v.group.traverse((c) => {
      const m = c as THREE.Mesh;
      if (m.geometry && !(m.userData.shared)) m.geometry.dispose?.();
    });
    v.label.element.remove();
    this.objects.delete(id);
  }

  setSelection(id: string | null) {
    this.selected = id;
    this.refreshHighlights();
  }

  setHover(id: string | null) {
    this.hovered = id;
    this.refreshHighlights();
  }

  private refreshHighlights() {
    for (const v of this.objects.values()) {
      const sel = v.id === this.selected;
      const hov = v.id === this.hovered;
      const bad = v.data.state === "tipped" || v.data.state === "fallen";
      const cant = !v.data.can_lift;
      const boxMat = v.box.material as THREE.MeshStandardMaterial;
      boxMat.emissive.copy(sel ? ACCENT : HOVER);
      boxMat.emissiveIntensity = sel ? 0.18 : hov ? 0.06 : 0;
      v.outline.visible = sel || hov || bad;
      const mat = v.outline.material as THREE.LineBasicMaterial;
      mat.color.copy(bad ? BAD : sel ? (cant ? WARN : ACCENT) : HOVER);
      v.label.visible = sel || hov;
      v.label.element.className = `label${sel ? " label-selected" : ""}${cant ? " label-warn" : ""}`;
    }
  }

  /** Planned legs: arcs from where each object is to where it goes, numbered, with a ghost at the target. */
  setPlan(legs: PlanLeg[] | null, objects: SceneObject[], activeIndex = -1) {
    this.clearGroup(this.planGroup);
    if (!legs) return;
    legs.forEach((leg, i) => {
      const o = objects.find((x) => x.id === leg.object_id);
      const from = new THREE.Vector3(...leg.from);
      const to = new THREE.Vector3(...leg.to);
      const mid = from.clone().add(to).multiplyScalar(0.5);
      mid.z = Math.max(from.z, to.z) + 0.12 + 0.12 * Math.min(1, from.distanceTo(to));
      const curve = new THREE.QuadraticBezierCurve3(from, mid, to);
      const done = activeIndex >= 0 && i < activeIndex;
      const color = !leg.ok ? BAD : leg.purpose === "move" ? ACCENT : WARN;
      const tube = new THREE.Mesh(
        new THREE.TubeGeometry(curve, 48, 0.0035, 6, false),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: done ? 0.18 : 0.85, depthWrite: false }),
      );
      tube.renderOrder = 5;
      this.planGroup.add(tube);
      const head = new THREE.Mesh(new THREE.ConeGeometry(0.012, 0.03, 16), tube.material);
      const tangent = curve.getTangent(1);
      head.position.copy(to).addScaledVector(tangent, -0.012);
      head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
      this.planGroup.add(head);
      if (o) {
        const g = new THREE.Mesh(
          new THREE.BoxGeometry(o.size[0], o.size[1], o.size[2]),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: done ? 0.05 : 0.16, depthWrite: false }),
        );
        g.position.copy(to);
        g.rotation.z = o.yaw;
        this.planGroup.add(g);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: done ? 0.2 : 0.8 }));
        edges.position.copy(to);
        edges.rotation.z = o.yaw;
        this.planGroup.add(edges);
      }
      const tag = this.makeLabel(String(i + 1), `step-tag${i === activeIndex ? " step-tag-active" : ""}${done ? " step-tag-done" : ""}`);
      tag.position.copy(curve.getPoint(0.5)).add(new THREE.Vector3(0, 0, 0.02));
      this.planGroup.add(tag);
    });
  }

  /** Destination pins for point parameters of the task being set up. */
  setMarkers(points: { p: [number, number]; label: string }[]) {
    this.clearGroup(this.markers);
    if (!this.world) return;
    for (const m of points) {
      const z = this.world.table_height;
      const pin = new THREE.Group();
      const disc = new THREE.Mesh(new THREE.RingGeometry(0.014, 0.024, 40), new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.95, depthWrite: false }));
      disc.position.z = 0.002;
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.0025, 0.0025, 0.09, 8).rotateX(Math.PI / 2), new THREE.MeshBasicMaterial({ color: ACCENT }));
      stem.position.z = 0.045;
      const ball = new THREE.Mesh(new THREE.SphereGeometry(0.011, 20, 12), new THREE.MeshStandardMaterial({ color: ACCENT, emissive: ACCENT, emissiveIntensity: 0.3 }));
      ball.position.z = 0.095;
      const tag = this.makeLabel(m.label, "label label-selected");
      tag.position.z = 0.125;
      pin.add(disc, stem, ball, tag);
      pin.position.set(m.p[0], m.p[1], z);
      this.markers.add(pin);
    }
  }

  private clearGroup(g: THREE.Group) {
    g.traverse((c) => {
      if (c instanceof CSS2DObject) c.element.remove();
    });
    g.clear();
  }

  private makeLabel(text: string, className: string): CSS2DObject {
    const div = document.createElement("div");
    div.className = className;
    div.textContent = text;
    return new CSS2DObject(div);
  }

  // ------------------------------------------------------------------ camera

  resetView() {
    this.animateCamera(HOME_VIEW.pos, HOME_VIEW.target);
  }

  topView() {
    this.animateCamera(new THREE.Vector3(0.12, -0.0001, 2.9), new THREE.Vector3(0.12, 0, 0.75));
  }

  private camAnim: { from: THREE.Vector3; to: THREE.Vector3; tFrom: THREE.Vector3; tTo: THREE.Vector3; t: number } | null = null;

  private animateCamera(pos: THREE.Vector3, target: THREE.Vector3) {
    this.camAnim = { from: this.camera.position.clone(), to: pos.clone(), tFrom: this.controls.target.clone(), tTo: target.clone(), t: 0 };
  }

  // ------------------------------------------------------------------ interaction

  private setPointer(ev: PointerEvent | DragEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
  }

  private hitObject(): string | null {
    const targets: THREE.Object3D[] = [];
    for (const v of this.objects.values()) targets.push(v.mesh ?? v.box);
    const hits = this.raycaster.intersectObjects(targets, true);
    for (const h of hits) {
      let o: THREE.Object3D | null = h.object;
      while (o && !o.userData.objectId) o = o.parent;
      if (o?.userData.objectId) return o.userData.objectId as string;
    }
    return null;
  }

  private hitTable(): THREE.Vector3 | null {
    const p = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(this.plane, p)) return null;
    if (this.world && Math.hypot(p.x, p.y) > this.world.table_radius) return null;
    return p;
  }

  private onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    this.setPointer(ev);
    const id = this.hitObject();
    this.down = { x: ev.clientX, y: ev.clientY, onObject: !!id };
    const mode = this.cb.pickMode();
    if (id && mode === "object") {
      this.controls.enabled = false;
      this.cb.onPickObject(id);
      return;
    }
    if (id && mode !== "point") {
      this.cb.onSelect(id);
      if (this.cb.canEdit()) {
        const v = this.objects.get(id)!;
        const hit = this.hitTable();
        if (hit && v.data.state !== "fallen") {
          this.controls.enabled = false;
          this.drag = { id, offset: new THREE.Vector2(v.group.position.x - hit.x, v.group.position.y - hit.y), start: new THREE.Vector2(ev.clientX, ev.clientY), moved: false };
          this.renderer.domElement.setPointerCapture(ev.pointerId);
        }
      }
    }
  };

  private onPointerMove = (ev: PointerEvent) => {
    this.setPointer(ev);
    if (this.drag) {
      const d = this.drag;
      if (!d.moved && Math.hypot(ev.clientX - d.start.x, ev.clientY - d.start.y) < 4) return;
      d.moved = true;
      const hit = this.hitTable();
      const v = this.objects.get(d.id);
      if (!hit || !v || !this.world) return;
      const x = hit.x + d.offset.x;
      const y = hit.y + d.offset.y;
      v.override = { x, y };
      const others = [...this.objects.values()].map((o) => o.data);
      const issue = placementIssue(this.world, this.gap, x, y, v.data.yaw, v.data.size, others, v.id);
      this.showFootprint(x, y, v.data.yaw, v.data.size, issue === null);
      this.showDragTag(x, y, v.data.size[2], issue);
      const mat = v.box.material as THREE.MeshStandardMaterial;
      mat.emissive.copy(issue ? BAD : ACCENT);
      mat.emissiveIntensity = issue ? 0.45 : 0.18;
      this.renderer.domElement.style.cursor = issue ? "not-allowed" : "grabbing";
      return;
    }
    const id = this.hitObject();
    if (id !== this.hovered) this.cb.onHover(id);
    const mode = this.cb.pickMode();
    this.renderer.domElement.style.cursor =
      mode === "point" ? (this.hitTable() ? "crosshair" : "default") : id ? (mode === "object" ? "pointer" : this.cb.canEdit() ? "grab" : "pointer") : "default";
    if (mode === "point" && this.world) {
      const hit = this.hitTable();
      if (hit) {
        const issue = this.pointIssue(hit.x, hit.y);
        const subject = this.cb.pickSubject();
        const size: Vec3 = subject ? subject.size : [0.04, 0.04, 0.001];
        this.showFootprint(hit.x, hit.y, subject ? subject.yaw : 0, size, subject ? issue === null : null);
        this.showDragTag(hit.x, hit.y, subject ? subject.size[2] : 0.02, issue, "Click to choose this spot");
      } else {
        this.hideFootprint();
        this.dragTag.visible = false;
      }
    }
  };

  /** Would the object of the task fit at (x, y)? (Instant check; the server's plan check decides.) */
  private pointIssue(x: number, y: number): string | null {
    const subject = this.cb.pickSubject();
    if (!subject || !this.world) return null;
    const others = [...this.objects.values()].map((o) => o.data);
    return placementIssue(this.world, this.gap, x, y, subject.yaw, subject.size, others, subject.id);
  }

  private onPointerUp = (ev: PointerEvent) => {
    this.controls.enabled = true;
    const clicked = this.down && Math.hypot(ev.clientX - this.down.x, ev.clientY - this.down.y) < 5;
    if (this.drag) {
      const d = this.drag;
      this.drag = null;
      this.renderer.domElement.releasePointerCapture(ev.pointerId);
      this.hideFootprint();
      this.dragTag.visible = false;
      this.refreshHighlights();
      this.renderer.domElement.style.cursor = "grab";
      const v = this.objects.get(d.id);
      if (d.moved && v?.override) {
        const { x, y } = v.override;
        v.pending = true;
        this.cb.onMove(d.id, x, y).finally(() => {
          v.override = null;
        });
      }
    } else if (clicked && !this.down?.onObject) {
      this.setPointer(ev);
      const mode = this.cb.pickMode();
      const hit = this.hitTable();
      if (mode === "point" && hit) {
        const issue = this.pointIssue(hit.x, hit.y);
        if (issue) this.cb.onIssue(`${issue}. Choose another spot.`);
        else {
          this.cb.onPickPoint([hit.x, hit.y]);
          this.hideFootprint();
          this.dragTag.visible = false;
        }
      } else if (mode === null) this.cb.onSelect(null);
    }
    this.down = null;
  };

  private onPointerLeave = () => {
    if (!this.drag) {
      this.cb.onHover(null);
      if (this.cb.pickMode() === "point") {
        this.hideFootprint();
        this.dragTag.visible = false;
      }
    }
  };

  private onDragOver = (ev: DragEvent) => {
    const item = libraryDrag.current;
    if (!item || !this.world) return;
    ev.preventDefault();
    this.setPointer(ev);
    const hit = this.hitTable();
    if (!hit) {
      this.hideGhost();
      if (ev.dataTransfer) ev.dataTransfer.dropEffect = "none";
      return;
    }
    const others = [...this.objects.values()].map((o) => o.data);
    const issue = this.cb.canEdit() ? placementIssue(this.world, this.gap, hit.x, hit.y, 0, item.size, others) : "A task is running";
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = issue ? "none" : "copy";
    this.showGhost(item, hit.x, hit.y, issue === null);
    this.showDragTag(hit.x, hit.y, item.size[2], issue);
  };

  private onDragLeave = (ev: DragEvent) => {
    if (ev.target === this.renderer.domElement || ev.target === this.labels.domElement) this.hideGhost();
  };

  private onDrop = (ev: DragEvent) => {
    const item = libraryDrag.current;
    this.hideGhost();
    if (!item || !this.world) return;
    ev.preventDefault();
    this.setPointer(ev);
    const hit = this.hitTable();
    if (!hit) {
      this.cb.onIssue("Drop objects on the table.");
      return;
    }
    const others = [...this.objects.values()].map((o) => o.data);
    const issue = placementIssue(this.world, this.gap, hit.x, hit.y, 0, item.size, others);
    if (issue) {
      this.cb.onIssue(`${item.name}: ${issue.toLowerCase()}.`);
      return;
    }
    this.cb.onDropTemplate(item, [hit.x, hit.y]);
  };

  private showGhost(item: LibraryItem, x: number, y: number, ok: boolean) {
    if (!this.world) return;
    if (this.ghostItem !== item) {
      this.ghost.geometry.dispose();
      this.ghost.geometry = new THREE.BoxGeometry(item.size[0], item.size[1], item.size[2]);
      this.ghostItem = item;
    }
    (this.ghost.material as THREE.MeshStandardMaterial).color.copy(ok ? ACCENT : BAD);
    this.ghost.position.set(x, y, this.world.table_height + item.size[2] / 2);
    this.ghost.visible = true;
    this.showFootprint(x, y, 0, item.size, ok);
  }

  private hideGhost() {
    this.ghost.visible = false;
    this.ghostItem = null;
    this.hideFootprint();
    this.dragTag.visible = false;
  }

  /** Why the object cannot go here (or what a click does), shown next to it while dragging or picking. */
  private showDragTag(x: number, y: number, h: number, issue: string | null, okText = "Release to place") {
    if (!this.world) return;
    this.dragTag.element.textContent = issue ?? okText;
    this.dragTag.element.className = `label drag-tag${issue ? " drag-tag-bad" : ""}`;
    this.dragTag.position.set(x, y, this.world.table_height + h + 0.06);
    this.dragTag.visible = true;
  }

  private showFootprint(x: number, y: number, yaw: number, size: Vec3, ok: boolean | null) {
    if (!this.world) return;
    const z = this.world.table_height + 0.002;
    const color = ok === null ? ACCENT : ok ? ACCENT : BAD;
    // Drawn grown by the clearance objects need from each other, so it shows around the object.
    const grow = ok === null ? 0 : this.gap / 2 + 0.006;
    size = [size[0] + 2 * grow, size[1] + 2 * grow, size[2]];
    this.footprintMesh.scale.set(size[0], size[1], 1);
    this.footprintMesh.position.set(x, y, z);
    this.footprintMesh.rotation.set(0, 0, yaw);
    (this.footprintMesh.material as THREE.MeshBasicMaterial).color.copy(color);
    (this.footprintMesh.material as THREE.MeshBasicMaterial).opacity = ok === false ? 0.3 : 0.2;
    this.footprintMesh.visible = true;
    const pts = footprint(x, y, yaw, size).map(([px, py]) => new THREE.Vector3(px, py, z + 0.0005));
    this.footprintLine.geometry.dispose();
    this.footprintLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    (this.footprintLine.material as THREE.LineBasicMaterial).color.copy(color);
    this.footprintLine.visible = true;
  }

  private hideFootprint() {
    this.footprintMesh.visible = false;
    this.footprintLine.visible = false;
  }

  // ------------------------------------------------------------------ loop

  private resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private loop = (timestamp?: number) => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    this.timer.update(timestamp);
    const dt = Math.min(this.timer.getDelta(), 0.1);
    const a = 1 - Math.exp(-dt / 0.035);
    for (const b of this.bodies.values()) {
      b.group.position.lerp(b.target.p, a);
      b.group.quaternion.slerp(b.target.q, a);
    }
    for (const v of this.objects.values()) {
      if (v.override) {
        v.group.position.set(v.override.x, v.override.y, (this.world?.table_height ?? 0.75) + v.data.size[2] / 2);
        continue;
      }
      if (v.pending) continue;
      if (v.group.position.distanceTo(v.target.p) > 0.4) {
        v.group.position.copy(v.target.p);
        v.group.quaternion.copy(v.target.q);
      } else {
        v.group.position.lerp(v.target.p, a);
        v.group.quaternion.slerp(v.target.q, a);
      }
    }
    if (this.camAnim) {
      const c = this.camAnim;
      c.t = Math.min(1, c.t + dt / 0.6);
      const e = c.t < 0.5 ? 2 * c.t * c.t : 1 - Math.pow(-2 * c.t + 2, 2) / 2;
      this.camera.position.lerpVectors(c.from, c.to, e);
      this.controls.target.lerpVectors(c.tFrom, c.tTo, e);
      if (c.t >= 1) this.camAnim = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  };

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    const el = this.renderer.domElement;
    el.removeEventListener("pointerdown", this.onPointerDown, { capture: true });
    el.removeEventListener("pointermove", this.onPointerMove);
    el.removeEventListener("pointerup", this.onPointerUp);
    el.removeEventListener("pointerleave", this.onPointerLeave);
    this.container.removeEventListener("dragover", this.onDragOver);
    this.container.removeEventListener("dragleave", this.onDragLeave);
    this.container.removeEventListener("drop", this.onDrop);
    this.controls.dispose();
    this.renderer.dispose();
    el.remove();
    this.labels.domElement.remove();
  }
}
