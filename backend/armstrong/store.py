"""Scene objects, the object library and uploaded assets: placement rules and JSON persistence
(adapted from Workcell Studio's studio/scene.py).

An object is a rigid box proxy (size = w, d, h along its own axes) with a mass, a colour and optionally an
uploaded mesh that the browser draws inside the box. The scene stores where each object is (centre and
w-x-y-z quaternion, world frame). Edits keep objects upright on the table; after a run the poses measured in
the simulation replace them (an object can then rest on another one, or be tipped over).
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .rules import Problem, RobotRules, World, quat_from_yaw, tilt_of_quat, yaw_of_quat

ASSET_TYPES = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".obj": "text/plain", ".stl": "model/stl"}
MAX_ASSET_BYTES = 30 * 1024 * 1024
MAX_OBJECTS = 16
STACK_TOL = 0.008     # m: an object rests on a surface whose top is this close to its bottom
NAME_MAX = 40
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class StoreError(Exception):
    """A rejected edit, with the problems explaining why."""

    def __init__(self, message: str, problems: list[Problem] | None = None, status: int = 400):
        super().__init__(message)
        self.message, self.problems, self.status = message, problems or [], status


@dataclass(kw_only=True)
class ObjectModel:
    name: str
    size: list[float]                 # w, d, h (m) along the object's x, y, z
    mass: float                       # kg
    color: str = "#b8bec6"
    asset_id: str | None = None       # uploaded mesh drawn inside the box (browser only)
    mesh: dict | None = None          # how the mesh maps onto the box; set by the browser


@dataclass(kw_only=True)
class LibraryItem(ObjectModel):
    id: str
    builtin: bool = False
    created_at: float = 0.0


@dataclass(kw_only=True)
class SceneObject(ObjectModel):
    id: str
    pos: list[float]                  # centre, world frame
    quat: list[float]                 # w, x, y, z
    template_id: str | None = None

    @property
    def half(self) -> np.ndarray:
        return 0.5 * np.asarray(self.size, float)

    @property
    def yaw(self) -> float:
        return yaw_of_quat(self.quat)

    @property
    def tilt_deg(self) -> float:
        return tilt_of_quat(self.quat)

    @property
    def bottom(self) -> float:
        return float(self.pos[2]) - self.half_height_world

    @property
    def top(self) -> float:
        return float(self.pos[2]) + self.half_height_world

    @property
    def half_height_world(self) -> float:
        R = quat_mat(self.quat)
        return float(np.abs(R[2]) @ self.half)


# Assembly parts first: they are what the "Assemble" task builds with.
BUILTIN_LIBRARY = [
    {"id": "lib_pillar", "name": "Pillar", "size": [0.04, 0.04, 0.07], "mass": 0.12, "color": "#3d8bfd"},
    {"id": "lib_beam", "name": "Beam", "size": [0.16, 0.04, 0.03], "mass": 0.15, "color": "#f2b134"},
    {"id": "lib_block_l", "name": "Base block", "size": [0.055, 0.055, 0.045], "mass": 0.30, "color": "#e8692c"},
    {"id": "lib_block_m", "name": "Middle block", "size": [0.048, 0.048, 0.045], "mass": 0.20, "color": "#3fb37f"},
    {"id": "lib_block_s", "name": "Top block", "size": [0.04, 0.04, 0.04], "mass": 0.10, "color": "#b464e0"},
    {"id": "lib_alu_block", "name": "Aluminium block", "size": [0.05, 0.05, 0.06], "mass": 0.40, "color": "#a9b2bb"},
    {"id": "lib_wood_cube", "name": "Wooden cube", "size": [0.045, 0.045, 0.045], "mass": 0.06, "color": "#c99a6a"},
    {"id": "lib_juice", "name": "Juice carton", "size": [0.06, 0.04, 0.10], "mass": 0.25, "color": "#f0b23e"},
    {"id": "lib_tea_box", "name": "Tea box", "size": [0.055, 0.055, 0.12], "mass": 0.12, "color": "#5d9f7b"},
    {"id": "lib_steel", "name": "Steel weight", "size": [0.045, 0.045, 0.08], "mass": 1.20, "color": "#5d6773"},
    {"id": "lib_brick", "name": "Brick", "size": [0.11, 0.05, 0.06], "mass": 0.60, "color": "#b4553d"},
    {"id": "lib_toolbox", "name": "Toolbox", "size": [0.28, 0.12, 0.12], "mass": 4.50, "color": "#c43d32"},
]
# (template, x, y, yaw, name)
LAYOUTS = {
    "default": [("lib_pillar", 0.40, 0.29, 0.0, "Pillar A"), ("lib_pillar", 0.53, 0.33, 0.0, "Pillar B"),
                ("lib_beam", 0.63, 0.19, 0.0, None),
                ("lib_block_l", 0.40, -0.29, 0.0, None), ("lib_block_m", 0.53, -0.33, 0.0, None),
                ("lib_block_s", 0.63, -0.19, 0.0, None)],
    "objects": [("lib_alu_block", 0.52, 0.20, 0.0, None), ("lib_juice", 0.50, -0.18, 0.0, None),
                ("lib_tea_box", 0.38, 0.33, 0.5, None), ("lib_wood_cube", 0.62, -0.02, 0.3, None)],
}


def quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def footprint(x: float, y: float, yaw: float, size, grow: float = 0.0) -> np.ndarray:
    """Corners (4, 2) of the object's footprint, grown by `grow` on every side."""
    hw, hd = 0.5 * float(size[0]) + grow, 0.5 * float(size[1]) + grow
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[hw, hd], [-hw, hd], [-hw, -hd], [hw, -hd]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons (corners in order)."""
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def object_footprint(o: SceneObject, grow: float = 0.0) -> np.ndarray:
    return footprint(o.pos[0], o.pos[1], o.yaw, o.size, grow)


def supports_of(o: SceneObject, others, table_height: float = 0.0) -> list[str]:
    """Objects that `o` rests on (their top within STACK_TOL of its bottom, footprints overlapping)."""
    out = []
    for p in others:
        if p.id == o.id or abs(p.top - o.bottom) > STACK_TOL:
            continue
        if polygons_overlap(object_footprint(o), object_footprint(p)):
            out.append(p.id)
    return out


def on_table(o: SceneObject, table_height: float) -> bool:
    return abs(o.bottom - table_height) <= STACK_TOL


class StudioStore:
    """Thread-safe scene, library and asset storage under `data_dir`."""

    def __init__(self, data_dir: Path | None, rules: RobotRules, world: World, persist: bool = True):
        self.persist = persist and data_dir is not None
        self.dir = Path(data_dir) if data_dir is not None else None
        if self.persist:
            self.assets_dir = self.dir / "assets"
            self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.rules, self.world, self.gap = rules, world, float(rules.lim["gap_m"])
        self.lock = threading.RLock()
        self.revision = 0
        self.objects: dict[str, SceneObject] = {}
        self.library: dict[str, LibraryItem] = {}
        self.assets: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        for item in BUILTIN_LIBRARY:
            self.library[item["id"]] = LibraryItem(**item, builtin=True)
        for d in self._read("library.json", []):
            try:
                item = LibraryItem(**{k: d[k] for k in d if k in LibraryItem.__dataclass_fields__})
            except TypeError:
                continue
            if not item.builtin:
                self.library[item.id] = item
        if self.persist:
            self.assets = {a["id"]: a for a in self._read("assets.json", [])
                           if (self.assets_dir / a["file"]).is_file()}
        scene = self._read("scene.json", None)
        if scene is None:
            self._seed("default")
        else:
            for d in scene.get("objects", []):
                try:
                    o = SceneObject(**{k: d[k] for k in d if k in SceneObject.__dataclass_fields__})
                except TypeError:
                    continue
                self.objects[o.id] = o
            self.revision = int(scene.get("revision", 0)) + 1

    def _read(self, name: str, default):
        if not self.persist:
            return default
        try:
            return json.loads((self.dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _write(self, name: str, data) -> None:
        if not self.persist:
            return
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.dir / name)

    def _save_scene(self) -> None:
        self._write("scene.json", {"revision": self.revision, "objects": [asdict(o) for o in self.objects.values()]})

    def _save_library(self) -> None:
        self._write("library.json", [asdict(i) for i in self.library.values() if not i.builtin])

    def _seed(self, layout: str) -> None:
        for tid, x, y, yaw, name in LAYOUTS.get(layout, LAYOUTS["default"]):
            t = self.library[tid]
            o = self._make_object(t, x, y, yaw, t.id)
            if name:
                o.name = name
            if not self.placement_problems(o, list(self.objects.values())):
                self.objects[o.id] = o
        self.revision += 1
        self._save_scene()

    # ------------------------------------------------------------------ library

    def library_list(self) -> list[dict]:
        with self.lock:
            return [self._library_dict(i) for i in self.library.values()]

    def _library_dict(self, i: LibraryItem) -> dict:
        d = asdict(i)
        d["problems"] = [p.to_dict() for p in self.rules.lift_problems(i.name, i.size, i.mass)]
        d["can_lift"] = not d["problems"]
        return d

    def add_library_item(self, data: dict) -> dict:
        model = self._clean_model(data)
        with self.lock:
            if model.asset_id is not None and model.asset_id not in self.assets:
                raise StoreError("Unknown uploaded file.")
            item = LibraryItem(**asdict(model), id="lib_" + uuid.uuid4().hex[:8], created_at=time.time())
            self.library[item.id] = item
            self._save_library()
            return self._library_dict(item)

    def delete_library_item(self, item_id: str) -> None:
        with self.lock:
            item = self.library.get(item_id)
            if item is None:
                raise StoreError("No such library item.", status=404)
            if item.builtin:
                raise StoreError("Built-in samples cannot be deleted.")
            del self.library[item_id]
            self._save_library()

    def _clean_model(self, data: dict) -> ObjectModel:
        name = str(data.get("name") or "").strip()[:NAME_MAX] or "Object"
        try:
            size = [float(v) for v in data.get("size")]
            mass = float(data.get("mass"))
        except (TypeError, ValueError):
            raise StoreError("An object needs a size [w, d, h] in metres and a mass in kg.") from None
        if len(size) != 3:
            raise StoreError("An object needs a size [w, d, h] in metres.")
        color = str(data.get("color") or "#b8bec6")
        if not COLOR_RE.match(color):
            raise StoreError("Colours are written as #rrggbb.")
        problems = self.rules.shape_problems(name, size, mass)
        if problems:
            raise StoreError(problems[0].message, problems)
        mesh = data.get("mesh")
        if mesh is not None and (not isinstance(mesh, dict) or len(json.dumps(mesh)) > 2000):
            raise StoreError("Invalid mesh placement data.")
        asset = data.get("asset_id")
        return ObjectModel(name=name, size=size, mass=mass, color=color, asset_id=asset if asset else None, mesh=mesh)

    # ------------------------------------------------------------------ assets

    def save_asset(self, filename: str, data: bytes) -> dict:
        if not self.persist:
            raise StoreError("Uploads are disabled in this mode.")
        ext = Path(filename).suffix.lower()
        if ext not in ASSET_TYPES:
            raise StoreError(f"Unsupported file type '{ext}'. Use .glb, .gltf, .obj or .stl.")
        if not data:
            raise StoreError("The file is empty.")
        if len(data) > MAX_ASSET_BYTES:
            raise StoreError(f"The file is larger than {MAX_ASSET_BYTES // (1024 * 1024)} MB.", status=413)
        asset_id = "ast_" + uuid.uuid4().hex[:12]
        meta = {"id": asset_id, "file": asset_id + ext, "filename": Path(filename).name[:120], "ext": ext,
                "bytes": len(data), "created_at": time.time()}
        with self.lock:
            (self.assets_dir / meta["file"]).write_bytes(data)
            self.assets[asset_id] = meta
            self._write("assets.json", list(self.assets.values()))
        return meta

    def asset_file(self, asset_id: str) -> tuple[Path, str]:
        with self.lock:
            meta = self.assets.get(asset_id)
        if meta is None:
            raise StoreError("No such file.", status=404)
        return self.assets_dir / meta["file"], ASSET_TYPES[meta["ext"]]

    # ------------------------------------------------------------------ scene

    def scene_dict(self) -> dict:
        with self.lock:
            objs = list(self.objects.values())
            return {"revision": self.revision, "objects": [self._object_dict(o, objs) for o in objs]}

    def _object_dict(self, o: SceneObject, objs) -> dict:
        d = asdict(o)
        d["yaw"] = o.yaw
        d["tilt_deg"] = o.tilt_deg
        state, notes = self.object_state(o, objs)
        d["state"], d["notes"] = state, notes
        d["supports"] = supports_of(o, objs)
        d["supporting"] = [p.id for p in objs if o.id in supports_of(p, objs)]
        problems = self.rules.lift_problems(o.name, o.size, o.mass, o.id)
        d["problems"] = [p.to_dict() for p in problems]
        d["can_lift"] = not problems and state in ("on_table", "stacked")
        return d

    def object_state(self, o: SceneObject, objs) -> tuple[str, list[str]]:
        w = self.world
        if o.pos[2] < w.table_height - 0.05 or np.hypot(o.pos[0], o.pos[1]) > w.table_radius + 0.02:
            return "fallen", [f"{o.name} fell off the table."]
        if o.tilt_deg > self.rules.lim["max_tilt_deg"]:
            return "tipped", [f"{o.name} is tipped over ({o.tilt_deg:.0f} deg); drag it to stand it up again."]
        if on_table(o, w.table_height):
            return "on_table", []
        if supports_of(o, objs):
            return "stacked", []
        return "unsupported", [f"{o.name} is not resting on the table or another object."]

    def snapshot(self) -> dict:
        with self.lock:
            return {"revision": self.revision, "objects": copy.deepcopy(list(self.objects.values()))}

    def _make_object(self, model: ObjectModel, x: float, y: float, yaw: float, template_id: str | None) -> SceneObject:
        h = float(model.size[2])
        return SceneObject(name=model.name, size=list(model.size), mass=model.mass, color=model.color,
                           asset_id=model.asset_id, mesh=copy.deepcopy(model.mesh), id="obj_" + uuid.uuid4().hex[:6],
                           pos=[float(x), float(y), self.world.table_height + 0.5 * h], quat=quat_from_yaw(yaw),
                           template_id=template_id)

    def overlaps(self, o: SceneObject, p: SceneObject) -> bool:
        """True if two objects at overlapping heights are closer than the placement gap."""
        if p.id == o.id or p.bottom > o.top or p.top < o.bottom:
            return False
        return polygons_overlap(object_footprint(o, 0.5 * self.gap), object_footprint(p, 0.5 * self.gap))

    def placement_problems(self, o: SceneObject, others, exclude: set[str] | None = None) -> list[Problem]:
        """Rules for an upright object on the table: in the usable area, on the table, clear of others."""
        exclude = (exclude or set()) | {o.id}
        out = []
        z = self.world.zone_problem(o.pos[0], o.pos[1], o.name)
        if z is not None:
            out.append(z)
        corners = object_footprint(o)
        if np.max(np.hypot(corners[:, 0], corners[:, 1])) > self.world.table_radius - 0.005:
            out.append(Problem("off_table", f"{o.name} would stick out over the table edge."))
        fp = object_footprint(o, 0.5 * self.gap)
        for p in others:
            if p.id in exclude or p.bottom > o.top or p.top < o.bottom:
                continue
            if polygons_overlap(fp, object_footprint(p, 0.5 * self.gap)):
                out.append(Problem("overlap", f"{o.name} would touch {p.name}.",
                                   f"Leave at least {1e3 * self.gap:.0f} mm between objects.", p.id))
                break
        return out

    def add_object(self, data: dict) -> dict:
        with self.lock:
            if len(self.objects) >= MAX_OBJECTS:
                raise StoreError(f"The table holds at most {MAX_OBJECTS} objects.")
            template_id = data.get("template_id")
            if template_id:
                t = self.library.get(template_id)
                if t is None:
                    raise StoreError("No such library item.", status=404)
                model = ObjectModel(name=t.name, size=list(t.size), mass=t.mass, color=t.color, asset_id=t.asset_id,
                                    mesh=copy.deepcopy(t.mesh))
            else:
                model = self._clean_model(data)
            try:
                x, y, yaw = float(data["x"]), float(data["y"]), float(data.get("yaw", 0.0))
            except (KeyError, TypeError, ValueError):
                raise StoreError("A placement needs x and y in metres.") from None
            o = self._make_object(model, x, y, yaw, template_id)
            same = [p for p in self.objects.values() if p.name == o.name or p.name.startswith(o.name + " ")]
            if same:
                o.name = f"{o.name} {len(same) + 1}"
            problems = self.placement_problems(o, self.objects.values())
            if problems:
                raise StoreError(problems[0].message, problems)
            self.objects[o.id] = o
            self._bump()
            return self._object_dict(o, list(self.objects.values()))

    def update_object(self, object_id: str, data: dict) -> dict:
        with self.lock:
            o = self.objects.get(object_id)
            if o is None:
                raise StoreError("No such object.", status=404)
            new = copy.deepcopy(o)
            objs = list(self.objects.values())
            if "name" in data:
                new.name = str(data["name"] or "").strip()[:NAME_MAX] or o.name
            if "mass" in data:
                try:
                    new.mass = float(data["mass"])
                except (TypeError, ValueError):
                    raise StoreError("Mass must be a number in kg.") from None
                bad = self.rules.shape_problems(new.name, new.size, new.mass)
                if bad:
                    raise StoreError(bad[0].message, bad)
            if "color" in data:
                if not COLOR_RE.match(str(data["color"])):
                    raise StoreError("Colours are written as #rrggbb.")
                new.color = str(data["color"])
            if any(k in data for k in ("x", "y", "yaw")):
                carrying = [p.name for p in objs if o.id in supports_of(p, objs)]
                if carrying:
                    raise StoreError(f"{carrying[0]} is standing on {o.name}. Move it off first.")
                try:
                    x = float(data.get("x", o.pos[0]))
                    y = float(data.get("y", o.pos[1]))
                    yaw = float(data.get("yaw", o.yaw))
                except (TypeError, ValueError):
                    raise StoreError("Positions are numbers in metres, yaw in radians.") from None
                if not all(np.isfinite(v) for v in (x, y, yaw)):
                    raise StoreError("Positions must be finite numbers.")
                # Editing always puts the object upright on the table (it also rights tipped objects).
                new.pos = [x, y, self.world.table_height + 0.5 * float(new.size[2])]
                new.quat = quat_from_yaw(yaw)
                problems = self.placement_problems(new, objs)
                if problems:
                    raise StoreError(problems[0].message, problems)
            self.objects[object_id] = new
            self._bump()
            return self._object_dict(new, list(self.objects.values()))

    def remove_object(self, object_id: str) -> None:
        with self.lock:
            o = self.objects.get(object_id)
            if o is None:
                raise StoreError("No such object.", status=404)
            objs = list(self.objects.values())
            carrying = [p.name for p in objs if o.id in supports_of(p, objs)]
            if carrying:
                raise StoreError(f"{carrying[0]} is standing on {o.name}. Remove it first.")
            del self.objects[object_id]
            self._bump()

    def clear(self) -> None:
        with self.lock:
            self.objects.clear()
            self._bump()

    def reset_layout(self, layout: str = "default") -> None:
        with self.lock:
            self.objects.clear()
            self._seed(layout)

    def restore(self, snap: dict) -> None:
        with self.lock:
            self.objects = {o.id: copy.deepcopy(o) for o in snap["objects"]}
            self._bump()

    def set_poses(self, poses: dict[str, tuple[list[float], list[float]]]) -> None:
        """Measured poses from the simulation (after a task step)."""
        with self.lock:
            for oid, (pos, quat) in poses.items():
                o = self.objects.get(oid)
                if o is not None:
                    o.pos = [round(float(v), 6) for v in pos]
                    o.quat = [round(float(v), 7) for v in quat]
            self._bump()

    def _bump(self) -> None:
        self.revision += 1
        self._save_scene()
