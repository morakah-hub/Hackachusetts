"""ArmStrong: engine extensions, object rules, scene store, planner, geometry export, a physical run and
the HTTP API (served by uvicorn on a free port; no extra test dependencies)."""

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robots.panda import Panda  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.manipulation import case_overrides  # noqa: E402
from studio import geometry  # noqa: E402
from studio.planner import TaskPlanner, scene_overrides  # noqa: E402
from studio.robot import RobotRules, quat_from_yaw, yaw_of_quat  # noqa: E402
from studio.runner import Runner  # noqa: E402
from studio.scene import StoreError, StudioStore, World, footprint, polygons_overlap  # noqa: E402
from studio.settings import StudioSettings  # noqa: E402
from studio.workspace import DEFAULT_FILE, derive_zone, load_or_measure, workspace_key  # noqa: E402


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    return StudioSettings.load(data_dir=tmp_path_factory.mktemp("studio"))


@pytest.fixture(scope="module")
def world(settings):
    ws = load_or_measure(settings, log=lambda *_: None)
    z = ws["zone"]
    return World(settings.table_height, ws["table_radius"], z["r_in"], z["r_out"],
                 min(z["theta_max_deg"], settings.studio["workspace"]["max_azimuth_deg"]))


@pytest.fixture()
def store(settings, world, tmp_path):
    return StudioStore(tmp_path, RobotRules(settings), world, settings.limits["gap_m"])


def _place(store, template, x, y, yaw=0.0):
    return store.add_object({"template_id": template, "x": x, "y": y, "yaw": yaw})


# ------------------------------------------------------------------ engine and supervisor extensions


def test_overrides_add_free_boxes_with_pose_and_mass():
    q = quat_from_yaw(0.7)
    e = SimulationEngine("studio", Panda(actuation="torque"), {"objects": [
        {"name": "box_a", "pos": [0.5, 0.1, 0.78], "quat": q, "half_size": [0.02, 0.03, 0.03], "mass": 0.7}]})
    m, d = e.model, e.data
    b = m.body("box_a").id
    assert m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
    assert np.isclose(m.body_mass[b], 0.7)
    np.testing.assert_allclose(m.geom_size[m.geom("box_a_geom").id], [0.02, 0.03, 0.03])
    np.testing.assert_allclose(d.xpos[b], [0.5, 0.1, 0.78], atol=1e-9)  # reset() keeps the spec pose
    assert np.isclose(yaw_of_quat(d.xquat[b]), 0.7)
    # The robot is still found by name although the qpos layout moved.
    assert np.allclose(e.arm_qpos(), Panda().home_qpos)


def test_override_errors_for_added_boxes():
    with pytest.raises(ValueError, match="duplicate"):
        SimulationEngine("studio", overrides={"objects": [{"name": "table", "pos": [0, 0, 1], "half_size": [.1, .1, .1],
                                                           "mass": 1.0}]})
    with pytest.raises(ValueError, match="positive half sizes"):
        SimulationEngine("studio", overrides={"objects": [{"name": "b", "pos": [0, 0, 1], "half_size": [.1, 0, .1],
                                                           "mass": 1.0}]})


def test_case_overrides_accept_a_measured_pose_and_place_height():
    pp = {"object": {"body": "o", "geom": "o_geom", "half_size": [0.02, 0.02, 0.03]}, "table_height": 0.75}
    out = case_overrides(pp, {"object_pose": {"pos": [0.4, 0.1, 0.86], "quat": [1, 0, 0, 0]}, "place": [0.5, 0.0],
                              "place_z": 0.91, "mass": 0.3})
    assert out["bodies"]["o"] == {"pos": [0.4, 0.1, 0.86], "quat": [1.0, 0.0, 0.0, 0.0]}
    assert out["bodies"]["place_marker"]["pos"] == [0.5, 0.0, 0.91]
    legacy = case_overrides(pp, {"object": [0.4, 0.1], "place": [0.5, 0.0], "mass": 0.3})
    assert legacy["bodies"]["o"]["pos"] == [0.4, 0.1, 0.78]


# ------------------------------------------------------------------ robot rules and work area


def test_lift_rules_follow_the_configured_limits(settings):
    rules = RobotRules(settings)
    assert [p.code for p in rules.lift_problems("x", [0.05, 0.05, 0.06], 0.4)] == []
    assert [p.code for p in rules.lift_problems("x", [0.05, 0.05, 0.06], 3.2)] == ["too_heavy"]
    assert "too_wide" in [p.code for p in rules.lift_problems("x", [0.12, 0.09, 0.06], 0.4)]
    assert [p.code for p in rules.lift_problems("x", [0.11, 0.05, 0.06], 0.4)] == []  # grips the 50 mm side
    assert "too_short" in [p.code for p in rules.lift_problems("x", [0.05, 0.05, 0.02], 0.1)]
    opts = rules.grasp_options([0.11, 0.05, 0.06], 0.3)
    assert len(opts) == 2 and all(np.isclose(o.width, 0.05) for o in opts)  # one side, two hand turns
    assert np.isclose(abs(opts[0].yaw - opts[1].yaw), np.pi)


def test_packaged_work_area_matches_the_configuration(settings, world):
    data = json.loads(DEFAULT_FILE.read_text(encoding="utf-8"))
    assert data["key"] == workspace_key(settings), "re-run: python -m studio.workspace --write-default"
    assert 0.25 < world.r_in < world.r_out < settings.studio["robot"]["datasheet_reach_m"]
    assert np.isclose(world.table_radius, world.r_out + settings.studio["workspace"]["table_rim_m"])


def test_zone_is_the_common_band_of_the_usable_azimuths():
    ws = {"azimuth_step_deg": 10, "min_ring_width_m": 0.3, "margin_m": 0.02}
    profile = [{"theta_deg": t, "r_min": 0.3, "r_max": 0.75 if abs(t) <= 120 else 0.5} for t in range(-180, 180, 10)]
    profile[18]["r_max"] = 0.72  # azimuth 0
    z = derive_zone(profile, ws)
    assert z == {"r_in": 0.32, "r_out": 0.70, "theta_max_deg": 120.0}


# ------------------------------------------------------------------ scene store


def test_placement_rules(store, world):
    assert store.scene_dict()["objects"], "a fresh store starts with the sample layout"
    store.clear()
    with pytest.raises(StoreError, match="reach"):
        _place(store, "lib_wood_cube", world.r_out + 0.02, 0.0)
    with pytest.raises(StoreError, match="base"):
        _place(store, "lib_wood_cube", 0.2, 0.0)
    with pytest.raises(StoreError, match="behind"):
        _place(store, "lib_wood_cube", -0.5, 0.01)
    a = _place(store, "lib_alu_block", 0.5, 0.0)
    with pytest.raises(StoreError, match="touch"):
        _place(store, "lib_wood_cube", 0.55, 0.0)
    b = _place(store, "lib_wood_cube", 0.5, 0.12)
    moved = store.update_object(b["id"], {"x": 0.45, "y": 0.2, "yaw": 0.5})
    assert np.isclose(moved["yaw"], 0.5) and moved["state"] == "on_table"
    heavy = _place(store, "lib_toolbox", 0.1, -0.6)
    assert not heavy["can_lift"] and "too_heavy" in [p["code"] for p in heavy["problems"]]
    assert {o["id"] for o in store.scene_dict()["objects"]} == {a["id"], b["id"], heavy["id"]}


def test_store_persists_scene_and_library(settings, world, tmp_path):
    rules = RobotRules(settings)
    s1 = StudioStore(tmp_path, rules, world, 0.015)
    item = s1.add_library_item({"name": "Mug box", "size": [0.05, 0.05, 0.09], "mass": 0.3, "color": "#123456"})
    o = s1.add_object({"template_id": item["id"], "x": 0.5, "y": -0.1})
    s2 = StudioStore(tmp_path, rules, world, 0.015)
    assert item["id"] in {i["id"] for i in s2.library_list()}
    assert o["id"] in {x["id"] for x in s2.scene_dict()["objects"]}
    with pytest.raises(StoreError):
        s2.add_library_item({"name": "bad", "size": [0.05, 0.05], "mass": 0.3})
    with pytest.raises(StoreError, match="Unsupported"):
        s2.save_asset("model.exe", b"x")
    meta = s2.save_asset("cube.stl", b"solid cube\nendsolid cube\n")
    path, media = s2.asset_file(meta["id"])
    assert path.read_bytes().startswith(b"solid") and media == "model/stl"


def test_footprint_overlap():
    a = footprint(0.0, 0.0, 0.0, [0.1, 0.1, 0.1])
    assert polygons_overlap(a, footprint(0.09, 0.0, 0.0, [0.1, 0.1, 0.1]))
    assert not polygons_overlap(a, footprint(0.11, 0.0, 0.0, [0.1, 0.1, 0.1]))
    assert polygons_overlap(a, footprint(0.115, 0.0, np.pi / 4, [0.1, 0.1, 0.1]))  # rotated corner reaches in


# ------------------------------------------------------------------ planner


def test_planner_decomposes_and_explains(settings, world, store):
    store.clear()
    rules = RobotRules(settings)
    planner = TaskPlanner(settings, store, world, rules)
    a = _place(store, "lib_alu_block", 0.5, 0.15)
    b = _place(store, "lib_wood_cube", 0.5, -0.15)
    heavy = _place(store, "lib_toolbox", 0.05, -0.62)
    plan = planner.plan("move", {"object": a["id"], "target": [0.45, 0.3]})
    assert plan.ok and len(plan.legs) == 1 and plan.legs[0].grasp_width == pytest.approx(0.05)
    swap = planner.plan("swap", {"a": a["id"], "b": b["id"]})
    assert swap.ok and [leg.purpose for leg in swap.legs] == ["buffer", "move", "move"]
    stack = planner.plan("stack", {"object": b["id"], "base": a["id"]})
    assert stack.ok and stack.legs[0].supports_to == [a["id"]]
    assert np.isclose(stack.legs[0].target[2], world.table_height + 0.06 + 0.0225)
    far = planner.plan("move", {"object": a["id"], "target": [-0.30, -0.62]})  # long swing: set down halfway
    assert far.ok and [leg.purpose for leg in far.legs] == ["via", "move"]
    too_heavy = planner.plan("move", {"object": heavy["id"], "target": [0.5, 0.0]})
    assert not too_heavy.ok and "too_heavy" in {p.code for p in too_heavy.problems}
    blocked = planner.plan("move", {"object": b["id"], "target": [0.5, 0.15]})
    assert not blocked.ok and "overlap" in {p.code for p in blocked.problems}


# ------------------------------------------------------------------ geometry export


def test_geometry_export_matches_the_compiled_model(world):
    e = SimulationEngine("studio", Panda(actuation="torque"), scene_overrides(world, []))
    bodies = ["table", *geometry.robot_body_names(e)]
    manifest, blob = geometry.export(e, bodies)
    assert manifest["byte_length"] == len(blob) and len(blob) % 4 == 0
    meshes = [g for g in manifest["geoms"] if g["type"] == "mesh"]
    assert len(meshes) == 58 and len(manifest["meshes"]) == 56  # Panda visual geoms (finger mesh shared)
    for mb in manifest["meshes"]:
        idx = np.frombuffer(blob, "<u2" if mb["index_type"] == "uint16" else "<u4", mb["index_count"], mb["indices"])
        assert idx.max() < mb["vertex_count"]
    top = next(g for g in manifest["geoms"] if g["name"] == "table_top")
    assert top["type"] == "cylinder" and np.isclose(top["size"][0], world.table_radius)
    poses = geometry.body_poses(e, ["hand"])
    mujoco.mj_kinematics(e.model, e.data)
    np.testing.assert_allclose(poses["hand"][:3], e.data.xpos[e.model.body("hand").id], atol=1e-6)


# ------------------------------------------------------------------ physical run


def _wait(runner, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = runner.status()["run"]
        if run and run["state"] in ("succeeded", "failed", "stopped"):
            return run
        time.sleep(0.05)
    raise TimeoutError("run did not finish")


def test_runner_moves_and_stacks_objects_physically(settings, world, store):
    store.clear()
    rules = RobotRules(settings)
    planner = TaskPlanner(settings, store, world, rules)
    a = _place(store, "lib_alu_block", 0.5, 0.15)
    b = _place(store, "lib_wood_cube", 0.5, -0.15)
    runner = Runner(settings, store, world)
    try:
        runner.set_speed(0)
        plan = planner.plan("move", {"object": a["id"], "target": [0.45, 0.3]})
        runner.start(plan)
        with pytest.raises(RuntimeError, match="already running"):
            runner.start(plan)  # one run at a time
        run = _wait(runner)
        assert run["state"] == "succeeded", run["failure"]
        res = run["legs"][0]["result"]
        assert res["placement_error_mm"] < 10.0 and res["transport"] == "arc"
        moved = next(o for o in store.scene_dict()["objects"] if o["id"] == a["id"])
        assert np.hypot(moved["pos"][0] - 0.45, moved["pos"][1] - 0.3) < 0.01 and moved["state"] == "on_table"
        assert runner.status()["robot_at_home"]  # the last leg returns to the rest posture
        runner.dismiss()
        runner.start(planner.plan("stack", {"object": b["id"], "base": a["id"]}))
        run = _wait(runner)
        assert run["state"] == "succeeded", run["failure"]
        top = next(o for o in store.scene_dict()["objects"] if o["id"] == b["id"])
        assert top["state"] == "stacked" and top["supports"] == [a["id"]]
        frame = runner.frame()
        assert set(frame["objects"]) == {a["id"], b["id"]} and "hand" in frame["bodies"]
    finally:
        runner.close()


# ------------------------------------------------------------------ HTTP API


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_api(tmp_path):
    import uvicorn

    from studio.server import create_app

    port = _free_port()
    app = create_app(StudioSettings.load(data_dir=tmp_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        status, body = call("GET", "/api/config")
        cfg = json.loads(body)
        assert status == 200 and cfg["robot"]["payload_kg"] == 3.0 and len(cfg["tasks"]) == 6
        status, body = call("GET", "/api/geometry.bin")
        assert status == 200 and len(body) == json.loads(call("GET", "/api/geometry")[1])["byte_length"]
        status, body = call("POST", "/api/scene/objects", {"template_id": "lib_wood_cube", "x": 0.9, "y": 0.0})
        err = json.loads(body)
        assert status == 400 and err["problems"][0]["code"] == "out_of_reach"
        scene = json.loads(call("GET", "/api/scene")[1])
        oid = scene["objects"][0]["id"]
        status, body = call("POST", "/api/tasks/check", {"task_id": "hand_over", "params": {"object": oid}})
        assert status == 200 and json.loads(body)["ok"]
        status, body = call("POST", "/api/tasks/run", {"task_id": "move", "params": {"object": oid, "target": [5, 5]}})
        assert status == 422
        status, body = call("POST", "/api/assets?filename=part.stl", None)
        assert status == 400  # empty upload
        state = json.loads(call("GET", "/api/state")[1])
        assert state["run"] is None and "link7" in state["frame"]["bodies"]
    finally:
        server.should_exit = True
        thread.join(15)
