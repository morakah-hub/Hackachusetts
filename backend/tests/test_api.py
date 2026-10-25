"""HTTP API against a live worker thread (the same code path as the app)."""
import time

import pytest
from fastapi.testclient import TestClient

from armstrong.api import Studio, create_app
from armstrong.rules import RobotRules, World
from armstrong.sim import SimWorker
from armstrong.store import StudioStore


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rules, world = RobotRules(), World()
    store = StudioStore(tmp_path_factory.mktemp("data"), rules, world)
    worker = SimWorker(store, speed=0)
    app = create_app(Studio(worker, store, world, rules))
    worker.start_thread()
    with TestClient(app) as c:
        yield c
    worker.shutdown()


def test_config_scene_library_geometry(client):
    cfg = client.get("/api/config").json()
    assert cfg["robot"]["name"] == "Franka Emika Panda" and cfg["tasks"][0]["id"] == "assemble"
    scene = client.get("/api/scene").json()
    assert len(scene["objects"]) == 6 and all(o["can_lift"] for o in scene["objects"])
    assert any(i["id"] == "lib_beam" for i in client.get("/api/library").json())
    man = client.get("/api/geometry").json()
    blob = client.get("/api/geometry.bin").content
    assert len(blob) == man["byte_length"] and "hand" in man["bodies"] and "table" in man["bodies"]


def test_edit_rules(client):
    r = client.post("/api/scene/objects", json={"template_id": "lib_wood_cube", "x": 0.9, "y": 0.0})
    assert r.status_code == 400 and r.json()["problems"][0]["code"] == "out_of_reach"
    r = client.post("/api/scene/objects", json={"template_id": "lib_wood_cube", "x": 0.45, "y": 0.0})
    assert r.status_code == 200
    oid = r.json()["id"]
    assert client.patch(f"/api/scene/objects/{oid}", json={"x": 0.47, "y": 0.02}).status_code == 200
    assert client.delete(f"/api/scene/objects/{oid}").status_code == 200


def test_check_run_and_camera(client):
    for _ in range(100):
        if client.get("/api/state").json().get("camera"):
            break
        time.sleep(0.1)
    assert client.get("/api/camera/rgb").headers["content-type"] == "image/jpeg"
    scene = client.get("/api/scene").json()
    pid = next(o["id"] for o in scene["objects"] if o["name"] == "Pillar A")
    params = {"object": pid, "target": [0.45, 0.0]}
    plan = client.post("/api/tasks/check", json={"task_id": "move", "params": params}).json()
    assert plan["ok"], plan["problems"]
    res = client.post("/api/tasks/run", json={"task_id": "move", "params": params})
    assert res.status_code == 200, res.text
    run_id = res.json()["run"]["id"]
    r = client.post("/api/scene/objects", json={"template_id": "lib_wood_cube", "x": 0.6, "y": -0.1})
    assert r.status_code == 409, "no edits while a task runs"
    st = None
    for _ in range(900):
        st = client.get("/api/state").json()
        if st["run"] and st["run"]["id"] == run_id and st["run"]["state"] not in ("running", "starting"):
            break
        time.sleep(0.1)
    assert st["run"]["state"] == "succeeded", st["run"]["failure"]
    exp = client.get(f"/api/runs/{run_id}/export")
    assert exp.status_code == 200 and exp.json()["run"]["id"] == run_id
    assert client.post("/api/scene/restore").status_code == 200
    assert client.post("/api/run/dismiss").status_code == 200
