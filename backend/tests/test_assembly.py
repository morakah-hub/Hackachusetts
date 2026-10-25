"""Physical assembly: the bridge and the tower are built through contact and friction only."""
import mujoco
import numpy as np

from conftest import names


def test_no_hidden_attachment_in_model(worker):
    m = worker.model
    for i in range(m.neq):          # only the Menagerie finger coupling; nothing welds an object to the hand
        assert m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
    for o in worker.ids.objects.values():
        assert m.geom_contype[o.geom] != 0 and m.geom_conaffinity[o.geom] != 0


def test_camera_guided_bridge(worker, planner, store):
    plan = planner.plan("assemble", {"recipe": "bridge"})
    assert plan.ok, [p.message for p in plan.all_problems()]
    assert [leg.object_name for leg in plan.legs] == ["Pillar A", "Pillar B", "Beam"]
    run = worker.run_sync(plan)
    assert run["state"] == "succeeded", run["failure"]
    for leg in run["legs"]:
        r = leg["result"]
        assert r["placement_error_mm"] < 5 and abs(r["tilt_deg"]) < 3 and r["perception"] == "camera_rgbd"
        assert r["lift_mm"] > 50
    ids = names(store)
    beam = store.objects[ids["Beam"]]
    pa, pb = store.objects[ids["Pillar A"]], store.objects[ids["Pillar B"]]
    assert abs(beam.bottom - pa.top) < 0.004 and abs(beam.bottom - pb.top) < 0.004, "beam rests on both pillars"
    assert np.hypot(pa.pos[0] - pb.pos[0], pa.pos[1] - pb.pos[1]) > 0.09
    states = {o["name"]: o["state"] for o in store.scene_dict()["objects"]}
    assert states["Beam"] == "stacked"


def test_tower_known_world(worker, planner, store):
    worker.mode = "known_world"
    plan = planner.plan("assemble", {"recipe": "tower", "target": [0.45, 0.10]})
    assert plan.ok, [p.message for p in plan.all_problems()]
    run = worker.run_sync(plan)
    assert run["state"] == "succeeded", run["failure"]
    ids = names(store)
    base, mid, top = (store.objects[ids[n]] for n in ("Base block", "Middle block", "Top block"))
    assert np.hypot(base.pos[0] - 0.45, base.pos[1] - 0.10) < 0.01
    assert abs(mid.bottom - base.top) < 0.004 and abs(top.bottom - mid.top) < 0.004
    assert np.hypot(top.pos[0] - base.pos[0], top.pos[1] - base.pos[1]) < 0.01


def test_missing_parts_are_explained(planner, store):
    store.remove_object(names(store)["Beam"])
    plan = planner.plan("assemble", {"recipe": "bridge"})
    assert not plan.ok and plan.problems[0].code == "missing_parts"
