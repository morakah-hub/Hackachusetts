"""Every pre-written task plans and runs physically (known-world mode for speed)."""
import numpy as np
import pytest

from conftest import names


@pytest.fixture
def objects_worker(worker, store):
    store.reset_layout("objects")
    worker._sync_scene()
    worker.mode = "known_world"
    return worker


def run_ok(worker, planner, task, params):
    worker._sync_scene()
    plan = planner.plan(task, params)
    assert plan.ok, [p.message + " " + p.hint for p in plan.all_problems()]
    run = worker.run_sync(plan)
    assert run["state"] == "succeeded", run["failure"]
    return plan, run


def test_move_and_stack(objects_worker, planner, store):
    ids = names(store)
    run_ok(objects_worker, planner, "move", {"object": ids["Aluminium block"], "target": [0.45, 0.05]})
    alu = store.objects[ids["Aluminium block"]]
    assert np.hypot(alu.pos[0] - 0.45, alu.pos[1] - 0.05) < 0.01
    run_ok(objects_worker, planner, "stack", {"object": ids["Wooden cube"], "base": ids["Aluminium block"]})
    assert abs(store.objects[ids["Wooden cube"]].bottom - store.objects[ids["Aluminium block"]].top) < 0.004


def test_swap(objects_worker, planner, store):
    ids = names(store)
    a0, b0 = list(store.objects[ids["Juice carton"]].pos), list(store.objects[ids["Tea box"]].pos)
    run_ok(objects_worker, planner, "swap", {"a": ids["Juice carton"], "b": ids["Tea box"]})
    a1, b1 = store.objects[ids["Juice carton"]].pos, store.objects[ids["Tea box"]].pos
    assert np.hypot(a1[0] - b0[0], a1[1] - b0[1]) < 0.01 and np.hypot(b1[0] - a0[0], b1[1] - a0[1]) < 0.01


def test_sort_by_weight(objects_worker, planner, store):
    run_ok(objects_worker, planner, "sort_weight", {})
    row = sorted(store.objects.values(), key=lambda o: o.pos[1])
    assert [o.mass for o in row] == sorted(o.mass for o in row)


def test_hand_over(objects_worker, planner, store):
    ids = names(store)
    run_ok(objects_worker, planner, "hand_over", {"object": ids["Juice carton"]})
    j = store.objects[ids["Juice carton"]]
    assert np.hypot(j.pos[0], j.pos[1]) > 0.6


def test_planner_rejections(planner, store):
    ids = names(store)
    plan = planner.plan("move", {"object": ids["Pillar A"], "target": [0.9, 0.0]})
    assert not plan.ok and plan.all_problems()[0].code == "out_of_reach"
    plan = planner.plan("move", {"object": ids["Pillar A"], "target": [-0.5, 0.0]})
    assert not plan.ok and plan.all_problems()[0].code == "outside_zone"
    store.add_object({"template_id": "lib_toolbox", "x": 0.55, "y": 0.0, "yaw": 1.5708})
    tool = names(store)["Toolbox"]
    plan = planner.plan("move", {"object": tool, "target": [0.45, 0.1]})
    assert not plan.ok and "too_heavy" in {p.code for p in plan.all_problems()}
