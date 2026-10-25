"""Failure handling: no blind motion, a missed grasp is detected, stale plans are refused."""
import numpy as np
import pytest

from armstrong.sim import CommandError

from conftest import names


def test_masked_camera_blocks_motion(worker, planner, store):
    worker.faults["camera_masked"] = True
    q0 = worker.arm_q()
    p0 = {k: v[0] for k, v in worker.object_poses().items()}
    run = worker.run_sync(planner.plan("move", {"object": names(store)["Pillar A"], "target": [0.45, 0.0]}))
    assert run["state"] == "failed"
    assert run["failure"]["stage"] == "observe" and run["failure"]["code"] == "PERCEPTION_UNAVAILABLE"
    assert np.abs(worker.arm_q() - q0).max() < 0.01, "no blind motion may start"
    for k, v in worker.object_poses().items():
        assert np.linalg.norm(np.subtract(v[0], p0[k])) < 1e-3


def test_disabled_gripper_fails_at_close(worker, planner, store):
    worker.mode = "known_world"
    worker.faults["gripper_disabled"] = True
    run = worker.run_sync(planner.plan("move", {"object": names(store)["Pillar A"], "target": [0.45, 0.0]}))
    assert run["state"] == "failed" and run["failure"]["code"] == "GRASP_FAILED"
    assert run["failure"]["stage"] == "close"


def test_stale_plan_is_refused(worker, planner, store):
    plan = planner.plan("move", {"object": names(store)["Pillar A"], "target": [0.45, 0.0]})
    store.update_object(names(store)["Beam"], {"x": 0.62, "y": 0.15})
    with pytest.raises(CommandError) as exc:
        worker.command_sync("run", plan=plan)
    assert exc.value.code == "STALE_SCENE"
