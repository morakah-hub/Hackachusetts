"""Tests for v0.3 benchmark extensions: model mismatch, Cartesian scenarios, actuator labelling."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.benchmark import BENCHMARK_CONFIG, run_benchmark  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402


def test_perturbed_model_scales_only_robot_bodies_and_leaves_plant_alone():
    engine = SimulationEngine("table")
    plant_mass = engine.model.body_mass.copy()
    wrong = engine.perturbed_model(mass_scale=0.85)
    robot = engine._robot_body_ids()
    others = [b for b in range(engine.model.nbody) if b not in robot]
    np.testing.assert_allclose(wrong.body_mass[robot], 0.85 * plant_mass[robot])
    np.testing.assert_allclose(wrong.body_inertia[robot], 0.85 * engine.model.body_inertia[robot])
    np.testing.assert_array_equal(wrong.body_mass[others], plant_mass[others])
    np.testing.assert_array_equal(engine.model.body_mass, plant_mass)  # plant untouched


def test_model_mismatch_reaches_model_based_controllers_only():
    cfg = load_config()
    cfg["controller"] = {**cfg["controller"], "type": "torque_mpc"}
    cfg["controller_model"] = {"mass_scale": 0.85}
    r = SimulationRunner(cfg)
    ctrl_model = r.controller.model.model
    assert ctrl_model is r.engine.controller_model and ctrl_model is not r.engine.model
    assert r.controller.fallback.dynamics.model is ctrl_model
    hand = r.engine.model.body("hand").id
    assert np.isclose(ctrl_model.body_mass[hand], 0.85 * r.engine.model.body_mass[hand])
    # The model error is real: holding still with the wrong gravity estimate drifts.
    exact = SimulationRunner({**cfg, "controller_model": {}})
    q = r.engine.arm_qpos()
    assert not np.allclose(r.controller.model.gravity_torque(q), exact.controller.model.gravity_torque(q), atol=0.1)


def test_benchmark_labels_actuators_and_scores_cartesian_targets(tmp_path):
    bench = yaml.safe_load(BENCHMARK_CONFIG.read_text())
    bench["controllers"] = {k: bench["controllers"][k] for k in ("pid", "torque_ctc")}
    bench["scenarios"] = {"reach": {**bench["scenarios"]["reach"], "move_time": 1.0, "hold_time": 0.5,
                                    "targets": [{"name": "front", "position": [0.5, 0.0, 0.7]},
                                                {"name": "far", "position": [1.3, 0.0, 0.6]}]}}
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(bench))
    report = run_benchmark(out_dir=None, plots=False, bench_path=path, verbose=False)
    rows = {r["controller"]: r for r in report["results"]}
    assert rows["pid"]["actuation"] == "position" and rows["torque_ctc"]["actuation"] == "torque"
    for r in rows.values():
        assert r["reach_targets"] == 2 and r["reach_rejected"] == 1
        assert r["reach_final_ee_error_max_mm"] < 1.0
        assert r["ee_max_error_mm"] >= 0 and r["finite_state"]
