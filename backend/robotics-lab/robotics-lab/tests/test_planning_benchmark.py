"""v0.5 planning benchmark: the scenarios are what they claim to be, and the driver runs."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.motion_planner import MotionPlanner, PlanningSettings  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.planning_benchmark import load_bench, run_planning_benchmark, scenario_planning  # noqa: E402

BENCH = load_bench()


@pytest.fixture(scope="module")
def planners():
    cache = {}

    def get(env, settings):
        engine = cache.setdefault(env, SimulationEngine(env))
        return MotionPlanner.from_engine(engine, PlanningSettings.from_config(settings))
    return get


def test_execution_scenarios_have_valid_starts_goals_and_the_claimed_blockage(planners):
    for name, sc in BENCH["scenarios"].items():
        if not sc.get("execute"):
            continue
        p = planners(sc["environment"], scenario_planning(BENCH, sc))
        assert p.checker.check(np.array(sc["start"])).valid, name
        plan = p.plan_to_position(np.array(sc["target"]), np.array(sc["start"]), seed=0)
        assert plan.success, (name, plan.reason)
        blocked = name != "easy_free"
        assert plan.stats["direct_edge_valid"] is (not blocked), name


def test_failure_scenarios_fail_for_the_documented_reason(planners):
    expected = {"unreachable": "unreachable", "start_in_collision": "invalid_start",
                "goal_in_collision": "no_valid_ik", "timeout": "timeout", "node_budget": "node_budget"}
    for name, status in expected.items():
        sc = BENCH["scenarios"][name]
        p = planners(sc["environment"], scenario_planning(BENCH, sc))
        start = np.array(sc["start"])
        plan = (p.plan_to_position(np.array(sc["target"]), start, seed=0) if "target" in sc
                else p.plan_joint(start, np.array(sc["goal"]), seed=0))
        assert not plan.success and plan.status == status, (name, plan.status, plan.reason)


def test_multi_ik_scenario_needs_a_posture_other_than_the_v04_ik_solution(planners):
    sc = BENCH["scenarios"]["multi_ik_near_obstacle"]
    p = planners(sc["environment"], scenario_planning(BENCH, sc))
    start, target = np.array(sc["start"]), np.array(sc["target"])
    v04 = p.kin.plan_target(target, start)
    assert v04.success  # accepted by the IK-only check (zero clearance) ...
    assert not p.checker.check(v04.q).valid  # ... but closer than the 20 mm clearance
    plan = p.plan_to_position(target, start, seed=0)
    assert plan.success and plan.stats["ik_valid"] >= 1
    assert np.abs(plan.q - v04.q).max() > 0.05


def test_driver_reports_every_seed_and_execution_outcomes(tmp_path):
    report = run_planning_benchmark(out_dir=tmp_path, scenarios=["easy_free", "node_budget"], seeds=2,
                                    execution=True, plots=False, verbose=False)
    runs = report["planning_runs"]
    assert len(runs) == 4  # failures are reported, not dropped
    summary = {s["scenario"]: s for s in report["planning_summary"]}
    assert summary["easy_free"]["success_rate"] == 1.0 and summary["node_budget"]["success_rate"] == 0.0
    assert summary["node_budget"]["statuses"] == {"node_budget": 2}
    ex = report["execution_runs"]
    assert {r["controller"] for r in ex} == set(BENCH["execution"]["controllers"])
    assert all(r["outcome"] == "success" and r["contact_steps"] == 0 for r in ex)
    for f in ("results.json", "planning_runs.csv", "planning_summary.csv", "execution_runs.csv", "summary.md"):
        assert (tmp_path / f).is_file()
