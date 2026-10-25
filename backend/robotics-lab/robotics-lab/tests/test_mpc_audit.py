"""v0.6 milestone 1: the torque MPC's assembled QP equals its documented formulation.

Cheap parts of tools/audit_torque_mpc.py (the full audit adds prediction-vs-rollout and closed-loop
status statistics). The expected values are computed independently of the controller's matrix code:
the 1-joint N = 2 case eliminates the states by hand and minimizes the documented cost directly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit_torque_mpc as audit  # noqa: E402


@pytest.mark.parametrize("tau_limit, d_hat", [(50.0, 0.0), (50.0, 1.0)])
def test_first_torque_equals_hand_minimizer_unconstrained(tau_limit, d_hat):
    r = audit.run_small_case(tau_limit=tau_limit, d_hat=d_hat)
    assert np.allclose(r["controller_plan"], r["expected_unconstrained"], atol=1e-6)
    assert r["prediction_vs_recursion"] < 1e-12
    assert r["objective_mismatch"] < 1e-10


def test_first_torque_equals_hand_minimizer_with_active_torque_limit():
    r = audit.run_small_case(tau_limit=3.2)
    assert r["expected_unconstrained"][0] > 3.2           # the limit is really active
    assert r["applied"] == pytest.approx(r["expected_bounded_u0"], abs=1e-5)
    assert r["applied"] <= 3.2 + 1e-9


def test_feedthrough_subtracts_disturbance_change_since_update():
    r = audit.feedthrough_sign()
    assert r["between_updates"] == pytest.approx(r["expected_between_updates"], abs=1e-12)


def test_observer_estimates_unmeasured_joint_torque_with_plant_sign():
    r = audit.plant_observer_sign(seconds=0.4)
    assert r["max_error_nm"] < 1e-3


def test_full_size_objective_and_constraint_rows_match_documentation():
    r = audit.full_size_checks()
    assert r["objective_mismatch"] < 1e-9
    for key in ("dynamics_row_residual", "torque_rows_vs_tau_over_scale", "state_lower_rows_vs_x_plus_s",
                "state_upper_rows_vs_x_minus_s", "slack_rows_vs_s", "state_bounds_on_matching_entries",
                "torque_bounds_vs_limits_nm"):
        assert r[key] < 1e-9, key
