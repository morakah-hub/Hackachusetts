"""v0.7 workbench: requests, registry, session lifecycle, CLI agreement, commands."""

import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.manipulation import run_case  # noqa: E402
from simulation.runner import SimulationRunner  # noqa: E402
from workbench.api import RequestError, RunRequest  # noqa: E402
from workbench.commands import CommandConsole  # noqa: E402
from workbench.runs import LiveRun, Recording, run_to_end  # noqa: E402
from workbench.session import Session  # noqa: E402


def wait(session, states, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if session.status.state in states:
            return session.status
        time.sleep(0.05)
    raise AssertionError(f"timeout in state {session.status.state}: {session.status.message}")


# ------------------------------------------------------------------ requests and registry


@pytest.mark.parametrize("changes, words", [
    ({"scenario": "pick_place", "controller": "joint_pid", "observer": False}, "only with the torque-level"),
    ({"scenario": "wall_reach", "controller": "joint_pid", "observer": True}, "observer exists only"),
    ({"payload_kg": 5.0}, "outside the supported"),
    ({"object_xy": [0.9, 0.0]}, "outside the supported range"),
    ({"scenario": "pick_place_obstacle", "object_xy": [0.5, 0.0]}, "overlaps the obstacle"),
    ({"scenario": "pick_place", "obstacle": {"pos": [0.5, 0, 0.5], "size": [0.05, 0.05, 0.1]}}, "no editable obstacle"),
    ({"force_pulse": {"body": "table", "direction": [0, 0, 1], "magnitude_n": 5, "start_s": 1, "duration_s": 1}},
     "body must"),
    ({"scenario": "wall_reach", "targets": [[0.45, 0.0, 2.0]]}, "outside the supported range"),
    ({"contact_preset": "magic"}, "unknown contact preset"),
])
def test_invalid_or_unsupported_requests_are_rejected_with_a_reason(changes, words):
    with pytest.raises(RequestError, match=words):
        RunRequest().replace(**changes).validate()


def test_request_resolves_to_the_cli_configs():
    res = RunRequest(scenario="pick_place", controller="torque_ctc", observer=True, payload_kg=1.15).resolve()
    cfg = res["config"]
    assert cfg["physics"] == {"cone": "elliptic", "impratio": 10}
    assert cfg["robot"] == {"gripper_stiffness": 1500.0, "gripper_damping": 20.0}
    assert cfg["controller_model"]["contacts"] is False
    assert cfg["controller"]["ctc"]["disturbance_observer"]["enabled"] is True
    assert res["case"]["mass"] == 1.15
    reach = RunRequest(scenario="wall_reach", controller="torque_mpc").resolve()
    assert reach["config"]["environment"] == "planning_demo"
    assert [t["position"] for t in reach["config"]["task"]["targets"]] == [[0.45, -0.25, 0.5], [0.45, 0.25, 0.5]]


# ------------------------------------------------------------------ same backend as the CLI


def test_pick_place_through_the_workbench_equals_the_cli_backend():
    req = RunRequest(scenario="pick_place", controller="torque_ctc", observer=True, payload_kg=0.4,
                     object_xy=[0.5, 0.2], place_xy=[0.5, -0.2], mpc_snapshots=False)
    res, rec = run_to_end(req, save=False)
    resolved = req.resolve()
    cli = run_case(resolved["config"], resolved["case"])  # what pick_place.py runs for this case
    assert res.success and cli["success"]
    assert res.metrics["placement_error_mm"] == pytest.approx(cli["placement_error_mm"], abs=1e-9)
    assert res.metrics["steps"] == cli["steps"]
    # The recording holds measured (not reference) motion, sampled every 5 physics steps.
    assert np.allclose(np.diff(rec.t), 5 * 0.002, atol=1e-9)
    assert np.abs(rec.data["q"] - rec.data["q_ref"]).max() > 0


def test_wall_reach_through_the_workbench_matches_main_py():
    req = RunRequest(scenario="wall_reach", controller="torque_mpc")
    res, rec = run_to_end(req, save=False)
    assert res.success, res.failure_reason
    runner = SimulationRunner(req.resolve()["config"])     # what main.py --headless runs
    log = runner.run_headless(runner.task.duration).arrays()
    for t in res.metrics["targets"]:
        plan = next(p for p in runner.task.plans if p.name == t["name"])
        i = np.searchsorted(log["time"], runner.task.hold_end_time(plan)) - 1
        cli_err = 1e3 * np.linalg.norm(log["ee"][i] - plan.position)   # the number main.py prints
        # main.py reads site_xpos (one physics step behind) at a 10-step log stride; at rest both are < 0.1 mm.
        assert t["error_at_end_of_hold_mm"] == pytest.approx(cli_err, abs=0.05)
    assert res.metrics["robot_contact_steps"] == 0
    assert len(rec.snapshots) == 450  # every MPC update of the 9 s run was captured


def test_force_pulse_is_applied_only_inside_its_window_and_cleared():
    req = RunRequest(scenario="table_reach", controller="torque_ctc", observer=False,
                     force_pulse={"body": "hand", "direction": [0, 0, -1], "magnitude_n": 20, "start_s": 1.0,
                                  "duration_s": 0.5})
    run = LiveRun(req)
    seen = []
    while run.engine.time < 2.0 and run.step():
        seen.append((run.engine.time, run.current_force[2]))
    t = np.array(seen)
    on = (t[:, 0] > 1.002) & (t[:, 0] <= 1.5)
    assert np.all(t[on, 1] == -20.0) and np.all(t[t[:, 0] <= 1.0, 1] == 0) and np.all(t[t[:, 0] > 1.502, 1] == 0)
    run.stop()
    run.finish()
    assert not run.engine.data.xfrc_applied.any()


# ------------------------------------------------------------------ session lifecycle


@pytest.fixture
def session(tmp_path):
    s = Session(runs_dir=tmp_path)
    yield s
    s.close()
    assert not s.alive


def test_session_runs_pauses_steps_stops_resets_and_runs_again(session):
    session.configure(scenario="pick_place", controller="torque_ctc", observer=True, mpc_snapshots=False)
    session.act("speed", value=0)
    session.act("run")
    wait(session, {"running"})
    time.sleep(0.5)
    session.act("pause")
    st = wait(session, {"paused"})
    time.sleep(0.2)
    t_paused = session.status.sim_time
    session.act("step", physics_steps=1)
    time.sleep(0.5)
    assert session.status.sim_time == pytest.approx(t_paused + 0.002, abs=1e-9)
    assert st.state == "paused"
    session.act("stop")
    wait(session, {"finished", "failed"})
    assert session.last_result["outcome"] == "stopped"
    session.act("reset")
    wait(session, {"ready"})
    assert session.status.sim_time == 0.0
    session.act("run")
    wait(session, {"finished", "failed"}, timeout=180)
    assert session.last_result["outcome"] == "success"
    fid, jpg = session.latest_frame()
    assert jpg[:2] == b"\xff\xd8" and fid > 0
    saved = sorted(p.name for p in session.runs_dir.iterdir())
    assert len(saved) == 2   # the stopped run and the successful one; never-started scenes are not saved
    rec = Recording.load(session.runs_dir / saved[-1])
    assert rec.result["outcome"] == "success" and len(rec.t) > 100


def test_configuration_never_changes_an_active_run(session):
    session.configure(scenario="pick_place", controller="torque_ctc", payload_kg=0.4, mpc_snapshots=False)
    session.act("run")
    wait(session, {"running"})
    session.configure(payload_kg=1.15)
    m = session.live.engine.model
    assert m.body_mass[m.body("object").id] == pytest.approx(0.4)
    assert session.request.payload_kg == 1.15
    session.act("stop")
    wait(session, {"finished", "failed"})


def test_commands_and_gui_produce_the_same_request(session):
    console = CommandConsole(session)
    out = console.execute("load obstacle; controller ctc; observer off; payload 1.15; object 0.5 0.2; "
                          "place 0.45 -0.2; seed 3")
    assert out["ok"], out["output"]
    via_commands = session.request.to_dict()
    gui = RunRequest(scenario="pick_place_obstacle", controller="torque_ctc", observer=False, payload_kg=1.15,
                     object_xy=[0.5, 0.2], place_xy=[0.45, -0.2], plan_seed=3)
    assert via_commands == session.set_request(gui).to_dict()
    bad = console.execute("payload heavy")
    assert not bad["ok"] and "not a number" in bad["output"][-1]
    assert console.execute("frobnicate")["output"][-1].startswith("error")
    assert "COMMANDS" in console.execute("help")["output"][0]


def test_http_server_serves_ui_and_api_and_closes():
    from workbench.server import Workbench
    wb = Workbench(port=0, session=Session(runs_dir=Path(tempfile.mkdtemp())))
    wb.serve_background()
    try:
        html = urllib.request.urlopen(wb.url, timeout=10).read().decode()
        assert "Robotics Lab" in html and "app.js" in html
        caps = json.loads(urllib.request.urlopen(wb.url + "api/capabilities", timeout=10).read())
        assert set(caps["scenarios"]) == {"wall_reach", "table_reach", "pick_place", "pick_place_obstacle"}
        req = urllib.request.Request(wb.url + "api/request", data=json.dumps({"payload_kg": 7}).encode(),
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=10)
        assert err.value.code == 400
    finally:
        wb.close()
    assert not wb.session.alive


def test_run_after_changing_settings_uses_the_new_settings(session):
    wait(session, {"ready"})                      # the default scene is built at start-up
    session.configure(scenario="pick_place", controller="torque_ctc", payload_kg=1.15, mpc_snapshots=False)
    session.act("run")
    wait(session, {"running", "finished", "failed"})
    m = session.live.engine.model
    assert m.body_mass[m.body("object").id] == pytest.approx(1.15)
    assert session.live.request.controller == "torque_ctc"
    session.act("stop")
    wait(session, {"finished", "failed"})


def test_idle_edits_preview_the_scene_and_keep_the_last_result(session):
    session.configure(scenario="pick_place", controller="torque_ctc", payload_kg=0.4, mpc_snapshots=False)
    session.act("speed", value=0)
    session.act("run")
    wait(session, {"running"})
    session.configure(payload_kg=1.15)                 # during the run: no preview, applies next time
    wait(session, {"finished", "failed"}, timeout=180)
    time.sleep(0.5)
    assert session.status.state in ("finished", "failed") and session.last_result is not None
    session.configure(scenario="pick_place_obstacle")  # idle edit: the viewport scene is rebuilt as a preview
    t0 = time.time()
    while time.time() - t0 < 30 and session.live.request.scenario != "pick_place_obstacle":
        time.sleep(0.1)
    assert session.live.request.scenario == "pick_place_obstacle" and session.status.state == "ready"
    m = session.live.engine.model
    assert m.body_mass[m.body("object").id] == pytest.approx(1.15)
    assert session.last_result is not None             # the previous result stays until a new run starts
