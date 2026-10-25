"""Robotics Lab Workbench (v0.7): local interactive front end over the project's simulation backends.

Python API (the same path the browser UI and the command console use):

    from workbench import RunRequest, run_to_end, Recording, inspect

    req = RunRequest(scenario="pick_place", controller="torque_mpc", observer=True, payload_kg=1.15)
    print(req.summary())                    # resolved settings; req.validate() raises RequestError with a reason
    result, rec = run_to_end(req)           # runs headless, saves to results/workbench/runs/<run_id>/
    print(result.outcome, result.metrics["placement_error_mm"])
    rec = Recording.load(result.directory)  # reopen later: rec.t, rec.data["q"], rec.snapshots, rec.events
    print(inspect.explain(rec, 6.0)["lines"])          # "explain this moment" from recorded data
    sections = inspect.inspect(rec, "mpc", 6.0, {"joint": 1})["sections"]

Modules: api (RunRequest, capability registry, presets), runs (LiveRun adapters, Recorder, Recording),
session (worker thread owning physics + rendering), server (local HTTP + UI), commands (console grammar),
inspect (inspectors, explanations, pre-flight), sandbox (one-joint MPC), experiments (guided comparisons).
"""

from workbench import inspect  # noqa: F401
from workbench.api import RequestError, RunRequest, RunResult, RunStatus, capabilities  # noqa: F401
from workbench.runs import LiveRun, Recording, run_to_end  # noqa: F401
