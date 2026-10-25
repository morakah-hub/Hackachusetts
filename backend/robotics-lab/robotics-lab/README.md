# Robotics Lab

A modular robotics simulation platform built on MuJoCo and Python. It runs a
physically simulated Franka Emika Panda (7-DOF) on a table. A closed-loop joint
controller moves the arm smoothly between joint configurations, and MuJoCo's
built-in interactive viewer shows the result.

- **v0.1:** joint-space PID outer loop on the Panda's position servos.
- **v0.2:** joint-space model predictive control (MPC) solved with OSQP, a
  MuJoCo-derived prediction model, and a reproducible PID vs MPC benchmark.
- **v0.3:** a torque-actuated Panda variant with computed-torque control and
  torque MPC, Cartesian (x, y, z) reaching via inverse kinematics, a four-way
  controller comparison, and a fix to the v0.2 MPC's fallback recovery
  (see [v0.3](#v03-torque-control-torque-mpc-and-cartesian-reaching)).
- **v0.4:** feasibility-checked trajectory generation (automatic time scaling to
  speed/acceleration/jerk/torque limits), a root-cause analysis of the torque MPC's
  fast-motion results, constraint-violation reporting and tightening, an offset-free
  torque MPC with a disturbance observer, and end-effector orientation metrics
  (see [v0.4](#v04-feasible-trajectories-constraint-diagnostics-disturbance-rejection)).
- **v0.5:** collision-aware motion planning: joint-space RRT-Connect with
  clearance-checked edges, shortcut smoothing and multiple IK postures, executed by the
  existing controllers through the v0.4 trajectory generator, plus an obstacle-avoidance
  demo and a planning benchmark (see [v0.5](#v05-collision-aware-motion-planning)).
- **v0.5.1:** a rendered demo video and a small robustness experiment: the wall's size
  and position change between episodes (known at episode start) and the robot replans
  (see [v0.5.1](#v051-demo-video-and-wall-robustness-experiment) and
  [`docs/research/v0.5.1-demo-brief.md`](docs/research/v0.5.1-demo-brief.md)).
- **v0.5.2:** collision-checker correctness: MuJoCo's box-box margin contacts are unreliable,
  so the fingertip boxes are now checked against box obstacles with `mj_geomDistance`, after an
  audit of every robot/obstacle pair type (see [v0.5.2](#v052-collision-checker-correctness) and
  [`docs/research/v0.5.2-results.md`](docs/research/v0.5.2-results.md)).
- **v0.6:** known-world *physical* pick-and-place: the Panda picks up a free box through
  simulated contact and friction, carries it (around an obstacle) and places it within 10 mm,
  driven by an event-based supervisor with measured success checks and negative controls; an
  audit of the torque MPC's QP; and a payload study comparing CTC and torque MPC with and
  without the disturbance observer at three unmodelled object masses (see
  [v0.6](#v06-known-world-pick-and-place-and-payload-study) and
  [`docs/research/v0.6-results.md`](docs/research/v0.6-results.md)).

- **v0.7:** an interactive local **workbench**: a browser UI served by Python (offline, no new
  dependencies) with a live MuJoCo viewport, run controls, a command console, scene editing, recording and
  replay, inspectors that tie the motion, the torque-MPC decision, the dynamics and the planning to recorded
  values, a one-joint MPC sandbox and guided experiments (see
  [v0.7](#v07-interactive-workbench) and [`docs/workbench-guide.md`](docs/workbench-guide.md)).

All motion is produced by MuJoCo physics: the controller sends commands to the
Panda's actuators and `mj_step` integrates the dynamics. Joint positions are
written directly only once, to set the initial state.

## Setup (Windows PowerShell)

Requires Python 3.13 (`py -0p` lists installed versions). From the project folder:

```powershell
cd C:\Users\rkhmh\robotics-lab
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands below call `.\.venv\Scripts\python.exe` directly, so you don't need
to activate the environment. If you prefer to activate it, run
`.\.venv\Scripts\Activate.ps1` and then use plain `python`. If PowerShell blocks
the script, first run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Run the simulation

```powershell
.\.venv\Scripts\python.exe main.py
```

A MuJoCo window opens. The arm cycles home → left → right → home, taking 3 s
per move and holding each pose for 1 s. The red sphere marks where the
fingertips are heading. Close the window to exit.

Choose the controller (default: `controller.type` in `configs/default.yaml`, which is PID):

```powershell
.\.venv\Scripts\python.exe main.py --controller pid                    # v0.1 PID (position servos)
.\.venv\Scripts\python.exe main.py --controller mpc                    # setpoint MPC, 50 Hz (position servos)
.\.venv\Scripts\python.exe main.py --controller mpc --control-rate 100 # setpoint MPC at 100 Hz
.\.venv\Scripts\python.exe main.py --controller ctc                    # computed-torque control (torque motors)
.\.venv\Scripts\python.exe main.py --controller torque_mpc             # torque MPC (torque motors)
```

Reach Cartesian targets instead of joint waypoints: see [v0.3](#v03-torque-control-torque-mpc-and-cartesian-reaching).

Other options:

```powershell
.\.venv\Scripts\python.exe main.py --duration 20                      # stop after 20 simulated seconds
.\.venv\Scripts\python.exe main.py --environment empty                 # robot on the floor, no table
.\.venv\Scripts\python.exe main.py --headless --duration 12            # no window; print a tracking summary
.\.venv\Scripts\python.exe main.py --headless --controller mpc         # ... plus MPC solver statistics
.\.venv\Scripts\python.exe main.py --config configs\default.yaml       # explicit config file
```

Viewer controls (MuJoCo built-in): left-drag rotates the view, right-drag pans,
and the scroll wheel zooms. Space pauses. Double-click a body to select it, then
Ctrl + right-drag pushes it with a force; the controller will pull it back. Tab
and Shift+Tab toggle the side panels.

## Compare controllers (headless benchmark)

```powershell
.\.venv\Scripts\python.exe benchmark.py                                     # everything (~15 min)
.\.venv\Scripts\python.exe benchmark.py --controllers pid torque_mpc --scenarios demo push
```

Each controller variant runs each scenario from the same initial state, with
the same robot model, environment and reference trajectory
(`configs/benchmark.yaml`). Results go to `results/benchmark_<timestamp>/`:
`results.json` (all metrics, settings and library versions), `results.csv`, and
`tracking_error.png` / `summary.png` (if matplotlib is installed). All metrics
are measured on the simulated robot at every physics step.

Scenarios: `demo` (the v0.1 motion), `fast` (1 s moves; the reference exceeds
the joint speed limits), `wrist` (large moves on the 12 Nm wrist joints; the
reference also exceeds their speed limits), `near_limits` (poses 0.08-0.10 rad
from joint limits, long base swings), and `push` (hold still while a 30 N
sideways force pushes the hand for 1 s). v0.3 adds `reach` (Cartesian targets
through IK, one out of reach) and `demo_mismatch` (model-based controllers get
robot masses and inertias 15% too low), plus the torque controllers. Each result
records its actuator model. The table prints joint and end-effector errors, peak
speed relative to the limit, saturation, solver timing and fallbacks. v0.4 adds
`fast_feasible`, `wrist_feasible` and `wrist_feasible_dsc` (the same poses with
feasibility-checked timing, identical for every controller), the controllers
`torque_ctc_dob` and `torque_mpc` (v0.4; `torque_mpc_v03` is the v0.3 one), orientation
error, reference feasibility, predicted vs actual violations, and the plots
`velocity_ratio.png` and `orientation.png`. `configs/benchmark_robustness.yaml` sweeps model
mass errors (-25% to +25%), push directions and sensor noise for the torque controllers.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v                  # 189 tests, ~3 min
.\.venv\Scripts\python.exe -m pytest tests\test_planning.py tests\test_planning_demo.py tests\test_planning_benchmark.py -v   # v0.5 planner tests (~1 min)
```

## v0.7: interactive workbench

User guide and 20-minute learning path: [`docs/workbench-guide.md`](docs/workbench-guide.md). Report:
[`docs/research/v0.7-results.md`](docs/research/v0.7-results.md).

```powershell
.\.venv\Scripts\python.exe workbench.py                      # http://127.0.0.1:8765 (or double-click Workbench.bat)
.\.venv\Scripts\python.exe -m pytest tests\test_workbench.py tests\test_workbench_math.py -v
.\.venv\Scripts\python.exe tools\capture_workbench.py        # screenshots + UI demo (headless Chrome/Edge)
```

![Workbench: MPC inspector on a recorded moment](docs/research/figures/v0.7-05_replay_mpc_inspector.jpg)

- One backend: the UI, the console and the Python API (`workbench.RunRequest`, `run_to_end`) build the same
  validated request, which resolves to the CLI configs and runs through `SimulationRunner` / `PickPlaceEpisode`.
- One worker thread owns the simulation and the offscreen renderer; the HTTP threads only queue actions and
  read copies. Display rate and quality never change the physics timestep.
- Every run is recorded (10 ms samples + every MPC update) and saved; replay shows recorded data only.

## v0.6: known-world pick-and-place and payload study

Report: [`docs/research/v0.6-results.md`](docs/research/v0.6-results.md) (progress log:
`docs/research/v0.6-progress.md`).

```powershell
.\.venv\Scripts\python.exe pick_place.py                                          # 5 development cases, CTC + observer
.\.venv\Scripts\python.exe pick_place.py --controller torque_mpc --case dev_obstacle_1   # around the obstacle, torque MPC
.\.venv\Scripts\python.exe pick_place.py --case dev_free_1 --observer off --mass 1.15    # unmodelled steel block, no observer
.\.venv\Scripts\python.exe pick_place.py --case dev_free_1 --fault weak_grip             # negative control: rejected at lift
.\.venv\Scripts\python.exe pick_place_study.py                                    # payload study: 4 configs x 3 masses x 10 cases
.\.venv\Scripts\python.exe pick_place_study.py --sensitivity                      # timestep / contact-parameter check
.\.venv\Scripts\python.exe tools\audit_torque_mpc.py                              # torque-MPC QP audit
.\.venv\Scripts\python.exe tools\render_pick_place_video.py                       # -> results\video\v0.6_pick_place.webp
.\.venv\Scripts\python.exe -m pytest tests\test_pick_place.py tests\test_mpc_audit.py -v
```

- Known world: the object pose and obstacles come from the simulator; there is no perception. The box
  moves only through simulated contact and friction (no weld, no pose writes). A rigid attachment exists
  only in the planner's private collision model, to check the carried object's geometry.
- Phases (approach, descend, close, lift, hold, transport, lower, release, retreat, verify) end on measured
  conditions with timeouts and failure reasons; contacts are classified at every physics step.
- The controllers' model has contacts disabled, so the payload is never visible to them (tested); only the
  disturbance observer can estimate it, from measured joint motion.
- Payload study: 4 configurations × 3 unmodelled masses (0.15 / 0.40 / 1.15 kg) × 10 frozen cases. Every
  configuration completed 30/30 with 0 forbidden contacts. Without the observer the carrying error grows
  with mass (CTC median 4.8 mm, MPC 2.8 mm at 1.15 kg) and the clearance falls to 14.8 mm (CTC). With it,
  both stay within 0.34 mm and ≥ 20 mm. Torque MPC + observer tracks tightest; CTC + observer is
  competitive at about half the wall time per episode.
- Caveat: holding the 1.15 kg block depends on the contact solver (elliptic cone, impratio 10). With
  MuJoCo's default pyramidal cone it slips out (0/6 in the sensitivity check).

## v0.5.2: collision-checker correctness

Report: [`docs/research/v0.5.2-results.md`](docs/research/v0.5.2-results.md).

```powershell
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml --headless   # corrected wall demo
.\.venv\Scripts\python.exe tools\render_demo_video.py        # -> results\video\v0.5.2_demo.webp (v0.5.1 video kept)
.\.venv\Scripts\python.exe tools\audit_clearance.py          # MuJoCo distance queries vs certified distances, per pair type
.\.venv\Scripts\python.exe tools\profile_collision.py        # checker cost and correctness, v0.5 vs v0.5.2
.\.venv\Scripts\python.exe -m pytest tests\test_clearance.py -v
```

- MuJoCo 3.16 misses box-box contacts inside the margin (as close as 1.85 mm under a 20 mm
  margin) and misreports their distance by up to 7 mm. All other pair types in the scenes, and
  `mj_geomDistance`, were correct on 44,000 audited poses.
- The checker (`planning.exact_box_box: true`) asks `mj_geomDistance` for box-box robot/obstacle
  pairs that a conservative AABB bound cannot rule out. `false` reproduces v0.5 for comparisons.
- Effect: with the v0.5 checker, 3 of 113 benchmark plans came within 18.5-19.9 mm; now every
  plan keeps >= 20 mm at its samples (>= 19.998 mm densely in between). Success rates and execution
  accuracy are unchanged. Median planning time rises 6-46% in the obstacle scenes.

## v0.5.1: demo video and wall-robustness experiment

Research brief: [`docs/research/v0.5.1-demo-brief.md`](docs/research/v0.5.1-demo-brief.md).

```powershell
.\.venv\Scripts\python.exe tools\render_demo_video.py                    # demo video -> results\video\v0.5.2_demo.webp (the 31 s v0.5.1 render is kept)
.\.venv\Scripts\python.exe tools\render_demo_video.py --include-unsafe   # adds the unsafe baseline (checks off, arm hits the wall)
.\.venv\Scripts\python.exe robustness_experiment.py                      # 6 wall variants x 5 seeds, torque MPC + CTC -> results\robustness_<timestamp>
.\.venv\Scripts\python.exe robustness_experiment.py --no-execution       # planning + old-plan checks only
```

The video is an animated WebP (opens in any browser) rendered offscreen with MuJoCo and
written by Pillow; no ffmpeg is needed. `configs/robustness_wall.yaml` defines the wall
variants (changed in code through `environment_overrides`, the XML stays untouched).

## v0.5: collision-aware motion planning

Full report: [`docs/research/v0.5-results.md`](docs/research/v0.5-results.md); development log:
[`docs/research/v0.5-worklog.md`](docs/research/v0.5-worklog.md).

**Watch the arm go around an obstacle** (MuJoCo viewer, torque MPC):

```powershell
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml
```

The arm starts left of a blue wall and reaches a target on the right, then returns. The
straight joint-space move would sweep the hand through the wall. The planner finds a route
over it before the simulation starts (~0.25 s). The green line is the end-effector path the
arm follows, the faint orange line is the raw RRT-Connect path before shortcutting, and the
red sphere marks the current target. The terminal prints, per target, whether the direct
edge was blocked, the planning time, nodes, collision checks and path length.

Compare with the same goals without the planner:

```powershell
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml --no-planning                     # blocked target rejected, arm stays put
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml --no-planning --unchecked-direct  # UNSAFE comparison: drives into the wall
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml --controller ctc                  # computed-torque control instead
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml --headless                        # summary incl. contact steps
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml --plan-seed 3                     # a different random route
```

Planning works for any config: `task.planning.enabled: true` (or `--planning`) on a
`cartesian_reach` or `joint_target` task. Free moves are unchanged; blocked moves follow the
planned path, stopping briefly at each via point. A target that cannot be planned is rejected
with a reason (unreachable, no collision-free IK posture, invalid start, timeout, node budget,
no path). A joint-waypoint move that cannot be planned stops with `Motion planning failed`.
The direct move is never used as a substitute.

How it works (`planning/`):

- **Collision checking** on a private model copy. The robot keeps 20 mm from the environment
  (MuJoCo contact margins), and edges are sampled so no point of the robot moves more than
  10 mm between checks. That certifies at least 15 mm everywhere on a planned edge. This is
  discrete checking with a clearance argument, not continuous collision detection;
  self-collision is checked at samples only.
- **RRT-Connect** in joint space with a lever-arm-weighted metric. Time, node and iteration
  budgets each end the search with their own status.
- **Multiple IK postures** for Cartesian targets: from the current posture, the rest posture,
  nearby perturbations and random seeds. A posture is used only if IK converged and it is
  collision-free with clearance.
- **Shortcut smoothing**, collision-checked, keeping the raw path as a fallback.

Results (20 seeds per scenario, all reported):

| scenario | planning success | median time | path raw → smoothed (rad) |
|---|---|---|---|
| wall (demo) | 20/20 | 0.26 s | 3.72 → 1.57 |
| obstacles scene, blocked pair | 20/20 | 0.19 s | 3.06 → 1.12 |
| narrow passage (wall under a ceiling) | 20/20 | 1.38 s | 10.27 → 5.59 |
| unreachable / start in collision / goal in collision | 0/20 each, with the correct reason | | |

In MuJoCo, all 24 planned executions (torque MPC and CTC, 3 seeds, 4 scenes) reached the
target within 0.1 mm, with 0 contact steps and at least 16.2 mm measured clearance. The
unchecked direct moves collided in all 8 runs (1005-1581 contact steps).

```powershell
.\.venv\Scripts\python.exe planning_benchmark.py --no-execution                    # planning statistics, 11 scenarios x 20 seeds
.\.venv\Scripts\python.exe planning_benchmark.py --execution-only --scenarios wall  # closed-loop execution of one scenario
.\.venv\Scripts\python.exe planning_benchmark.py                                     # both (~10 min; needs free memory)
```

Results go to `results/planning_<timestamp>/` (`results.json`, `planning_runs.csv`,
`planning_summary.csv`, `execution_runs.csv`, `summary.md`, plots).

## v0.4: feasible trajectories, constraint diagnostics, disturbance rejection

Full report: [`docs/research/v0.4-results.md`](docs/research/v0.4-results.md); development log:
[`docs/research/v0.4-worklog.md`](docs/research/v0.4-worklog.md).

**Why torque MPC "lost" to CTC on `fast`/`wrist` in v0.3:** those reference trajectories ask
joint 7 for 1.43x its speed limit. CTC follows them and breaks the limit; the MPC enforces it
and therefore must lag (no speed-limited controller can do better than ~147 / ~304 mrad there).
Without the speed limit, or with feasible timing, the torque MPC tracks to < 0.05 mrad.

**Feasibility-checked trajectories** (`tasks/trajectory.py`, `task.trajectory` in the configs):
each move takes at least `move_time` and is lengthened automatically to respect 90% of the
datasheet speed limits, 50% of the acceleration limits, a jerk limit and an 80% torque budget
(inverse dynamics). `main.py` prints which moves were lengthened and why; impossible moves are
reported (`Infeasible trajectory: joint 2 needs 35.2 Nm just to hold against gravity ...`).
Profiles: `min_jerk` (default) or `double_s` (jerk-limited, cruises at the speed limit). Remove
the `trajectory` section to get the unchecked v0.3 timing.

**Disturbance observer / offset-free torque MPC** (`controllers/disturbance.py`): estimates the
unknown joint torque (pushes, payload, wrong masses) from measured motion and the applied torque.
On by default for `torque_mpc`, optional for `ctc` (`ctc.disturbance_observer.enabled: true`).
The torque MPC also keeps 0.5% below the speed limit (`velocity_margin`), which removed every
actual speed-limit violation.

| max joint error (mrad) | CTC (v0.3) | CTC + observer | torque MPC v0.3 | torque MPC v0.4 |
|---|---|---|---|---|
| wrist poses, feasible timing | 0.16 | 0.16 | 0.0053 | 0.0053 |
| 30 N push on the hand | 35.2 | **3.14** | 12.1 | 3.62 |
| model masses 25% too low | 9.09 | **0.84** | 3.44 | 0.95 |
| model masses 25% too high | 6.20 | **0.54** | 2.61 | 0.69 |
| `fast` (infeasible reference): peak speed / limit | 1.43x | 1.43x | 1.0006x | **0.9955x** |

CTC with the observer rejects disturbances slightly better than the offset-free MPC; the MPC
tracks nominal motion more accurately and respects the speed limits. The benchmark now also
reports reference feasibility, predicted vs actual constraint violations (size, duration,
integral), solver slack, and end-effector orientation error.

```powershell
.\.venv\Scripts\python.exe main.py --controller torque_mpc                              # v0.4 torque MPC
.\.venv\Scripts\python.exe benchmark.py --config configs\benchmark_robustness.yaml     # mass-error / push / noise sweep
.\.venv\Scripts\python.exe tools\diagnose_torque_mpc.py baseline no_vlimit slow        # the fast/wrist investigation
.\.venv\Scripts\python.exe tools\profile_mpc.py                                        # MPC timing at 50/100 Hz
```

## v0.3: torque control, torque MPC and Cartesian reaching

Full report with equations, methodology and all measurements:
[`docs/research/v0.3-results.md`](docs/research/v0.3-results.md). Development log:
[`docs/research/v0.3-worklog.md`](docs/research/v0.3-worklog.md).

**Controllers and the actuators they need** (the engine builds the matching Panda automatically):

| `--controller` | config `type` | actuators | what it is |
|---|---|---|---|
| `pid` | `joint_pid` | position servos (Menagerie) | v0.1 PID outer loop, 500 Hz |
| `mpc` | `joint_mpc` | position servos | v0.2 setpoint MPC, 50 Hz (v0.3 fixes its fallback recovery) |
| `ctc` | `torque_ctc` | torque motors | computed-torque control (MuJoCo inverse dynamics + PD), 500 Hz |
| `torque_mpc` | `torque_mpc` | torque motors | torque MPC (OSQP), 50 Hz, 0.5 s horizon, hard torque limits |

The torque motors are built from the same Menagerie model in code
(`Panda(actuation="torque")`): same masses, inertias, damping, armature and joint
limits; commands are joint torques limited to 87 Nm (joints 1-4) and 12 Nm (5-7).

**Cartesian reaching.** Give end-effector targets (x, y, z) in metres; the robot
solves inverse kinematics (damped least squares on MuJoCo's Jacobian, joint-limit
aware), checks for collisions at the goal and along the joint path, and moves
there smoothly. Unreachable or colliding targets are reported and skipped. The
robot base is at (0, 0, 0.40) on the table top.

```powershell
.\.venv\Scripts\python.exe main.py --config configs\reach.yaml --controller torque_mpc   # 3 valid + 2 rejected targets
.\.venv\Scripts\python.exe main.py --target 0.5 0.2 0.7 --target 0.3 -0.4 0.9           # ad-hoc targets (repeatable)
.\.venv\Scripts\python.exe main.py --config configs\reach.yaml --controller pid --headless
```

Edit the targets in `configs/reach.yaml` (no code changes needed). Collisions are
*checked*, not avoided: there is no planner that goes around obstacles yet.

**Headline results** (max joint error, mrad; full tables in the report):

| scenario | pid | mpc | ctc | torque_mpc |
|---|---|---|---|---|
| demo | 3.42 | 7.83 | 0.063 | **0.0028** |
| reach (IK targets) | 3.42 | 0.079 | 0.031 | **0.0022** |
| model mismatch (masses -15%) | 3.42 | 20.8 | 4.98 | **1.91** |
| 30 N push | 3.44 | **2.02** | 35.2 | 12.1 |
| fast (over-speed reference) | **2.98** (1.43x speed limit) | 300 (1.06x) | 0.54 (1.43x) | 169 (**1.0006x**) |

`pid`/`mpc` and `ctc`/`torque_mpc` use different actuators, so compare within
each pair. The model-based controllers use the exact plant model, except in the
model-mismatch row. Torque control has no integral action, so it gives way
under the push far more than the stiff servos.


## MPC design (v0.2)

`controllers/mpc.py` solves a finite-horizon quadratic program with OSQP at
every control update (default 50 Hz, horizon 15 = 0.3 s), using the latest
measured joint state. It applies the first command, holds it until the next
update, and repeats. Between updates the physics runs at 500 Hz.

**What is optimized.** The Panda's arm actuators are MuJoCo *position servos*:
the command is a joint-angle setpoint `u` (rad), and MuJoCo produces the torque
`tau = kp (u - q) - kd qd` (kd/kp = 0.1), clipped to 87 Nm (joints 1-4) or 12
Nm (joints 5-7). The MPC optimizes the setpoints, not torques. Torque appears
in the cost and constraints through that servo law.

**Prediction model** (`controllers/prediction.py`). The model is linearized
from MuJoCo at every update: one-step Jacobians from `mjd_transitionFD`,
composed over the 10 physics steps of a control interval, plus a drift term
from a real MuJoCo rollout, so the model is exact at the current state. It runs
on private copies of the simulation data and never touches the live state. It
also never sees applied external forces, which a real controller couldn't
measure. Validation (`tests/test_mpc.py`): with torques inside the force limits,
predictions match MuJoCo to ~1e-6 rad over 0.4 s.

**Cost:** joint-position and joint-velocity tracking (with reference preview
over the horizon), effort (sustained servo torque beyond gravity compensation,
normalized by the torque limit), and setpoint change between updates.

**Constraints:**

| constraint | type | notes |
|---|---|---|
| command (setpoint) limits = actuator `ctrlrange` | hard | joint-angle units |
| setpoint slew: abs(u[k] - u[k-1]) <= v_max * Ts | hard | keeps the setpoint from outrunning the joint speed limit |
| joint position limits minus 0.02 rad margin | soft | |
| joint velocity limits (Franka datasheet) | soft | not in the MuJoCo model; from `robots/panda.py` |
| sustained torque <= 95% of force range | soft | torque at the end of each interval |

"Soft" constraints use slack variables with an exact-penalty weight, so the QP
stays feasible from any measured state (set `soft_constraints: false` for hard
constraints). Every weight, limit, rate and solver setting is in
`configs/default.yaml` under `controller.mpc`.

**Fallback.** If an update fails (OSQP not solved, iteration budget exceeded,
non-finite or implausible model data, any exception), the controller hands
over to the v0.1 PID until the next successful update. The PID runs in the
background every step, so it is warm when it takes over. The OSQP iteration
budget (`max_iter: 500`, about 16 ms) bounds the solve time deterministically.
Benchmarks stay reproducible because no wall-clock timeout is involved.

**Physical limits not enforced predictively:** the ~1 ms torque spike at each
setpoint step (MuJoCo clips it; see below), torque *between* interval ends,
joint acceleration and jerk limits (Franka datasheet), self-collision and
collision with the environment, and finger dynamics (frozen).

## Benchmark results (v0.2, as published with v0.2)

Measured on 2026-10-17 (Windows 11, Intel CPU, Python 3.13.7, MuJoCo 3.16.0,
OSQP 1.1.3) with `benchmark.py` at default settings. Errors are in mrad
(1 mrad = 0.001 rad); solve = OSQP time per update; update = full MPC update
(linearization + QP); overruns = updates slower than the control period;
v-viol = physics steps above a joint speed limit; sat = share of physics steps
with an arm actuator at its force limit. No scenario had joint-limit
violations, contacts or NaNs.

| scenario | controller | RMS err | max err | RMS torque (Nm) | sat | v-viol | solve mean / p95 (ms) | update p95 (ms) | overruns | fallbacks / updates |
|---|---|---|---|---|---|---|---|---|---|---|
| demo | PID | **0.27** | **3.4** | 15.7 | 0.00 | 0 | - | - | - | - |
| demo | MPC 50 Hz | 1.06 | 7.8 | 16.4 | 0.21 | 0 | 2.4 / 3.1 | 4.5 | 0 | 1 / 600 |
| demo | MPC 100 Hz | 0.66 | 6.4 | 16.2 | 0.07 | 0 | 2.4 / 3.1 | 4.3 | 2 | 0 / 1200 |
| fast | PID | **0.65** | **3.0** | 16.2 | 0.00 | 354 (1.43x v_max) | - | - | - | - |
| fast | MPC 50 Hz | 25.2 | 300 | 18.4 | 0.13 | **194 (1.06x)** | 3.5 / 7.2 | 8.7 | 0 | 0 / 300 |
| wrist | PID | **0.34** | **3.4** | 13.8 | 0.00 | 430 | - | - | - | - |
| wrist | MPC 50 Hz | 1.67 | 24.9 | 14.9 | 0.10 | 473 | 11.6 / 17.7 | 19.0 | 5 | **275 / 450** |
| near_limits | PID | **0.25** | **3.4** | 12.0 | 0.00 | 0 | - | - | - | - |
| near_limits | MPC 50 Hz | 1.34 | 10.8 | 15.4 | 0.23 | 52 | 2.5 / 3.2 | 4.5 | 0 | 0 / 900 |
| push | PID | 0.63 | 3.4 | 14.6 | 0.00 | 0 | - | - | - | - |
| push | MPC 50 Hz | **0.51** | **2.0** | 14.6 | 0.00 | 0 | 2.4 / 3.0 | 4.4 | 0 | 0 / 200 |

All 20 controller/scenario combinations, including `mpc_100hz` and
`mpc_no_preview` everywhere, are in `results.csv` from a benchmark run.

How to read it:

- **PID tracks moving references more tightly** in every motion scenario. MPC
  at 50 Hz has 3-8x the max error. The cause is structural: the servo setpoint
  changes in 20 ms steps, and the stiff servos turn each step into a ~1 ms
  torque spike that MuJoCo clips (21% of physics steps in `demo`). The linear
  model doesn't capture that, so MPC lags slightly behind its own plan.
  100 Hz roughly halves the clipping and lowers the error.
- **MPC is better at rest and against disturbances.** During every hold its
  error falls to ~1e-10 mrad, against ~1e-3 mrad for PID's slowly decaying
  integrator (see `tracking_error.png`). It also has no start-up sag. In `push`,
  MPC's max error is 2.0 vs 3.4 mrad, and it is back at the setpoint ~0.7 s
  after the push ends, while PID takes several seconds.
- **MPC respects speed limits that PID ignores.** In `fast`, the reference
  needs 1.4x the joint speed limits. PID follows it, reaching 1.43x v_max. MPC
  keeps to 1.06x v_max and accepts a large tracking error (0.3 rad) to do so.
  That is the intended trade-off, not a tracking failure.
- **The wrist scenario is MPC's main failure mode.** The reference exceeds the
  wrist speed limits, and catching up saturates several 12 Nm and 87 Nm
  actuators at once. Those QPs hit the iteration budget, so 61% of updates fall
  back to PID. The arm stays safe and on track (PID takes over), but this is
  not a successful MPC run.
- **Timing:** typical updates take 3.5-4.5 ms of a 20 ms budget (10 ms at 100
  Hz). Real-time factor is ~4.5x at 50 Hz, versus ~30x for PID. MPC with
  reference preview beat the no-preview variant on max error in `demo` (7.8 vs
  15.2 mrad) and was about equal elsewhere.

## MPC design notes (what was tried)

These measured alternatives were rejected. They are recorded so they aren't
retried blindly.

- **Torque limit at the start of each interval:** physically valid, but the
  setpoint-step spike then caps the wrist at 0.29 rad/s. Replaced by the
  sustained (end-of-interval) torque.
- **Linearizing through the force clip at the next planned command:** more
  accurate in gentle motion (max error 5.3 vs 7.8 mrad at 50 Hz, 2.7 mrad at
  100 Hz). The clipped linearization is locally unstable under gravity,
  however, and the closed loop diverged in `fast` (10x v_max). A
  first-interval-only variant also diverged.
- **Offset-free disturbance estimation** (full-state or input disturbance):
  worse tracking in every test (e.g. 20 vs 7.8 mrad), so it ships disabled
  (`disturbance_gain: 0`).
- **Position weight 1e5:** the loop went unstable (model mismatch exceeds the
  stability margin).
- **Composing the full 18-state Jacobian:** the finger joints sit on their
  joint limit, and finite differences across it produced a spectral radius of
  1e22. Only the arm block is composed now. At the actuator force-clip boundary
  the one-step map is discontinuous under `implicitfast`. There the model uses
  Jacobians of a force-unlimited model copy, and any remaining implausible model
  triggers the fallback.

## Known limitations and next steps

v0.5 status (details in `docs/research/v0.5-results.md`):

- Collision checking is discrete. Robot-environment clearance is certified to 15 mm along
  planned edges (for MuJoCo's convex-hull geometry, given exact per-sample distances; since
  v0.5.2 these are audited, not proven); self-collision is checked at samples only.
- The world is static: no moving obstacles and no replanning during execution. The
  controllers do not know about obstacles, so safety relies on tracking the planned reference.
- RRT-Connect is probabilistically complete only. A timeout does not prove that no path exists.
- Paths stop at every via point (rest to rest). Blending them into one smooth trajectory, with
  the blended curve collision-checked, is the obvious next step. Orientation is not planned.

v0.4 status (details in `docs/research/v0.4-results.md`):

- Both MPCs are linear MPC (one linearization per update), not nonlinear MPC. The
  open-loop prediction error reaches ~2-5e-2 rad at the end of the 0.5 s horizon, so
  late-horizon constraint predictions are approximate; soft constraints give no guarantee.
- The disturbance observer differentiates measured velocity; it was tested only with
  simulated Gaussian noise. With noise the torque MPC is more noise-sensitive than CTC.
- 100 Hz torque MPC with a 0.5 s horizon does not fit its 10 ms budget in Python/OSQP;
  50 Hz does (p99 ~10 ms). Wall-clock timing varies 2-3x between runs.
- Model mismatch was tested as uniform mass/inertia scaling only.

v0.3 status:

- (Fixed in v0.4 by the disturbance observer, see above.) The model-based controllers were
  only as good as their model: with masses 15% off, torque MPC went from 2.8 urad to 1.9 mrad
  and computed torque to 5 mrad; without an observer they deflect 12-35 mrad under a 30 N push.
- (Addressed in v0.5 by motion planning.) Reaching used position-only IK with straight
  joint-space motion; obstacles were detected, not avoided.
- The v0.2 servo MPC still saturates on 13-25% of physics steps at speed and
  falls back on 43 of 450 wrist updates (275 before the v0.3 fix).

The v0.2 notes below remain accurate for the servo MPC:

- MPC tracks moving references less tightly than the well-tuned PID on these
  servos (see results). Next steps: run MPC at 100 Hz by default once timing
  margins are confirmed. Or drive the arm with torque actuators (done in
  v0.3), which removes the setpoint-step spikes and makes the linear model
  far more accurate.
- Step references and large over-speed references need ~700-1800 OSQP
  iterations, more than the real-time budget, so they fall back to PID. A
  condensed QP formulation, or an active-set or interior-point QP solver, would
  be worth benchmarking.
- Constraints not enforced predictively: setpoint-step torque spikes, torque
  between interval ends, acceleration and jerk limits, collisions. Torque
  limits are still enforced physically by MuJoCo's force clip.
- The model linearizes once per update around the current state (gravity and
  inertia vary only to first order across the 0.3 s horizon), and the fingers
  are frozen.
- Wall-clock timing is measured in Python on a desktop OS and includes
  occasional scheduler hiccups. Headless runs are not real-time constrained;
  the viewer runs in real time.

## Project layout

```
main.py                  CLI entry point (viewer or headless)
configs/default.yaml     environment, waypoints, timing, PID gains, MPC settings
configs/benchmark.yaml   benchmark scenarios and controller variants
configs/reach.yaml       Cartesian reaching targets (extends default.yaml)
configs/benchmark_robustness.yaml  v0.4 mass-error / disturbance / noise sweep
configs/planning_demo.yaml         v0.5 obstacle-avoidance demo (wall scene, torque MPC, planning on)
configs/benchmark_planning.yaml    v0.5 planning benchmark scenarios
configs/robustness_wall.yaml       v0.5.1 wall variants for the replanning experiment
configs/pick_place.yaml            v0.6 pick-and-place: gripper, contact solver, phases, criteria, development cases
configs/pick_place_study.yaml      v0.6 payload study (frozen evaluation cases, masses, configurations)
benchmark.py             CLI: headless controller comparison
planning_benchmark.py    CLI: motion-planning benchmark (planning statistics + execution)
robustness_experiment.py CLI: replanning under changed (known) wall geometry (v0.5.1)
pick_place.py            CLI: known-world pick-and-place episodes, negative controls (v0.6)
pick_place_study.py      CLI: payload study and sensitivity check, resumable JSONL output (v0.6)
environments/*.xml       scenes (table, empty, obstacles, planning_demo, planning_narrow, pick_place, pick_place_obstacle) with a robot_mount site and target marker
planning/                v0.5 motion planning: collision checker, RRT-Connect, shortcut, IK candidates, planner facade, viewer markers
robots/panda.py          Panda description and model loading; position-servo or torque-motor variant
robots/models/franka_emika_panda/   official MuJoCo Menagerie model (unmodified, see SOURCE.txt)
simulation/engine.py     builds the scene (environment + robot via MjSpec), steps physics, reads state
simulation/runner.py     control loop: task -> controller -> actuators -> mj_step; viewer and headless modes
controllers/base.py      controller interface
controllers/pid.py       joint-space PID outer loop on the Panda's position servos
controllers/prediction.py  MuJoCo-linearized prediction model for the arm
controllers/mpc.py       receding-horizon setpoint MPC (OSQP) with PID fallback
controllers/torque_mpc.py  torque MPC (OSQP) with computed-torque fallback
controllers/computed_torque.py  computed-torque baseline controller
controllers/dynamics.py  MuJoCo inverse dynamics on scratch data
controllers/disturbance.py  joint-torque disturbance observer (v0.4)
simulation/kinematics.py FK, Jacobian, damped least-squares IK, collision checks
simulation/benchmark.py  benchmark scenarios, metrics, CSV/JSON/plots
tasks/joint_target.py    joint-waypoint task (fixed or feasibility-checked timing)
tasks/trajectory.py      min-jerk / double-S timing, time scaling, feasibility checks (v0.4)
tasks/reach.py           Cartesian (x, y, z) target reaching via IK (optionally through the planner)
tasks/planned_joint.py   joint-waypoint task whose moves follow planned paths (v0.5)
simulation/planning_benchmark.py  planning benchmark scenarios, metrics, CSV/JSON/plots
simulation/robustness.py wall-variant episodes: old-plan re-check, replanning, execution (v0.5.1)
simulation/manipulation.py  pick-and-place supervisor, contact monitor, verifier (v0.6)
tasks/pick_place.py      planned-path and straight Cartesian-line references for manipulation (v0.6)
tests/                   test_simulation (v0.1), test_mpc (v0.2), test_torque, test_reach, test_benchmark_v03, test_trajectory, test_v04, test_planning* (v0.5), test_robustness (v0.5.1), test_clearance (v0.5.2), test_mpc_audit, test_pick_place (v0.6)
tools/                   MPC audit, diagnostics, profiling, timing, sweep and plotting scripts (plot_v05.py: v0.5 figures; render_demo_video.py: demo video; audit_clearance.py, profile_collision.py: v0.5.2; audit_torque_mpc.py, render_pick_place_video.py: v0.6)
docs/research/           v0.3, v0.4, v0.5, v0.5.2, v0.6 reports, v0.5.1 brief, worklogs, benchmark CSVs and figures
```

`tasks/pick_place.py` (empty until v0.5.2) now holds the v0.6 manipulation references.

## Model source and license

The Panda model comes from
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`franka_emika_panda/`, commit `d7b481b`). It is licensed under Apache-2.0; see
`robots/models/franka_emika_panda/LICENSE`.
