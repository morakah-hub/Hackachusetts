# CLAUDE.md

Guidance for working in this repository.

## What this is

A modular robotics simulation platform: MuJoCo 3.16 + Python 3.13 on Windows.
A Franka Panda moves through joint waypoints or Cartesian targets under
closed-loop control: v0.1 joint PID; v0.2 joint-space MPC (OSQP) on the
position servos; v0.3 a torque-actuated Panda variant with computed-torque
control and torque MPC, Cartesian reaching (IK), and a four-way benchmark;
v0.4 feasibility-checked trajectory generation, constraint diagnostics,
offset-free torque MPC with a disturbance observer, and orientation metrics;
v0.5 collision-aware motion planning (joint-space RRT-Connect, clearance-checked
edges, shortcutting, multiple IK postures) executed through the v0.4 trajectory
generator by the existing controllers; v0.5.1 a rendered demo video and a
wall-geometry replanning experiment; v0.5.2 a collision-checker correctness fix
(exact distances for box-box robot/obstacle pairs) and an audit of MuJoCo's
distance queries; v0.6 known-world physical pick-and-place (pose IK, Cartesian
approach/lift lines, an event-based supervisor, phase-dependent collision rules, a
step-resolution contact monitor and verifier) and a payload study (CTC vs torque MPC, with
and without the disturbance observer, three unmodelled object masses); v0.7 a local browser workbench
(`workbench.py`) over the same backends: live viewport, commands, replay, math inspectors, guided
experiments. Perception, reactive
replanning, RL and learning come later. The only custom GUI is the v0.7 workbench (local browser UI over the shared backend); otherwise use
MuJoCo's built-in viewer.
Research reports: `docs/research/v0.3-results.md`, `v0.4-results.md`,
`v0.5-results.md`, `v0.5.1-demo-brief.md`, `v0.5.2-results.md`, `v0.6-results.md`, `v0.7-results.md` (keep old reports unchanged); development logs
`docs/research/v0.*-worklog.md`.

## Commands (PowerShell, from the repo root)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py                               # interactive viewer (controller from config)
.\.venv\Scripts\python.exe main.py --controller mpc              # pid | mpc (servos) | ctc | torque_mpc (torque motors)
.\.venv\Scripts\python.exe main.py --config configs\reach.yaml --controller torque_mpc   # Cartesian reaching
.\.venv\Scripts\python.exe main.py --target 0.5 0.2 0.7          # reach ad-hoc (x, y, z) targets
.\.venv\Scripts\python.exe main.py --headless --duration 12      # headless summary
.\.venv\Scripts\python.exe benchmark.py                          # all controllers x scenarios -> results/ (~12 min)
.\.venv\Scripts\python.exe benchmark.py --config configs\benchmark_robustness.yaml   # mismatch/push/noise sweep
.\.venv\Scripts\python.exe tools\diagnose_torque_mpc.py          # instrumented torque-MPC experiments
.\.venv\Scripts\python.exe tools\profile_mpc.py                  # MPC timing breakdown, 50/100 Hz
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml                 # v0.5 planned obstacle avoidance (torque MPC)
.\.venv\Scripts\python.exe main.py --config configs\planning_demo.yaml --no-planning   # same goals, planning off
.\.venv\Scripts\python.exe planning_benchmark.py                 # planning x seeds + execution -> results/
.\.venv\Scripts\python.exe robustness_experiment.py              # v0.5.1 wall variants x seeds, replan + execute -> results/
.\.venv\Scripts\python.exe tools\render_demo_video.py            # demo video (offscreen, animated WebP) -> results/video/v0.5.2_demo.webp
.\.venv\Scripts\python.exe tools\audit_clearance.py              # v0.5.2 audit of MuJoCo distance queries per geom pair type (~7 min)
.\.venv\Scripts\python.exe tools\profile_collision.py            # v0.5.2 checker cost and correctness, v0.5 vs v0.5.2
.\.venv\Scripts\python.exe tools\audit_torque_mpc.py             # v0.6 torque-MPC QP audit (objective, rows, signs, prediction)
.\.venv\Scripts\python.exe pick_place.py                         # v0.6 pick-and-place development cases (CTC + observer)
.\.venv\Scripts\python.exe pick_place.py --controller torque_mpc --case dev_obstacle_1   # one case, torque MPC
.\.venv\Scripts\python.exe pick_place.py --case dev_free_1 --fault weak_grip            # negative control (rejected)
.\.venv\Scripts\python.exe pick_place_study.py                   # v0.6 payload study, 120 episodes (~10 min) -> results\v0.6\study
.\.venv\Scripts\python.exe pick_place_study.py --sensitivity     # bounded timestep/contact sensitivity check
.\.venv\Scripts\python.exe tools\render_pick_place_video.py      # v0.6 demo video -> results\video\v0.6_pick_place.webp
.\.venv\Scripts\python.exe workbench.py                          # v0.7 local workbench UI (http://127.0.0.1:8765); Workbench.bat
.\.venv\Scripts\python.exe tools\capture_workbench.py            # v0.7 UI screenshots + demo via headless Chrome/Edge
.\.venv\Scripts\python.exe -m pytest tests -v                    # tests (~2.5 min)
```

## Architecture

Data flow, once per physics step (dt = 0.002 s):

```
task.reference(t) -> controller.compute(q, qd, q_ref, qd_ref) -> engine.set_arm_command(ctrl) -> mj_step
```

- **`simulation/engine.py` — `SimulationEngine`.** Builds the scene with
  `mujoco.MjSpec`: it loads `environments/<name>.xml` and attaches the robot
  spec at the `robot_mount` site. It then owns `MjModel`/`MjData`, exposes arm
  state (`arm_qpos`, `arm_qvel`, `ee_position`), clips commands to `ctrlrange`,
  moves the mocap target marker, and reports robot contacts. `reset()` is the
  only place that writes `qpos`.
- **`simulation/runner.py` — `SimulationRunner`.** Wires config → engine, task
  and controller. `step()` runs one control + physics step. `run_headless()`
  returns a `RunLog`. `run_viewer()` uses `mujoco.viewer.launch_passive`,
  paced to wall-clock time.
- **`robots/panda.py` — `Panda`.** Names of joints, actuators and the base
  body, plus the home pose and datasheet speed limits. `load_spec()` loads the
  Menagerie MJCF and adds an `ee_site` between the fingertips.
  `Panda(actuation="torque")` turns actuators 1-7 into torque motors via MjSpec
  (ctrl = Nm, limits 87/12 Nm); everything else is identical.
- **`controllers/` — `Controller` interface; `JointPIDController`;
  `JointMPCController` (`mpc.py`); `MuJoCoPredictionModel` (`prediction.py`).**
  `compute(q, qd, q_ref, qd_ref, dt, t=None)` is called every physics step; the
  MPC re-solves every `substeps` calls and holds its command in between. The
  MPC gets future references only through an optional `reference_preview(t)`
  callable wired by the runner, so it never imports tasks or environments.
- **`controllers/torque_mpc.py` (`TorqueMPCController`), `computed_torque.py`
  (`ComputedTorqueController`), `dynamics.py` (`ArmInverseDynamics`, mj_inverse).**
  Torque-level controllers; CTC is also the torque MPC's fallback.
- **`simulation/runner.py` — `build_controller()`** picks the controller from
  `config["controller"]["type"]` (`joint_pid` | `joint_mpc` | `torque_ctc` |
  `torque_mpc`). The engine is built with the actuation that controller needs
  (`CONTROLLER_ACTUATION`). `load_config` supports `extends:` (task replaced
  wholesale). `config["controller_model"]` gives model-based controllers a
  perturbed model copy (`engine.controller_model`) for mismatch experiments.
- **`simulation/kinematics.py` — `ArmKinematics`.** FK, site Jacobian, DLS IK
  (position only, joint clamping, null-space bias), collision checks at the
  goal and along the joint path. Scratch data only.
- **`simulation/benchmark.py` + `benchmark.py` + `configs/benchmark.yaml`.**
  Every controller variant runs every scenario from the same start; metrics
  come from the simulated robot. Output goes to `results/` (git-ignored).
- **`tasks/joint_target.py` — `JointTargetTask`.** Quintic (minimum-jerk)
  segments between waypoints, with hold periods; it can loop. It validates
  waypoints against the joint limits. `Reference` carries q, qd and qdd.
- **`tasks/trajectory.py` (v0.4).** Straight joint-space moves timed by
  `min_jerk` (quintic) or `double_s` (jerk-limited); `plan_segment` lengthens a
  move to meet scaled velocity/acceleration/jerk/torque limits or raises
  `TrajectoryInfeasible`; `check_reference` reports violations. Tasks use it when
  `task.trajectory` is configured; without it they give the v0.3 unchecked timing.
  Moves have per-segment durations: use `task.segment_bounds(k)`, not
  `segment_time` (which raises when durations differ).
- **`controllers/disturbance.py` (v0.4) — `JointDisturbanceObserver`.** Lumped
  joint-torque disturbance from measured motion and the *applied* torque, via
  discrete-time inverse dynamics (`mjENBL_INVDISCRETE` on a private model copy).
  Used by CTC (`ctc.disturbance_observer`) and the torque MPC (offset-free MPC).
- **`tasks/reach.py` — `CartesianReachTask`.** (x, y, z) targets resolved by an
  injected planner (IK + collision checks); rejected targets are skipped with a
  reason. Without `task.planning` it is collision *checking* only (v0.3); with
  it, the injected planner is `MotionPlanner.plan_to_position` (v0.5).
- **`planning/` (v0.5).** `collision.py` `CollisionChecker`: configuration and
  edge validity on a private model copy (joint limits minus margin, MuJoCo
  contacts with `geom_margin = clearance` on environment geoms, midphase off; box-box
  robot/obstacle pairs via `mj_geomDistance` since v0.5.2);
  edges are sampled so no robot point moves more than `cartesian_step` between
  samples (lever-arm bound). `rrt_connect.py` `RRTConnect` (weighted metric,
  time/node/iteration budgets with distinct statuses). `shortcut.py`. `ik.py`
  `ik_candidates` (current, rest, local and random seeds; only converged *and*
  valid postures). `motion_planner.py` `MotionPlanner.plan_joint` /
  `plan_to_position` -> `MotionPlan` (IKResult-compatible + `path`, `raw_path`,
  `status`, `stats`). `visualization.py` draws planned EE paths into the viewer's
  `user_scn`.
- **Planning integration.** `task.planning.enabled` makes the runner build a
  `MotionPlanner` once, at task creation. `CartesianReachTask` and
  `PlannedJointTask` (`tasks/planned_joint.py`) turn path vertices into via points
  (rest to rest, no hold) via `JointTargetTask(move_times=, hold_times=)`.
  Controllers never see the planner. `PlanningFailed` is raised when a joint
  move cannot be planned. `planning_benchmark.py` + `simulation/planning_benchmark.py`
  + `configs/benchmark_planning.yaml` benchmark planning and execution.
- **`environments/*.xml`.** Each must define a `robot_mount` site and a mocap
  body named `target`, and must use `integrator="implicitfast"` to match the
  Panda (otherwise MjSpec warns about an attach conflict). Contact between the
  robot base and a body named `table` is excluded automatically.

## Key facts about the Panda model

- `robots/models/franka_emika_panda/` is a verbatim copy of the official
  Menagerie model (`SOURCE.txt` records the commit). **Do not edit it.** Make
  changes in code via `MjSpec` instead (see `Panda.load_spec`). Do not use the
  `mujoco-menagerie` PyPI package.
- `actuator1`–`actuator7` are **position servos**: `ctrl` is a joint-angle
  setpoint, and torque = kp·(ctrl − q) − kd·q̇, with kd/kp = 0.1. The force
  limits are 87 Nm (joints 1–4) and 12 Nm (joints 5–7). `actuator8` is the
  gripper: 0 = closed, 255 = open.
- qpos layout: 7 arm joints, then 2 finger joints (nq = nv = 9, nu = 8).
- The bare servos sag ~0.008 rad under gravity. `JointPIDController` closes an
  outer loop on the measured state:
  `ctrl = q_ref + kp·e + ki·∫e + kd·ė + 0.1·qd_ref`. The integral removes the
  sag; the velocity feed-forward cancels the servo damping lag. Result: max
  tracking error < 0.004 rad, steady-state error ~1e-4 rad.
- To command torques directly, change the actuators via MjSpec in code; keep
  the vendored XML untouched. The v0.2 MPC does not do this: it optimizes
  servo setpoints. Never call setpoints "torques".

## MPC facts learned the hard way (v0.2)

- The servos are stiff and heavily damped relative to link inertia (joint 7:
  I/kd ≈ 0.5 ms). Each 50 Hz setpoint step du produces a ~1 ms torque spike
  kp·du that MuJoCo clips. The MPC's torque constraint is on the *sustained*
  torque at the end of each interval; spikes are left to the physical clip.
- With `implicitfast`, the one-step map is discontinuous at the force-clip
  boundary (servo damping leaves the implicit update while clipped). Finite
  differences straddling it explode. `MuJoCoPredictionModel` then falls back to
  Jacobians of a force-unlimited model copy; the MPC rejects implausible models.
- The finger joints rest on a joint limit. Never compose the full 18-state
  Jacobian; use the arm block only (spectral radius 1e22 otherwise).
- Linearizing through the clip at the planned command was more accurate in
  gentle motion but destabilized the loop in fast motion. Don't reintroduce it
  without the fast/wrist benchmark scenarios passing.
- The setpoint slew limit (|Δu| ≤ v_max·Ts) is what keeps step references from
  blowing up the model. Keep it.
- OSQP 1.x silently ignores `update()` calls with inconsistent bounds (l > u)
  and returns the old solution as "solved". Validate data before updating.
- PyYAML reads `1.0e4` as a string; write `1.0e+4`.
- After a PID fallback, the MPC must anchor to the command actually sent last
  (`_applied`), not its stale solution; v0.2 cascaded fallbacks (275 -> 43 of 450).
- The servo MPC's remaining hard QPs come from conditioning: the sustained
  servo torque barely depends on u, yet effort and torque rows act on it.
  `tools/audit_fallback.py` reproduces the analysis.

## Torque MPC facts (v0.3)

- Optimize normalized torques (tau / tau_max). Raw Nm made P's diagonal span
  ~10 orders of magnitude; OSQP polishing failed and plans drifted 2.7 mrad.
- Soft-constraint exact penalty must exceed the constraint multipliers:
  slack_linear 1e6 with Q = 1e5 (1e5 let a 2.5 rad step plan 3.8x v_max).
- Default 50 Hz, horizon 25 (0.5 s preview), Q = 1e5. At 100 Hz, N = 30 overran
  its 10 ms budget under active constraints.
- Without the v0.4 observer, torque-level controllers have no integral action:
  CTC deflects 35 mrad and torque MPC 12 mrad under the 30 N push (servos: 2-3 mrad). Model-based
  controllers use the exact plant model unless `controller_model` is set;
  with masses 15% too low, torque MPC degrades from 2.8 urad to 1.9 mrad.
- `MjData.site_xpos` read after `mj_step` is one physics step behind `qpos`.
  Use FK of the logged joints for end-effector errors.

## v0.4 facts

- The v0.3 `fast`/`wrist` references ask joint 7 for 1.43x its speed limit.
  Torque MPC enforces the limit (CTC does not), so it must lag: >= 147 / 304
  mrad even with perfect preview. Without the speed limit, or on feasible
  timing, it tracks to < 0.05 mrad. Do not "fix" this by tuning weights.
- Benchmark scenarios before `fast_feasible` are the historical v0.3 ones;
  keep them unchanged. New scenarios must give all controllers the same reference.
- Torque MPC plans exactly at the speed bound when it is active; the real arm
  then exceeds it by the prediction error (~1e-3 rad/s). `velocity_margin`
  0.005 removes that. Servo-MPC violations are intersample (setpoint-step
  transients); knot tightening does not help there.
- The LTI prediction's open-loop error grows ~k^3 over the horizon (~2-5e-2
  rad after 0.5 s) even when closed-loop tracking is perfect: late-horizon
  constraint predictions are approximate. Never claim hard guarantees from them.
- Disturbance observer time constant 10 ms: 5 ms rejects faster but triples
  torque jitter with sensor noise (`sensor_noise` config). The observer and all
  controller models must use `engine.controller_model`, never `engine.model`.
- `torque_mpc_v03` / `torque_ctc` in the benchmark configs reproduce v0.3; keep
  them as regression baselines.

## Motion planning facts (v0.5)

- MuJoCo's midphase ignores the other geom's margin: with it on, `mj_collision`
  missed ~20% of finger-obstacle pairs inside a 20 mm margin. The checker's model
  copy disables it (`mjDSBL_MIDPHASE`); keep it off, and keep
  `test_checker_matches_exhaustive_distance_ground_truth`.
- Clearance (20 mm) + `cartesian_step` (10 mm) certify >= 15 mm to the
  environment everywhere on a checked edge. Self-collision is discrete only.
  Never claim continuous collision avoidance.
- Points on a validated edge are certified only to clearance - step/2, so the
  shortcut must re-check new vertices and shortened edges (it fell back to raw
  paths otherwise). The shortcut minimizes the *unweighted* joint length it reports.
- IK posture choice dominates hard scenes: in `planning_narrow` purely random IK
  seeds gave 12/20; adding seeds perturbed around the current posture gave 20/20.
- `rrt.max_time` covers IK + all RRT attempts + shortcutting.
- A planned move whose path is the free direct edge keeps `move_time`, so
  enabling planning leaves free tasks identical to v0.4.
- The v0.4 torque check (`mj_inverse`) includes contact forces, so it rejects a
  reference that penetrates an obstacle ("needs 140903 Nm"). It is not a collision
  check; `--unchecked-direct` disables it explicitly for the unsafe baseline.
- `environment_overrides` (config) / `SimulationEngine(overrides=)` change body `pos` and
  geom `pos`/`size` in the environment spec before compile (v0.5.1); the XML is untouched
  and the planner's checker copies the compiled model, so it sees the same geometry.
  `configs/robustness_wall.yaml` holds the wall variants. Geometry is known at episode
  start; there is no perception and no replanning during motion. Never execute an old
  plan in a changed scene; re-check it (`checker.path_valid`) and replan.
- MuJoCo 3.16 box-box collision is unreliable *inside the margin*: it can generate no
  contact 1.25 mm apart under a 20 mm margin and misreports `contact.dist` by -5.7..+7.0 mm
  (penetration is always detected). The Panda's fingertip pads are boxes. Fixed in v0.5.2:
  with `exact_box_box` (default; `planning.exact_box_box` in configs) the checker ignores
  MuJoCo's contacts for box-box robot/obstacle pairs and calls `mj_geomDistance` for those a
  conservative bound (box center vs the obstacle's AABB grown by clearance + bounding radius)
  cannot rule out. `tools/audit_clearance.py` certified `mj_geomDistance` (duality gap < 1 um)
  and MuJoCo's margin contacts for every other pair type in the scenes (mesh/box vs
  box/cylinder/plane) on 44,000 poses; only box-box margin contacts failed.
  `test_mujoco_box_box_still_misses_this_margin_contact` pins the MuJoCo behavior (the old
  xfail is now `test_checker_rejects_box_box_near_miss_inside_the_margin`); if the pin fails
  after an upgrade, re-run the audit. `exact_box_box: false` is for comparisons only.
- With exact per-sample answers the edge certificate (>= clearance - cartesian_step / 2 =
  15 mm between samples) holds for MuJoCo's geometry, up to the audited, not proven,
  accuracy of the queries. Self-collision stays discrete. On the whole robot the finger
  meshes sit behind the pads, so v0.5's worst robot-level miss found was 17.8 mm.
- The exact check costs up to ~25 us per configuration, mostly numpy call overhead of the
  filter; `mj_geomDistance` runs only tens to a few hundred times per plan. Querying all
  box-box pairs without the filter is 1.2-2.3x slower; keep the filter.
- `results/video/v0.5.1_demo.webp` (31 s) is historical evidence. The renderer refuses to
  overwrite an existing file without `--overwrite` and stamps `VERSION` on every frame.
- The demo video is written frame by frame through Pillow's `_webp.WebPAnimEncoder`
  (constant memory). `Image.save(save_all=True)` would hold every frame in RAM.
  Offscreen buffers larger than 640x480 need `model.vis.global_.offwidth/offheight`.
- Never substitute the direct move when planning fails: the target is rejected
  (reach) or `PlanningFailed` is raised (joint waypoints).

## Pick-and-place facts (v0.6)

- `pick_place.py` / `simulation/manipulation.py` `PickPlaceEpisode`: an event-based supervisor
  (generator; one `yield` = one physics step) with phases approach, descend, close, lift, hold,
  transport, lower, release, retreat, verify. Phases end on measured conditions or bounded timeouts
  with a stage and reason. The arm reference is built one motion at a time (`tasks/pick_place.py`:
  `JointPathSegment` = planner path with v0.4 timing, `CartesianSegment` = straight line at constant
  orientation from dense pose IK + spline + quintic). The gripper is commanded with
  `engine.set_gripper`, never through the arm controller.
- Known world: object pose/size and obstacles are read from the simulator when a phase is planned.
  The object is a free body moved only by contact and friction. Never write its pose, weld it or
  attach it in the plant. The only rigid attachment is `CollisionChecker.attach` in the planner's
  private model copy.
- Controller model boundary: manipulation configs set `controller_model.contacts: false`, which gives
  every model-based controller `engine.contact_free_model()`. With contacts on, the copied live state
  lets mj_inverse/mj_forward see the finger-object contact, i.e. the payload. Keep
  `test_payload_mass_and_contact_state_do_not_reach_the_controller_model` and its control test.
- The Menagerie gripper servo (100 N/m on the split tendon) squeezes a 50 mm box with only 2.5 N.
  Manipulation configs use `robot.gripper_stiffness: 1500` (~37 N; real Franka Hand 70 N), set via
  MjSpec. The vendored XML is unchanged.
- Plant contact solver for manipulation: `physics: {cone: elliptic, impratio: 10}`. With MuJoCo's
  default pyramidal cone the soft contacts let the box creep ~5 mm through the grasp at 0.4 kg (more
  at 1 kg). Results depend on this; see the sensitivity check in `docs/research/v0.6-results.md`.
- MuJoCo combines friction with the max of the two geoms: lowering only the object's friction does
  nothing against the 1.0 pads (use `physics.friction_scale` for all geoms).
- Collision rules are phase-dependent and narrow (`_configure_checker`): hand/fingers vs object >= 5 mm
  while descending/retreating; finger-object contact allowed and the object attached while
  lifting/carrying/lowering; object-table >= -2 mm only while lifting off / lowering. Every other pair
  keeps 20 mm. The plant monitor (`_monitor_contacts`) classifies every applied contact at every step.
- Lowering recomputes the grasp offset at the start of the phase (the object may slip in transport).
  Release waits for the fingers to stop before the retreat check.
- Torque MPC QP audit (`tools/audit_torque_mpc.py`): objective, rows, signs and the hand-solved
  1-joint case all match. The 1.6-5.3 mrad horizon prediction error is the frozen-Jacobian LTI
  approximation (re-linearizing removes it), not a bug.

## Workbench facts (v0.7)

- `workbench/` is a front end only. `RunRequest` (api.py) -> `resolve()` gives the CLI configs; `LiveRun`
  (runs.py) steps `SimulationRunner` or `PickPlaceEpisode` (`start/advance/abort/finish`). Never add planning,
  control, physics or success logic to the workbench; extend the backends and expose them through the registry.
- Thread ownership: `Session` has one worker thread that creates, steps, renders and closes every live MuJoCo
  object and every `mujoco.Renderer`. HTTP handlers only call `Session.act/configure` or read copies. Replay
  renders recorded `qpos` into a separately built model. Inspector calculations use cached private scratch
  models (`inspect.Scratch`), never the live plant.
- Configuration changes only the pending request; it is applied at the next run/reset.
- Offscreen rendering costs ~11 ms per 960x600 frame without shadows/reflections and ~55 ms with them. Only the
  "high" quality preset enables them, and the worker throttles rendering to about 1/3 of its time.
- The console grammar (commands.py) is deterministic; keep it free of eval/exec/shell.
- The MPC inspector's objective terms plus the omitted constant equal the solver objective
  (`test_mpc_inspector_objective_equals_the_solver_objective_plus_its_constant`). Keep that test when the
  torque-MPC cost changes.
- `pick_place_study.py` refuses to resume an output folder recorded with another commit, config or library
  versions (`--allow-mixed` overrides).

## Conventions and rules

- Never set `qpos`/`qvel` to produce motion. Motion must come from `data.ctrl` + `mj_step`.
- Keep modules separated by role: environments know nothing about robots;
  tasks produce references only; controllers produce actuator commands only.
- Settings belong in `configs/*.yaml`, not hard-coded in modules.
- Keep actuator models apart in comparisons: position-servo and torque-motor
  controllers differ in the actuator interface, not only the controller.
- Any new waypoint must be within the joint limits and collision-free.
  `test_waypoints_within_limits_and_collision_free` and
  `test_benchmark_waypoints_within_limits_and_collision_free` check this. Note
  that folding joint 4 toward its lower limit at the home pose puts the fingers
  into link0/link1. Neither controller models collisions.
- The prediction model must never write the live `MjData`; it uses private
  scratch data and must not see `xfrc_applied` (a controller can't measure pushes).
- Run the tests after every change. Keep the tests headless; the offscreen
  render test skips if no OpenGL context is available.
- Windows viewer quirk: after the passive viewer closes, its background thread
  is still tearing down OpenGL. `run_viewer()` joins that thread before
  returning. Removing the join brings back an access-violation crash at
  interpreter exit.
- Do not commit or push without the user's explicit approval.
