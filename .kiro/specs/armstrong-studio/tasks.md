# Implementation Plan

`[x]` means done and verified; `[ ]` means open. Requirement numbers refer to `requirements.md`.

- [x] 1. Set up the environment and verify the existing backend
  - Python 3.12 venv in `backend\.venv` with the pinned robotics-lab requirements (the bundled `.venv` targets
    another machine)
  - Smoke test: `pick_place.py --case dev_free_1` succeeded with a 0.11 mm placement error
  - _Requirements: 8.1_

- [x] 2. Extend the simulation backend for multi-object scenes (opt-in)
  - [x] 2.1 Free boxes through `environment_overrides["objects"]`, and body quaternions (`simulation/engine.py`)
    - _Requirements: 3.1, 6.2_
  - [x] 2.2 Measured object pose, placement height, support bodies, grasp width, fingertip clearance to supports,
    and an optional `home` phase (`simulation/manipulation.py`)
    - _Requirements: 5.1, 6.1_
  - [x] 2.3 Arc transport at constant grip orientation, with fallback to the joint-space planner
    (`CartesianSegment.plan_path`, `carry_arc_points`, `_carry_arc`)
    - _Requirements: 6.1, 6.2_
  - [x] 2.4 Approach and pre-place postures filtered for the straight descent and lowering
    (`plan_to_position(accept=)`, `straight_line_postures`)
    - _Requirements: 5.4_
  - The existing robotics-lab tests still pass (236)

- [x] 3. Measure the work area and size the table
  - `studio/reach.py` (`GraspProbe`), `studio/workspace.py`, `environments/studio.xml`, `configs/studio.yaml`
  - Measured: reachable all around at 330–745 mm, limited to ±135°, table radius 0.805 m; packaged in
    `studio/data/workspace_default.json`
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 4. Robot rules for payload, gripper and height (`studio/robot.py`, `configs/studio.yaml`)
  - Payload 3.0 kg (3.0 kg held in a check run, 3.5 kg slipped); grip width 15–58 mm; height 30–250 mm
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Scene store, library and uploaded assets (`studio/scene.py`)
  - Placement rules (work area, behind the robot, table edge, 15 mm gap), stacked supports, object states,
    7 built-in samples, JSON persistence, asset storage up to 30 MB
  - _Requirements: 2.1, 2.6, 2.7, 3.3, 3.5, 3.6, 3.8_

- [x] 6. Task planner (`studio/planner.py`)
  - [x] 6.1 Six tasks composed into legs: Move, Bring to me, Swap, Stack, Line up, Sort by weight
    - _Requirements: 5.1, 5.8_
  - [x] 6.2 Rearrangement ordering with buffer spots, and row spacing that leaves room for the open hand
    - _Requirements: 5.5_
  - [x] 6.3 Feasibility: pick and place poses, lift, carry arc, start posture, neighbour finger room
    - _Requirements: 5.4, 5.7_
  - [x] 6.4 Split swings over 110° with an intermediate put-down (`via` legs)
    - _Requirements: 5.6_
  - [x] 6.5 Plain-language problems with hints
    - _Requirements: 5.3_

- [x] 7. Simulation runner (`studio/runner.py`)
  - One worker thread, an episode per leg, wall-clock pacing, pause/resume/stop, speed, settling after failures,
    measured poses written back, plain-language failure messages, return to rest
  - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.6, 6.8_

- [x] 8. Geometry export for the browser (`studio/geometry.py`)
  - Compiled visual meshes as a binary blob plus body poses; 61 geoms, 56 meshes
  - _Requirements: 6.3_

- [x] 9. HTTP API and launcher
  - `studio/server.py`, `studio.py`, `requirements-studio.txt`, `start-studio.ps1`; binds to 127.0.0.1 and warns
    on other interfaces
  - _Requirements: 8.1, 8.2_

- [x] 10. Frontend foundation
  - Vite + React + TypeScript, typed API client, Zustand store with polling, `/api` dev proxy
  - _Requirements: 6.3, 8.3_

- [x] 11. 3D workspace (`three/SceneView.ts`, `three/robotModel.ts`)
  - Robot at streamed poses, work area and non-usable areas, objects (box or uploaded mesh), selection, dragging
    with live checks, dropping from the library, picking objects and points, plan arcs, camera presets
  - _Requirements: 1.5, 3.1, 3.2, 5.2, 6.3_

- [x] 12. Panels and dialogs
  - [x] 12.1 Library panel and scene list
    - _Requirements: 2.1, 3.1, 3.8_
  - [x] 12.2 Object inspector: weight and payload bar, x/y entry, rotate, rename, remove, keyboard shortcuts
    - _Requirements: 3.4, 4.1_
  - [x] 12.3 Task panel: task cards, parameter pickers, Check and Run, plan and problem display
    - _Requirements: 5.1, 5.2, 5.3_
  - [x] 12.4 Run panel: progress, phases, speed, pause/stop, failure details, undo
    - _Requirements: 6.3, 6.4, 6.6, 6.7_
  - [x] 12.5 Upload dialog (preview, units, up axis, live checks, sample drinks can) and custom-box dialog
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 4.5_

- [x] 13. ArmStrong name and logo
  - Original smiling, gender-neutral robot flexing a mechanical arm; top bar, favicon, titles, docs
  - _Requirements: 7.1, 7.2_

- [x] 14. Verification
  - `tests/test_studio.py`: 13 passed; robotics-lab suite: 236 passed (run separately); `npm run build` passes
  - All six tasks ran through the API with placement errors of 0.2–3.4 mm; refusals (out of reach, too heavy,
    edit while running) returned the expected errors; stop worked
  - Headless-Chrome checks of dragging, picking, OBJ and GLB uploads and full runs, with no console errors
  - _Requirements: all_

- [x] 15. Documentation
  - `README.md`, `.kiro/steering/armstrong.md`, this spec
  - _Requirements: 8.1_

- [x] 21. Single-file Windows executable (`dist\ArmStrong.exe`, about 71 MB)
  - `armstrong.py` entry point, `packaging/ArmStrong.spec` (PyInstaller 6.22.3), `build-exe.ps1`, icon from the logo
  - Bundles Python, MuJoCo, the backend, the Panda model, configs and the built frontend; user data in
    `%LOCALAPPDATA%\ArmStrong`; reuses a running instance; picks another port if 8770 is taken
  - Verified: the exe served the app and ran a physical Move (0.93 mm placement error) with no Python on the path
  - _Requirements: 8.1, 8.2_

- [ ] 16. Carry long swings without the intermediate put-down
  - Current behaviour: moves that swing more than 110° around the base are split with a put-down, because
    joint 7 has to counter-rotate by the swing to keep the object's heading
  - Options: choose the grip representation and joint-7 branch that keeps the wrist in range for the whole
    swing; rotate the grip during the carry within joint 7's range; or let the heading change, with the user's
    consent
  - Update `test_planner_decomposes_and_explains`, which currently expects `["via", "move"]`
  - _Requirements: 5.6, 6.1_

- [ ] 17. Commit the work to a shared branch
  - Needs a decision first: `backend/` is untracked in the outer repo and contains a nested git repo
  - _Requirements: 8.1_

- [ ] 18. Write handoff files (`HANDOFF.md`, `NEXT_SESSION_PROMPT.md`) for the single-operator workflow
  - _Requirements: 8.1_

- [ ] 19. (Optional) Split the frontend bundle (about 1 MB, 269 kB gzipped) to load the 3D code on demand
  - _Requirements: 7.3_

- [ ] 20. (Optional, future) RGB-D camera perception from the operator brief; out of scope in this version
