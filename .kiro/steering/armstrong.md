---
inclusion: always
---

# ArmStrong: project steering

## Product

ArmStrong is a local browser app that simulates a robot arm doing everyday tabletop tasks. A Franka Emika Panda
stands at the centre of a round table sized from the arm's measured reach. Users add objects (samples, custom
boxes or uploaded 3D models), enter their weight, drag them onto the table and run pre-written tasks: Move,
Bring to me, Swap, Stack, Line up, Sort by weight. Each move is a physical MuJoCo pick-and-place, and the
browser shows the real simulated motion.

Users are not roboticists: keep the UI simple, explain every refusal in plain words, and never report a result
the simulation did not measure.

The spec lives in `.kiro/specs/armstrong-studio/` (requirements, design, tasks). Keep `tasks.md` honest: mark a
task done only after it is verified.

## Tech stack

- Backend: Python 3.12 in `backend\.venv`; MuJoCo 3.16.0, NumPy 2.5.4, SciPy 1.18.1, OSQP 1.1.3, PyYAML 6.0.3,
  Pillow 13.0.0, FastAPI 0.162.0, Uvicorn 0.54.0 (pinned in `requirements-studio.txt`).
- Frontend: React 19.3, TypeScript 7.0, Vite 8.3, Three.js 0.186, Zustand 5.0 (exact versions in `package.json`).
- Platform: Windows, PowerShell, Node 24.
- Simulation: torque-controlled Panda, computed-torque control with a disturbance observer, 2 ms physics steps,
  elliptic friction cone (`configs/studio.yaml` extends `pick_place.yaml`).

## Structure

```
.kiro/                         steering and the spec
start-studio.ps1               one-command launcher (installs on first run, serves :8770)
frontend/                      React + Three.js app
  src/api/                     typed API client and types
  src/components/              panels, dialogs, top bar
  src/three/                   SceneView (3D workspace), robot model, mesh loading
  src/lib/geom.ts              client copy of the placement and lift rules (instant feedback only)
  src/store.ts                 app state and polling
  public/                      logo, favicon, sample drinks-can.obj
backend/robotics-lab/robotics-lab/
  studio.py                    server entry point
  studio/                      ArmStrong service: settings, robot rules, reach, workspace, geometry,
                               scene store, planner, runner, FastAPI server
  studio/data/                 packaged reach measurement (workspace_default.json)
  configs/studio.yaml          all ArmStrong settings
  environments/studio.xml      round table scene
  simulation/ tasks/ planning/ robotics-lab backend (ArmStrong extensions are opt-in)
  tests/test_studio.py         ArmStrong tests
```

## Commands (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\start-studio.ps1          # from the repo root
powershell -ExecutionPolicy Bypass -File .\build-exe.ps1             # -> dist\ArmStrong.exe (single file)

# from backend\robotics-lab\robotics-lab
..\..\.venv\Scripts\python.exe studio.py --no-browser                # http://127.0.0.1:8770
..\..\.venv\Scripts\python.exe -m pytest tests\test_studio.py -q
..\..\.venv\Scripts\python.exe -m pytest tests -q                    # full suite, about 2 min
..\..\.venv\Scripts\python.exe -m studio.workspace --write-default   # re-measure the reach, about 3 min

# from frontend
npm run dev      # http://127.0.0.1:5173, proxies /api to :8770
npm run build    # typecheck + build into frontend\dist (served by studio.py)
```

## Rules

1. Physics is the source of truth. During a run, objects move only through actuator commands, contact and
   friction: never teleport, weld or animate an object or a grasp. The browser renders poses streamed by the
   server, and its smoothing is cosmetic. Scene edits (placing, dragging, undo) happen only while idle.
2. One worker thread (`studio-sim` in `studio/runner.py`) owns the live MuJoCo state. HTTP handlers only queue
   commands or read copies, and planner checks use private scene copies. One run at a time; scene edits are
   refused (HTTP 409) while a run is active.
3. The server decides placement and lift rules (`studio/scene.py`, `studio/robot.py`). `frontend/src/lib/geom.ts`
   mirrors them for instant feedback, so change both together.
4. Settings belong in `configs/studio.yaml`. After changing workspace, grasp-depth, collision or approach
   settings, re-measure the reach with `--write-default`. `test_packaged_work_area_matches_the_configuration`
   catches a stale cache.
5. Changes in `simulation/`, `tasks/` and `planning/` stay opt-in through config, so existing robotics-lab
   behaviour and tests do not change. Never edit the vendored Panda model (`robots/models/franka_emika_panda`).
6. Units: metres, kilograms, radians; right-handed world, z up. Server quaternions are MuJoCo order (w, x, y, z);
   convert them explicitly for Three.js (x, y, z, w).
7. Explain refusals as a `Problem` (code, message, hint) in plain words. Translate simulator reasons with
   `_friendly` (planner) and `explain_failure` (runner).
8. UI: charcoal surfaces, one teal accent (`#2ec4b6`), accessible labels, keyboard shortcuts (Q/E rotate,
   Delete removes, Esc deselects), reduced-motion support.
9. The server is local only (127.0.0.1) and has no authentication. Flag any change that exposes it.
10. The product name is ArmStrong; internal names stay `studio` (`studio/`, `studio.py`, `/api/...`).
11. Use `backend\.venv`, not `backend\robotics-lab\robotics-lab\.venv` (it points at another machine).
12. Don't commit or push without the user's approval. `backend/` is untracked in the outer repo and contains a
    nested git repo. Never commit `.venv`, `node_modules`, `frontend/dist` or `results/`.

## Workflow

One operator at a time. On takeover, read `README.md`, this file and `tasks.md`, run the studio tests, then
continue with the next open task without rebuilding working modules. After a change, run the studio tests (plus
the full suite for backend changes) and `npm run build` for frontend changes, then update `tasks.md`.
