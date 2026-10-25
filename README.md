# ArmStrong

ArmStrong is a browser app where you place everyday objects on a table and a simulated Franka Emika Panda arm carries
out simple tasks with them. The physics is real MuJoCo simulation (the arm grips and carries objects through
contact and friction), using the `backend/robotics-lab` pick-and-place pipeline.

## Run it (Windows)

Easiest: double-click `ArmStrong.exe`. It is one self-contained file (about 71 MB, Windows 64-bit) with Python,
MuJoCo, the simulation, the robot model and the web app inside; nothing else needs to be installed. It opens the
browser at http://127.0.0.1:8770 when ready, keeps your layout and uploads in `%LOCALAPPDATA%\ArmStrong`, and quits
when you close its window. The exe is unsigned, so Windows SmartScreen may ask you to confirm (More info, Run anyway).

Build the exe (needs Python 3.12 and Node 24 on the build machine only):

```powershell
powershell -ExecutionPolicy Bypass -File .\build-exe.ps1      # -> dist\ArmStrong.exe
```

From source: double-click `start.bat` (or run `start-studio.ps1`). Nothing has to be installed first. The first run
needs internet and takes a few minutes: it creates `backend\.venv`, installs the Python packages and builds the web
app, then opens http://127.0.0.1:8770. If Python 3.12 or Node.js is missing, it downloads a private copy into
`%LOCALAPPDATA%\ArmStrong\tools` (no administrator rights, nothing added to PATH; delete that folder to remove it).
If `backend\.venv` was copied from another computer and does not run, it is rebuilt.

```powershell
.\start.bat                                                    # or: -NoBrowser -Port 8771
powershell -ExecutionPolicy Bypass -File .\start-studio.ps1    # same thing without the .bat
```

Manual steps (Python 3.12, Node 24):

```powershell
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\robotics-lab\robotics-lab\requirements-studio.txt
cd frontend; npm install; npm run build; cd ..
cd backend\robotics-lab\robotics-lab; ..\..\.venv\Scripts\python.exe studio.py
```

Frontend development: run `studio.py` as above, then `npm run dev` in `frontend` (http://127.0.0.1:5173, `/api` is proxied).

## What it does

- **Table sized by the reach.** The arm's top-down grasp reach is measured with IK and collision checks
  (`studio/workspace.py`, cached in `studio/data/workspace_default.json`): usable ring 330–745 mm from the base,
  ±135° (the area behind the base is kept clear), table diameter 1.61 m. Objects can only be placed inside the ring.
- **Objects.** Built-in samples, custom boxes, or uploaded 3D models (.glb, .gltf, .obj, .stl, with a sample
  drinks can). The physics uses the box around the model. Every object's weight is checked against the 3.0 kg
  payload, and its size against the gripper (15–58 mm across, 30–250 mm tall). Objects the arm can't lift stay
  on the table as obstacles.
- **Tasks.** Move, Bring to me, Swap, Stack, Line up, Sort by weight. Each task is checked first (reach, room for
  the fingers, payload) and then runs as one physical pick-and-place per move, with live 3D view, pause/stop,
  speed control, measured placement errors and an undo.

## Where the code is

- `.kiro/`: project steering (`steering/armstrong.md`) and the spec (`specs/armstrong-studio/`: requirements, design, tasks).
- `frontend/`: React + TypeScript + Three.js app.
- `backend/robotics-lab/robotics-lab/studio/`: FastAPI service (scene, planner, simulation worker, geometry export);
  `configs/studio.yaml`, `environments/studio.xml`. Backend extensions are opt-in, so existing configs are unchanged:
  extra scene objects, stacking supports, arc transport, return home.
- Tests: `..\..\.venv\Scripts\python.exe -m pytest tests -q` in the robotics-lab folder (249 pass).

## Limitations

The robot keeps each object's heading (it does not rotate objects). Uploaded models are simulated as boxes. The
server is local-only and has no authentication. Only the Panda is available.
