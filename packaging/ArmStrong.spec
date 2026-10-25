# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ArmStrong.exe: one self-contained executable with Python, MuJoCo, the simulation backend,
# the Panda model, the configs and the built frontend. Build it with build-exe.ps1 from the repo root.
#
# Inside the exe, sys._MEIPASS is the project root: modules live there, and so do configs/, environments/,
# robots/models/ and studio/data/, exactly where the code looks for them (simulation/engine.py PROJECT_ROOT).
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent
LAB = ROOT / "backend" / "robotics-lab" / "robotics-lab"
DIST = ROOT / "frontend" / "dist"
PANDA = LAB / "robots" / "models" / "franka_emika_panda"
ICON = Path(SPECPATH) / "ArmStrong.ico"

if not (DIST / "index.html").is_file():
    raise SystemExit("frontend/dist is missing: run `npm run build` in frontend/ first (build-exe.ps1 does this).")

datas = [
    (str(DIST), "frontend_dist"),
    (str(LAB / "configs"), "configs"),
    (str(LAB / "environments"), "environments"),
    (str(LAB / "studio" / "data"), "studio/data"),
    (str(PANDA / "panda.xml"), "robots/models/franka_emika_panda"),
    (str(PANDA / "LICENSE"), "robots/models/franka_emika_panda"),
    (str(PANDA / "SOURCE.txt"), "robots/models/franka_emika_panda"),
    (str(PANDA / "assets"), "robots/models/franka_emika_panda/assets"),
]
binaries = []
hiddenimports = collect_submodules("uvicorn") + ["osqp.ext_builtin"]
for package in ("mujoco", "osqp"):  # native libraries, MuJoCo plugins and dynamically imported modules
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
for distribution in ("osqp", "mujoco", "numpy", "scipy", "fastapi", "starlette", "pydantic", "pydantic-core", "uvicorn"):
    try:
        datas += copy_metadata(distribution)
    except Exception:  # noqa: BLE001 - metadata is only a safety net for version lookups
        pass

a = Analysis(
    [str(LAB / "armstrong.py")],
    pathex=[str(LAB)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "PIL", "pytest", "IPython", "glfw", "OpenGL", "workbench", "pxr", "jax",
              "torch", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ArmStrong",
    console=True,  # the window shows the address and status; closing it quits ArmStrong
    upx=False,
    icon=str(ICON) if ICON.is_file() else None,
)
