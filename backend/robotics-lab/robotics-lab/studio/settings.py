"""Studio settings: configs/studio.yaml (which extends pick_place.yaml) and runtime paths."""

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulation.engine import PROJECT_ROOT
from simulation.runner import load_config

CONFIG_PATH = PROJECT_ROOT / "configs" / "studio.yaml"
# Runtime data (scene, library, uploads, reach cache). results/ is git-ignored.
DEFAULT_DATA_DIR = PROJECT_ROOT / "results" / "studio"
# Built frontend (npm run build in <repo>/frontend). PROJECT_ROOT = <repo>/backend/robotics-lab/robotics-lab.
_PARENTS = PROJECT_ROOT.parents  # shallow when frozen (ArmStrong.exe sets STUDIO_FRONTEND_DIST instead)
DEFAULT_FRONTEND_DIST = (_PARENTS[2] if len(_PARENTS) > 2 else PROJECT_ROOT) / "frontend" / "dist"
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class StudioSettings:
    cfg: dict
    data_dir: Path
    frontend_dist: Path
    config_path: Path = CONFIG_PATH
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str | Path | None = None, data_dir: str | Path | None = None) -> "StudioSettings":
        path = Path(config_path or os.environ.get("STUDIO_CONFIG") or CONFIG_PATH)
        cfg = load_config(path)
        if "studio" not in cfg:
            raise ValueError(f"{path} has no 'studio' section")
        data = Path(data_dir or os.environ.get("STUDIO_DATA_DIR") or DEFAULT_DATA_DIR)
        dist = Path(os.environ.get("STUDIO_FRONTEND_DIST") or DEFAULT_FRONTEND_DIST)
        return cls(cfg=cfg, data_dir=data, frontend_dist=dist, config_path=path)

    # ------------------------------------------------------------------ sections

    @property
    def studio(self) -> dict:
        return self.cfg["studio"]

    @property
    def pick_place(self) -> dict:
        return self.cfg["pick_place"]

    @property
    def environment(self) -> str:
        return self.cfg["environment"]

    @property
    def table_height(self) -> float:
        return float(self.pick_place["table_height"])

    @property
    def home(self) -> np.ndarray:
        return np.asarray(self.studio["robot"]["home"], float)

    @property
    def limits(self) -> dict:
        return self.studio["objects"]

    def episode_config(self) -> dict:
        """A private copy of the pick-and-place config for one episode."""
        return copy.deepcopy(self.cfg)
