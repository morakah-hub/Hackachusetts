import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from armstrong.planner import TaskPlanner  # noqa: E402
from armstrong.rules import RobotRules, World  # noqa: E402
from armstrong.sim import SimWorker  # noqa: E402
from armstrong.store import StudioStore  # noqa: E402


@pytest.fixture
def store():
    """The default scene (assembly kit), in memory only."""
    return StudioStore(None, RobotRules(), World(), persist=False)


@pytest.fixture
def planner(store):
    return TaskPlanner(store, store.world, store.rules)


@pytest.fixture
def worker(store):
    """Fresh simulation worker driven synchronously on the test thread (renderer included)."""
    w = SimWorker(store, realtime=False)
    w.setup_sync()
    yield w
    if w.camera is not None:
        w.camera.close()


def names(store) -> dict[str, str]:
    return {o.name: o.id for o in store.objects.values()}
