"""Collision-aware motion planning in the arm's joint space (v0.5).

collision.py      configuration / edge validity on a private model copy (clearance via contact margins)
rrt_connect.py    bidirectional RRT-Connect
shortcut.py       collision-checked path shortcutting
ik.py             multiple collision-free IK postures for a Cartesian target
motion_planner.py facade used by the tasks (joint or Cartesian goals)
visualization.py  planned-path markers for MuJoCo's viewer scene
"""

from planning.collision import CollisionChecker, CollisionReport
from planning.motion_planner import MotionPlan, MotionPlanner, PlanningSettings
from planning.rrt_connect import PlanStatus, RRTConnect, RRTResult, RRTSettings
from planning.shortcut import ShortcutResult

__all__ = [
    "CollisionChecker", "CollisionReport", "MotionPlan", "MotionPlanner", "PlanningSettings",
    "PlanStatus", "RRTConnect", "RRTResult", "RRTSettings", "ShortcutResult",
]
