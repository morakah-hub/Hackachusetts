"""ArmStrong: a browser app for placing objects on a reach-sized table and running simple tasks.

The package is an application layer over the existing backends, in the same spirit as workbench/:

* settings.py   configs/studio.yaml and runtime paths
* workspace.py  measured top-down reach -> usable ring and table radius (cached)
* robot.py      robot profile, object limits (payload, gripper width, height) and grasp geometry
* geometry.py   compiled MuJoCo visual geometry exported for a Three.js renderer
* scene.py      scene objects, object library, uploaded assets, placement rules, persistence
* planner.py    task catalog, rearrangement into pick-and-place steps, pre-flight feasibility checks
* runner.py     the single simulation worker: one PickPlaceEpisode per step, paced, with snapshots
* server.py     FastAPI routes and static hosting of the built frontend

Physics, control, planning and success checks stay in simulation/, planning/ and tasks/: every task step is
a v0.6 PickPlaceEpisode, so the objects move only through simulated contact and friction.
"""
