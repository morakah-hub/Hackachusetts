# Requirements Document

## Introduction

ArmStrong lets anyone simulate a robot arm doing everyday tasks with objects they choose. The website shows a
Franka Emika Panda and a table. The table's size comes from the arm's maximum reach, which is also the arm's
working space. Users import objects or use samples, enter each object's weight, drag objects onto the table (only
where the arm can reach) and run pre-written tasks. Each object's weight is checked against the arm's payload.
All motion is physically simulated in MuJoCo.

This version covers one robot (Franka Emika Panda), a known world (object poses come from the simulator, with no
camera perception), rigid box proxies for objects, and a single local user.

## Glossary

- **Work area:** the ring around the robot base where top-down grasps are reachable and collision-free, limited
  to ±135° around the front. Currently 330–745 mm from the base.
- **Box proxy:** the solid box the simulation uses for an object. For an uploaded model, it is the box that
  encloses the model.
- **Move:** one physical pick-and-place episode. A task consists of one or more moves.
- **Payload:** the heaviest object the arm may lift (3.0 kg, Franka Emika Panda datasheet).
- **Check:** the pre-flight evaluation of a task on a private copy of the scene.

## Requirements

### Requirement 1: Reach-sized table

**User Story:** As a user, I want the table to match what the arm can reach, so that I can only place objects
where the arm can work with them.

#### Acceptance Criteria

1. WHEN the server starts THEN the system SHALL load the cached work-area measurement whose key matches the
   current configuration and MuJoCo version, or measure the work area again.
2. WHEN the system measures the work area THEN it SHALL probe top-down pre-grasp and grasp poses with pose IK and
   the planner's collision rules for object heights of 40, 120 and 250 mm, four gripper yaws, and every 10° of
   azimuth around the base.
3. The system SHALL set the work area to the common reachable band shrunk by a 20 mm margin, and the table
   radius to the work area's outer radius plus a 60 mm rim.
4. The system SHALL limit the work area to ±135° around the front of the robot and SHALL show the area behind
   the base as not usable.
5. The 3D view SHALL show the work area, the areas that are not usable, and a legend.

### Requirement 2: Object library and import

**User Story:** As a user, I want to import the objects I care about, so that the simulation uses my own items.

#### Acceptance Criteria

1. The system SHALL provide built-in sample objects, each with a name, size, weight and colour.
2. WHEN the user creates a custom box THEN the system SHALL ask for its name, width, depth and height (mm),
   weight (kg) and colour, and SHALL add it to the library.
3. WHEN the user uploads a .glb, self-contained .gltf, .obj or .stl file of at most 30 MB THEN the system SHALL
   preview the model inside its collision box and let the user set the file units, an extra scale, the up axis,
   the name, the weight and (for OBJ and STL) the colour.
4. IF the file type is unsupported, the file is empty or larger than 30 MB, the file contains no meshes, or the
   file uses unsupported compression (Draco, KTX2) THEN the system SHALL reject it with a specific message.
5. The system SHALL simulate an uploaded model as its box proxy and SHALL label that box as the collision shape.
6. The system SHALL keep the library, uploaded files and the scene across server restarts.
7. WHEN the user deletes one of their own library items THEN the system SHALL remove it. Built-in samples SHALL
   NOT be deletable.

### Requirement 3: Placing and editing objects

**User Story:** As a user, I want to drag objects onto the table and arrange them, so that I can set up a task
quickly.

#### Acceptance Criteria

1. WHEN the user drags a library item onto the table THEN the system SHALL place the object upright at the drop
   point. WHEN the user presses Add THEN the system SHALL place it at the nearest free spot.
2. WHILE the user drags an object or a library item THEN the system SHALL show its footprint and, if the spot is
   not allowed, the reason.
3. IF a spot is outside the work area, behind the robot, over the table edge, or closer than 15 mm to another
   object THEN the system SHALL reject the placement and explain why.
4. The system SHALL let the user move (by dragging or typed x/y in mm), rotate (15° steps, Q/E keys), rename,
   recolour, reweigh and remove the selected object.
5. IF another object stands on an object THEN the system SHALL refuse to move or remove the lower object until
   the top one is moved.
6. The system SHALL hold at most 24 objects.
7. WHILE a task is running THEN the system SHALL refuse scene edits and disable the editing controls.
8. The system SHALL offer a sample layout and a way to clear the table.

### Requirement 4: Load and gripper limits

**User Story:** As a user, I want each object checked against what the arm can carry, so that I know before
running whether the robot can handle it.

#### Acceptance Criteria

1. The system SHALL compare each object's weight with the 3.0 kg payload and show the ratio for the selected
   object.
2. IF an object is heavier than the payload, has no side between 15 and 58 mm across, or is shorter than 30 mm
   or taller than 250 mm THEN the system SHALL mark it as not liftable, explain why, and allow it on the table
   only as an obstacle.
3. IF an object is longer than 300 mm or lighter than 0.01 kg THEN the system SHALL refuse to create it.
4. IF an object is tipped more than 10° or has fallen off the table THEN the system SHALL refuse to pick it up
   and say so.
5. The upload and custom-box dialogs SHALL show these checks live, before the object is saved.

### Requirement 5: Pre-written tasks

**User Story:** As a user, I want simple ready-made tasks, so that I can make the robot do useful work without
programming it.

#### Acceptance Criteria

1. The system SHALL offer six tasks: Move (object, destination), Bring to me (object), Swap (two objects), Stack
   (object, base), Line up (objects; order kept, by name, or by height) and Sort by weight (objects).
2. The user SHALL be able to choose task objects from a list or by clicking them in the 3D view, and
   destinations by clicking the table or typing coordinates.
3. WHEN the user presses Check THEN the system SHALL return either an ordered list of moves with an estimated
   duration, or the problems that prevent the task, each with a hint.
4. The check SHALL verify that the objects can be lifted, that destinations follow the placement rules, that the
   pick and place poses are reachable and collision-free, that the lift and the carry path are feasible, that the
   robot's start posture is clear, and that no placement leaves a nearby object without room for the gripper.
5. WHEN a destination is occupied by another object of the same task THEN the system SHALL order the moves and
   use a temporary spot where needed (for example, for Swap).
6. WHEN a move would swing an object more than 110° around the base THEN the system SHALL split it with a
   put-down in between, so that the grip orientation can change while the object keeps its heading.
7. IF the check fails THEN the system SHALL NOT start the task.
8. Stack SHALL refuse stacks taller than 400 mm and objects whose footprint is more than 2.5 times that of the
   base.

### Requirement 6: Physical execution and live view

**User Story:** As a user, I want to watch the robot do the task in a real simulation, so that I can trust the
result.

#### Acceptance Criteria

1. The system SHALL run each move as a physical pick-and-place episode with the phases approach, descend, close,
   lift, hold, transport, lower, release, retreat and verify, and SHALL return the arm to rest after the last
   move.
2. During a run, objects SHALL move only through simulated actuator commands, contact and friction. The system
   SHALL NOT teleport, weld or animate objects.
3. WHILE a task runs THEN the browser SHALL render the robot and object poses streamed by the simulation (about
   30 updates per second) and show the current move, phase, progress and simulated time.
4. The user SHALL be able to pause, resume and stop a run, and set the speed to ½×, 1×, 2×, 4× or as fast as
   possible.
5. WHEN a move is verified THEN the system SHALL report the measured placement error. A move SHALL count as
   successful only if the object ends within 10 mm of its target and tilted at most 10°.
6. IF a move fails or is stopped THEN the system SHALL open the gripper, let the objects settle, keep their
   measured poses, skip the remaining moves, and explain the failure in plain language with the simulation
   details available on request.
7. WHEN a run ends THEN the user SHALL be able to restore the layout from before the run, or dismiss the result.
8. The system SHALL run at most one task at a time.

### Requirement 7: Usability and branding

**User Story:** As a general user, I want a clear and friendly interface, so that I can use the app without
training.

#### Acceptance Criteria

1. The app SHALL be named ArmStrong and SHALL show the ArmStrong logo (an original smiling, gender-neutral robot
   flexing a mechanical arm) in the top bar and as the favicon.
2. The top bar SHALL show the connection and run status and a robot selector with the Panda selected and other
   robots marked as planned and disabled.
3. Controls SHALL have accessible names, dialogs SHALL be modal and close with Escape, and animations SHALL
   respect the reduced-motion setting.
4. Errors SHALL appear as dismissible notifications with a hint where one is available.

### Requirement 8: Local operation

**User Story:** As a team member, I want to start the app with one command on a laptop, so that demos are
reproducible.

#### Acceptance Criteria

1. WHEN the user runs `start-studio.ps1` THEN the system SHALL, on first run, create the Python environment,
   install the pinned dependencies and build the frontend, and SHALL then serve the app at
   http://127.0.0.1:8770.
2. The server SHALL bind to 127.0.0.1 by default and SHALL warn when bound to another interface, because it has
   no authentication.
3. The frontend development server SHALL forward `/api` requests to the ArmStrong server.

## Out of scope (this version)

Camera perception, robots other than the Panda, changing an object's heading, deformable objects, accounts and
cloud deployment.
