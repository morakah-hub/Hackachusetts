You are the sole active Kiro coding agent for our Hackachusetts project.

Repository: https://github.com/morakah-hub/Hackachusetts

Build and finish the entire application in this repository: simulation, control, camera perception, backend, frontend, integration, and launch documentation. Only one teammate operates Kiro at a time. When credits run low, we transfer the same repository and branch to the next person's laptop.

This instruction supersedes all previous A/B/C/D ownership, parallel-development, role-prompt, and "Mike alone may edit" restrictions in our project documents. You own all application layers. Preserve useful existing code and technical constraints. Do not restart a functioning implementation or wait for another role.

START WITH THE ACTUAL PROJECT

1. Inspect the current folder, git status, branch, remotes, recent commits, README, .kiro instructions, and any HANDOFF.md. Preserve uncommitted work. If the repository is not present, clone the URL into a new folder; never clone over existing files.
2. Inspect relevant remote branches for teammates' existing work. Bring in compatible useful work deliberately after reviewing it; do not blindly merge every branch. Continue the established shared development branch. If none exists, create build/workcell and record it clearly.
3. Read the supplied handoff kit and historical robotics-lab reference if available. Use their technical lessons, but replace their team-allocation instructions with this single-operator workflow. This prompt is sufficient to begin if those attachments are missing.
4. Preserve current hackathon work. Treat pre-hackathon robotics-lab code as reference by default; only inherit implementation where the event rules permit it, and document inherited versus newly built work. Keep asset licenses and attribution.
5. Spend at most about ten minutes on inspection and a short actionable plan. Then implement. Make ordinary engineering decisions yourself; ask only when a genuine missing requirement blocks progress.

PRODUCT TO DELIVER

ArmStrong: a polished browser application that helps a small workshop simulate repetitive tabletop handling.

The first complete demonstration:
- Open the included workcell with a Franka Panda, table, one opaque colored block, and a destination pad.
- Select the block and destination, then press Run.
- The simulated RGB-D camera estimates the block's position.
- The arm physically grasps, lifts, carries, releases, and verifies placement.
- The UI shows actual movement, camera output, task progress, and measured results.
- Stop/reset, move the block to another supported position, and run again using a fresh observation and plan.

One reliable complete task is the first milestone. Build additional capabilities only afterward.

IMPLEMENT IN THIS ORDER

A. Establish a working complete application path.
Use existing functioning modules first. Otherwise create the Panda/table/block scene, incremental simulation stepping, reset/stop, backend state, and a minimal browser page displaying the actual simulation. Connect the page to the backend immediately.

Use Python + MuJoCo for physics, FastAPI/Uvicorn for the API, and React/TypeScript/Vite/Three.js for the frontend if those choices are not already implemented. Preserve a compatible existing stack rather than migrating for its own sake.

One simulation worker owns live MuJoCo state and renderer contexts. HTTP handlers enqueue commands and read snapshots. Support one active run. Keep state/frame queues bounded. Use straightforward polling initially. Serve locally with a documented development proxy or a single origin.

If the Three.js geometry bridge becomes a prolonged blocker, display the real rendered MuJoCo viewport with working controls first, then improve rendering. Never let frontend architecture block a runnable task.

B. Make one physical pick-and-place reliable.
Use a fixed, validated Panda mount and table. If the reference is present, begin from its known working scene dimensions, robot configuration, and gripper/contact settings.

Use full pose IK for a top-down grasp, a bounded task state machine, and a working joint controller. Approach, descend, close, lift, transport, lower, release, retreat, and verify.

The block must move through actuator commands and simulated contact/friction. Do not teleport it during execution, weld it to the gripper, disable relevant collisions, or animate a successful grasp in the browser.

A known-world mode is acceptable as an explicitly labelled development baseline. Get it working, then implement camera mode. Use existing validated MPC if available; otherwise establish reliable basic control before adding torque MPC. Report the controller actually used.

C. Add genuine, deliberately limited camera perception.
Render synchronized RGB and metric depth from one fixed calibrated camera, initially around 320 x 240. Include camera intrinsics, optical-to-world transform, frame ID, scene revision, and timestamps. Validate projection/back-projection before trusting estimates.

Use meters and a right-handed Z-up world. Keep quaternion conventions explicit; convert deliberately between backend and browser formats.

Begin with one upright cuboid of known dimensions and distinctive color. Identify visible pixels from RGB, back-project depth, and estimate its position using the known object dimensions. Constrain unsupported orientation rather than pretending to recover it.

The detector receives images, calibration, and object identity/dimension/color priors. It must not receive hidden simulator poses, editor coordinates, or privileged segmentation IDs. Camera-mode planning must consume its estimate. Simulator truth is allowed for rendering and evaluation, clearly separated from policy input.

Missing, stale, masked, or ambiguous images must invalidate the observation and prevent starting a new blind motion phase. Do not silently fall back to known-world mode. If static obstacles are known geometry, label that limitation.

D. Make the interface beautiful and usable.
Use a large central 3D workspace, a compact setup/task sidebar, and a camera/results panel. Use restrained charcoal or warm-gray surfaces, one teal/blue accent, readable typography, coherent spacing, good lighting, and correct robot geometry.

Include:
- Connection and run status.
- Panda selected in a robot dropdown; future robots disabled and marked Planned.
- Included sample scene that works without uploads.
- Block selection, table-plane dragging, and numeric placement controls.
- Source/destination selection and obvious Run, Pause, Stop, Reset controls.
- Task phases and useful failure explanations.
- Real RGB/depth views and camera freshness.
- Measured placement result.
- A compact Developer drawer with available diagnostics, controller, timing, logs, and export.

Render authoritative backend transforms. Keep client visual interpolation separate from physics and measurement. Test alignment on a robot link and rotated object.

Editing is allowed only while stopped/idle. Accepted edits increment the scene revision and invalidate old detections/plans. Wait for the backend to accept the revised scene before starting. Show why an action is unavailable. Missing metrics display unavailable.

E. Extend only after the complete loop passes.
First improve visual polish and reproducibility. Then, if time permits:
- Add or retain validated torque MPC with its real active/fallback status.
- Support a bounded OBJ import with explicit units, scale, mass, friction, and a supported collision proxy.
- Demonstrate one static obstacle or repeated pick-and-place.

A room scan is initially visual scenery with separate explicit collision surfaces. A mesh alone does not define physical properties or a grasp. Arbitrary meshes do not automatically become supported manipulation objects.

Defer accounts, public cloud deployment, new arms, RL/VLA training, deformable objects, and general-purpose manipulation. GitHub stores our source; the Python simulation runs on a laptop. Make the local demonstration reproducible first.

CHECK WHAT ACTUALLY WORKS

Run focused checks on the new implementation:
- Real grasp, lift, release, and settled placement.
- Changed supported block position produces a new observation and successful plan.
- Camera masking prevents a camera-guided task.
- Disabled gripper closure cannot report success.
- An unreachable target gives a clear failure.
- Run/stop/reset and scene revisions behave correctly.
- The frontend builds and controls the real backend.

Record actual commands and results. Do not claim an old reference test proves the new implementation works. Never invent benchmark values, screenshots, success rates, Kiro usage, or completed features.

Preserve an already-working task while improving it. Benchmark the actual laptop and lower display/sensor rates if needed before changing physics settings. Keep dependencies modest and avoid unnecessary GPU training packages.

MAKE EVERY SESSION TRANSFERABLE

Create and maintain these small files from the beginning:

- README.md: exact tested setup/launch commands, dependencies, sample workflow, and limitations.
- HANDOFF.md: active branch; completed capabilities; latest verified milestone; environment/version requirements; exact commands; test results; known bugs; current unfinished change; and the next three concrete tasks.
- NEXT_SESSION_PROMPT.md: a short prompt telling the next Kiro session to read this project and HANDOFF.md, verify the current state, and continue the next task without rebuilding working modules.
- .kiro/: concise project steering and specs/tasks that reflect this single-operator workflow and actual work.

After each meaningful runnable milestone, update the handoff, inspect the diff, commit the relevant project files, and push to the agreed branch in the named repository when authentication is available. Verify push success. Never force-push or discard teammates' work. If push fails, keep the local commit and clearly report the blocker and commit ID.

Keep dependency declarations, essential sample assets, licenses, configs, and launch scripts in Git. Keep secrets, virtual environments, node_modules, caches, large generated outputs, and historical reference archives out. Make the next laptop able to run from repository files and documented dependencies without your chat history or machine-specific absolute paths. Document primary Windows PowerShell commands and other platforms where actually verified.

Do not rely on detecting the exact remaining credit balance. Keep checkpoints current throughout. When I say HANDOFF:
1. Stop adding features and leave the smallest understandable checkpoint.
2. Update the handoff and next-session prompt, explicitly marking unfinished or failing work.
3. Commit and push relevant changes, verifying the remote contains the checkpoint.
4. Report branch, commit, launch command, last working capability, remaining problems, and the next task.
5. End the session so another teammate can become the sole operator.

On a takeover, inspect the branch and local changes before pulling. Use a fast-forward update when possible; preserve and resolve divergence deliberately. Read the handoff, run the documented smoke check, and continue. If the previous session stopped abruptly, reconstruct progress from the files and Git diff instead of starting over.

START NOW

Briefly state what already exists, the branch you will use, and the first executable milestone. Then implement and test continuously. Do not stop after producing a plan, scaffolding, or a mock interface. Keep moving toward the complete physical camera-guided demonstration.

