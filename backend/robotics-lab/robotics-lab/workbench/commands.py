"""Workbench command console (v0.7): a small deterministic grammar mapped onto the same session actions and
`RunRequest` changes as the GUI. No eval/exec, no shell, no language model: each line is split into words
and matched against the table below. Several commands can be joined with ';'.

Settings commands change the *pending* request (shown as resolved settings after every command); they apply
to the next run or reset, never to an active trajectory.
"""

from __future__ import annotations

import re

from workbench.api import CONTACT_PRESETS, GRIP_PRESETS, SCENARIOS, RequestError

HELP = """\
COMMANDS (case-insensitive; join several with ';'; units: m, kg, N, s, rad)
  help                          this text            settings          show the resolved pending settings
  load <scenario>               wall | table | pick | obstacle   (wall_reach, table_reach, pick_place, pick_place_obstacle)
  controller <name>             mpc | ctc | pid | servo-mpc      (torque_mpc, torque_ctc, joint_pid, joint_mpc)
  observer on|off               disturbance observer (torque controllers only)
  payload <kg>                  object mass, e.g. 'payload 0.40'           (pick-and-place)
  object <x> <y>                initial object position on the table (m)    (pick-and-place)
  place <x> <y>                 placement target (m)                        (pick-and-place)
  target <x> <y> <z>            replace the reach targets by one target; 'target add <x> <y> <z>' appends
  targets default               scenario default targets
  obstacle pos <x> <y> <z>      obstacle centre (m);  'obstacle size <hx> <hy> <hz>' half extents;  'obstacle default'
  contact <preset>              v0.6-working | mujoco-default
  grip <preset>                 strong | weak     (v0.6-strong, menagerie-weak)
  seed <n>                      planning seed
  force <body> <N> dir <x> <y> <z> at <t> for <d>   e.g. 'force hand 20 dir 0 0 -1 at 3 for 0.5';  'force off'
  noise <q_std> <qd_std> [seed <n>]                 arm measurement noise (rad, rad/s);  'noise off'
  check                         pre-flight: validation, IK search for the key poses, start collisions
  scan                          workspace scan for the top-down grasp (pick-and-place), shown in the editor
  run | pause | resume | stop | reset
  step [n]                      n physics steps (2 ms each);  'step interval' = one controller interval
  speed <x> | speed max         real-time factor of the live run (physics timestep unchanged)
  replay [run_id|last]          open a recorded run;  'seek <t>';  'play [speed]';  'live'
  show <panel>                  motion | mpc | dynamics | planning | explain | sandbox | experiments | runs | result
  explain [t]                   explain the selected (or given) moment from recorded data
  compare <what>                guided experiment: payload | observer | controllers | contact | grip | force | noise
  camera reset
EXAMPLES
  load wall; controller mpc; run
  load pick; payload 1.15; observer off; run
  compare observer
"""

SCENARIO_ALIASES = {"wall": "wall_reach", "wall_reach": "wall_reach", "wall-demo": "wall_reach",
                    "table": "table_reach", "table_reach": "table_reach", "reach": "table_reach",
                    "pick": "pick_place", "pick_place": "pick_place", "pick-place": "pick_place",
                    "obstacle": "pick_place_obstacle", "pick_place_obstacle": "pick_place_obstacle"}
CONTROLLER_ALIASES = {"mpc": "torque_mpc", "torque_mpc": "torque_mpc", "torque-mpc": "torque_mpc",
                      "ctc": "torque_ctc", "torque_ctc": "torque_ctc", "computed-torque": "torque_ctc",
                      "pid": "joint_pid", "joint_pid": "joint_pid", "servo-mpc": "joint_mpc", "joint_mpc": "joint_mpc"}
GRIP_ALIASES = {"strong": "v0.6-strong", "weak": "menagerie-weak", **{k: k for k in GRIP_PRESETS}}
CONTACT_ALIASES = {"working": "v0.6-working", "default": "mujoco-default", **{k: k for k in CONTACT_PRESETS}}
PANELS = ("motion", "mpc", "dynamics", "planning", "explain", "sandbox", "experiments", "runs", "result")


def _num(word: str, what: str) -> float:
    try:
        return float(word)
    except ValueError:
        raise RequestError(f"{what}: '{word}' is not a number") from None


def _nums(words: list[str], n: int, what: str) -> list[float]:
    if len(words) != n:
        raise RequestError(f"{what} needs {n} numbers, got {len(words)}")
    return [_num(w, what) for w in words]


class CommandConsole:
    def __init__(self, session, experiments=None):
        self.session = session
        self.history: list[str] = []
        self.experiments = experiments

    def help_text(self) -> str:
        return HELP

    def execute(self, text: str) -> dict:
        """Run one console line. Returns {"ok", "output": [lines], "settings": [resolved], "ui": {...}}."""
        text = (text or "").strip()
        out, ui, ok = [], {}, True
        if text:
            self.history.append(text)
        for part in [p.strip() for p in text.split(";") if p.strip()]:
            try:
                lines, u = self._one(part)
                out += lines
                ui.update(u)
            except RequestError as exc:
                out.append(f"error in '{part}': {exc}")
                ok = False
                break
        return {"ok": ok, "output": out, "settings": self.session.request.summary(), "ui": ui,
                "history": self.history[-50:]}

    def _set(self, **changes) -> list[str]:
        req = self.session.configure(**changes)
        applies = " (applies at the next run/reset)" if self.session.status.state in ("running", "paused") else ""
        return [f"set {', '.join(f'{k} = {v}' for k, v in changes.items())}{applies}"] + \
               ([] if not applies else []) + ([f"  scenario: {req.scenario}"] if "scenario" in changes else [])

    def _one(self, line: str) -> tuple[list[str], dict]:
        words = re.split(r"[\s,()]+", line.lower().strip())
        words = [w for w in words if w]
        cmd, args = words[0], words[1:]
        s = self.session
        if cmd in ("help", "?"):
            return HELP.splitlines(), {}
        if cmd == "settings":
            return s.request.summary(), {}
        if cmd == "load" or cmd == "scenario":
            name = SCENARIO_ALIASES.get("_".join(args)) or SCENARIO_ALIASES.get(args[0] if args else "")
            if not name:
                raise RequestError(f"unknown scenario; use one of {sorted(SCENARIOS)}")
            ctrl = s.request.controller if s.request.controller in SCENARIOS[name].controllers else "torque_mpc"
            lines = self._set(scenario=name, controller=ctrl, targets=None, obstacle=None)
            return lines, {"panel": "result"}
        if cmd in ("controller", "select", "use"):
            key = "-".join(a for a in args if a not in ("controller",)) or ""
            key = {"torque-mpc": "torque_mpc", "torque": "torque_mpc"}.get(key, key)
            name = CONTROLLER_ALIASES.get(key) or CONTROLLER_ALIASES.get(args[-1] if args else "")
            if not name:
                raise RequestError(f"unknown controller; use one of {sorted(set(CONTROLLER_ALIASES))}")
            changes = {"controller": name}
            if name not in ("torque_mpc", "torque_ctc"):
                changes["observer"] = False
            return self._set(**changes), {}
        if cmd == "observer":
            if not args or args[0] not in ("on", "off", "enable", "disable"):
                raise RequestError("use 'observer on' or 'observer off'")
            return self._set(observer=args[0] in ("on", "enable")), {}
        if cmd in ("payload", "mass"):
            if not args:
                raise RequestError("use e.g. 'payload 0.40'")
            return self._set(payload_kg=_num(args[0].removesuffix("kg"), "payload")), {}
        if cmd == "object":
            return self._set(object_xy=_nums(args, 2, "object x y")), {}
        if cmd == "place":
            return self._set(place_xy=_nums(args, 2, "place x y")), {}
        if cmd == "target":
            if args and args[0] == "add":
                return self._set(targets=s.request.resolved_targets() + [_nums(args[1:], 3, "target x y z")]), {}
            return self._set(targets=[_nums(args, 3, "target x y z")]), {}
        if cmd == "targets" and args == ["default"]:
            return self._set(targets=None), {}
        if cmd == "obstacle":
            if args == ["default"]:
                return self._set(obstacle=None), {}
            cur = s.request.resolved_obstacle()
            if cur is None:
                raise RequestError(f"scenario '{s.request.scenario}' has no editable obstacle")
            cur = {"pos": list(cur["pos"]), "size": list(cur["size"])}
            if not args or args[0] not in ("pos", "size"):
                raise RequestError("use 'obstacle pos x y z', 'obstacle size hx hy hz' or 'obstacle default'")
            cur[args[0]] = _nums(args[1:], 3, f"obstacle {args[0]}")
            return self._set(obstacle=cur), {}
        if cmd == "contact":
            name = CONTACT_ALIASES.get(args[0] if args else "")
            if not name:
                raise RequestError(f"contact preset must be one of {sorted(CONTACT_PRESETS)}")
            return self._set(contact_preset=name), {}
        if cmd == "grip":
            name = GRIP_ALIASES.get(args[0] if args else "")
            if not name:
                raise RequestError("grip preset must be 'strong' or 'weak'")
            return self._set(grip_preset=name), {}
        if cmd == "seed":
            return self._set(plan_seed=int(_num(args[0] if args else "x", "seed"))), {}
        if cmd == "force":
            if args == ["off"]:
                return self._set(force_pulse=None), {}
            m = re.fullmatch(r"(\w+) ([-\d.]+)n? dir ([-\d.]+) ([-\d.]+) ([-\d.]+) at ([\d.]+)s? for ([\d.]+)s?",
                             " ".join(args))
            if not m:
                raise RequestError("use 'force <body> <N> dir <x> <y> <z> at <t> for <d>', e.g. "
                                   "'force hand 20 dir 0 0 -1 at 3 for 0.5'")
            body, mag, x, y, z, t0, d = m.groups()
            return self._set(force_pulse={"body": body, "magnitude_n": float(mag), "direction": [float(x), float(y),
                                          float(z)], "start_s": float(t0), "duration_s": float(d)}), {}
        if cmd == "noise":
            if args == ["off"]:
                return self._set(noise=None), {}
            seed = 0
            if "seed" in args:
                i = args.index("seed")
                seed = int(_num(args[i + 1] if i + 1 < len(args) else "x", "noise seed"))
                args = args[:i]
            qs, vs = _nums(args, 2, "noise q_std qd_std")
            return self._set(noise={"position_std": qs, "velocity_std": vs, "seed": seed}), {}
        if cmd == "check":
            from workbench.inspect import preflight
            checks = preflight(s.request)
            return [("ok   " if c["ok"] else "FAIL ") + f"{c['check']}: {c['detail']}" for c in checks], {}
        if cmd == "scan":
            from workbench.inspect import reach_scan
            r = reach_scan(s.request)
            c = r["counts"]
            return [f"workspace scan ({1e3 * r['step']:.0f} mm grid, top-down grasp): {c['found']} solution found, "
                    f"{c['blocked']} collision-blocked, {c['not_found']} search did not find a solution",
                    r["note"]], {"scan": r}
        if cmd in ("run", "start", "go"):
            s.act("run")
            return ["run requested: " + "; ".join(s.request.summary()[:2])], {}
        if cmd in ("pause", "resume", "stop", "reset"):
            s.act(cmd)
            return [f"{cmd} requested"], {}
        if cmd == "step":
            if args and args[0] in ("interval", "control"):
                s.act("step", unit="control")
                return ["step one controller interval requested"], {}
            n = int(_num(args[0], "step count")) if args else 1
            if not 1 <= n <= 5000:
                raise RequestError("step count must be 1-5000 physics steps")
            s.act("step", physics_steps=n)
            return [f"step {n} physics step(s) requested"], {}
        if cmd == "speed":
            v = 0.0 if args and args[0] in ("max", "fast") else _num(args[0] if args else "x", "speed")
            if v < 0 or v > 20:
                raise RequestError("speed must be in (0, 20] or 'max'")
            s.act("speed", value=v)
            return [f"live speed {'max' if v == 0 else f'{v}x real time'}"], {}
        if cmd == "replay":
            s.act("replay", run_id=args[0] if args else None)
            return [f"replay {'of the latest run' if not args else args[0]} requested (recorded data)"], {"panel": "result"}
        if cmd == "seek":
            s.act("seek", t=_num(args[0] if args else "x", "seek time"))
            return [f"seek to t = {args[0]} s (recorded data)"], {}
        if cmd == "play":
            s.act("replay_play", speed=_num(args[0], "replay speed") if args else 1.0)
            return ["replay playing"], {}
        if cmd == "live":
            s.act("live")
            return ["showing the live simulation"], {}
        if cmd == "show":
            panel = {"inspector": "mpc"}.get(args[-1] if args else "", args[0] if args else "")
            if panel not in PANELS:
                raise RequestError(f"panel must be one of {list(PANELS)}")
            return [f"showing the {panel} panel"], {"panel": panel}
        if cmd == "explain":
            return ["explanation shown in the Explain panel"], {"panel": "explain",
                                                                 "explain_t": _num(args[0], "time") if args else None}
        if cmd == "compare":
            what = args[0] if args else ""
            res = self.start_experiment(what)
            return res["output"], {"panel": "experiments"}
        if cmd == "camera" and args == ["reset"]:
            s.act("camera_reset")
            return ["camera reset"], {}
        raise RequestError(f"unknown command '{cmd}' (type 'help')")

    def start_experiment(self, name: str, cases=None) -> dict:
        from workbench.experiments import EXPERIMENTS, start
        aliases = {"observer": "observer", "payload": "payload", "controllers": "controllers", "controller": "controllers",
                   "contact": "contact", "grip": "grip", "grasp": "grip", "force": "force", "noise": "noise"}
        key = aliases.get(name)
        if key is None or key not in EXPERIMENTS:
            raise RequestError(f"unknown experiment '{name}'; available: {sorted(EXPERIMENTS)}")
        exp = start(self.session, key)
        return {"ok": True, "output": [f"experiment '{key}' queued: {exp['question']}",
                                       f"{len(exp['runs'])} sequential matched runs; see the Experiments panel"]}
