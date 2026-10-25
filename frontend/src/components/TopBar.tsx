import { isRunActive, useStudio } from "../store";

const ROBOTS = [
  { id: "panda", name: "Franka Emika Panda", available: true },
  { id: "ur5e", name: "Universal Robots UR5e (planned)", available: false },
  { id: "xarm", name: "UFACTORY xArm 6 (planned)", available: false },
];

export function TopBar() {
  const connection = useStudio((s) => s.connection);
  const run = useStudio((s) => s.live?.run ?? null);
  const running = useStudio((s) => isRunActive(s));
  const robot = useStudio((s) => s.config?.robot);

  const status = !run ? "Ready" : running ? (run.state === "paused" ? "Paused" : "Running a task") : run.state === "succeeded" ? "Task done" : run.state === "failed" ? "Task failed" : "Stopped";

  return (
    <header className="topbar">
      <div className="brand">
        <img className="brand-logo" src="/armstrong-logo.svg" alt="" aria-hidden="true" />
        <span className="brand-name">ArmStrong</span>
      </div>
      <label className="robot-select">
        <span className="sr-only">Robot</span>
        <select value={robot?.id ?? "panda"} onChange={() => undefined} aria-label="Robot">
          {ROBOTS.map((r) => (
            <option key={r.id} value={r.id} disabled={!r.available}>
              {r.name}
            </option>
          ))}
        </select>
      </label>
      <div className="topbar-spacer" />
      <span className={`status-pill${running ? " status-live" : run?.state === "failed" ? " status-bad" : ""}`} role="status">
        {running && <span className="pulse" aria-hidden="true" />}
        {status}
      </span>
      <span className={`conn conn-${connection}`} title={connection === "online" ? "Connected to the simulation server" : "No connection to the simulation server"}>
        <span className="conn-dot" aria-hidden="true" />
        {connection === "online" ? "Connected" : connection === "connecting" ? "Connecting…" : "Offline"}
      </span>
    </header>
  );
}
