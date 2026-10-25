import type { Vec3 } from "../api/types";
import { liftIssues } from "../lib/geom";
import { useStudio } from "../store";

/** Live checks of an object definition against the robot's limits (payload, gripper, height). */
export function ObjectChecks({ size, mass }: { size: Vec3; mass: number }) {
  const config = useStudio((s) => s.config);
  if (!config) return null;
  const lim = { ...config.robot.limits, gap_m: config.gap_m };
  const payload = config.robot.payload_kg;
  const issues = liftIssues(size, mass, payload, lim);
  const narrow = Math.min(size[0], size[1]);
  const okWidth = [size[0], size[1]].some((s) => s >= lim.min_grasp_width_m && s <= lim.max_grasp_width_m);
  const rows = [
    {
      ok: mass <= payload && mass >= lim.min_mass_kg,
      label: "Weight",
      text: `${mass.toFixed(2)} kg of ${payload.toFixed(1)} kg payload`,
    },
    {
      ok: okWidth,
      label: "Gripper",
      text: okWidth
        ? `Fits between the fingers (${Math.round(Math.min(...[size[0], size[1]].filter((s) => s >= lim.min_grasp_width_m && s <= lim.max_grasp_width_m)) * 1000)} mm, max ${Math.round(lim.max_grasp_width_m * 1000)} mm)`
        : `${Math.round(narrow * 1000)} mm at its narrowest; the fingers take ${Math.round(lim.min_grasp_width_m * 1000)}–${Math.round(lim.max_grasp_width_m * 1000)} mm`,
    },
    {
      ok: size[2] >= lim.min_height_m && size[2] <= lim.max_height_m,
      label: "Height",
      text: `${Math.round(size[2] * 1000)} mm (supported ${Math.round(lim.min_height_m * 1000)}–${Math.round(lim.max_height_m * 1000)} mm)`,
    },
  ];
  return (
    <div className="checks" aria-live="polite">
      <ul>
        {rows.map((r) => (
          <li key={r.label} className={r.ok ? "check-ok" : "check-bad"}>
            <span className="check-icon" aria-hidden="true">{r.ok ? "✓" : "✕"}</span>
            <span className="check-label">{r.label}</span>
            <span>{r.text}</span>
          </li>
        ))}
      </ul>
      <p className={issues.length ? "check-summary check-summary-bad" : "check-summary"}>
        {issues.length
          ? `The ${config.robot.name} cannot pick this up. You can still put it on the table as an obstacle.`
          : `The ${config.robot.name} can pick this up.`}
      </p>
    </div>
  );
}
