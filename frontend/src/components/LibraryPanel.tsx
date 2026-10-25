import type { LibraryItem } from "../api/types";
import { libraryDrag, TEMPLATE_MIME } from "../lib/dnd";
import { dims, kg } from "../lib/format";
import { isRunActive, useStudio } from "../store";

const SHORT_PROBLEM: Record<string, string> = {
  too_heavy: "Over payload",
  too_wide: "Too wide to grip",
  too_thin: "Too thin to grip",
  no_grip_side: "No side fits the gripper",
  too_short: "Too short",
  too_tall: "Too tall",
};

function ItemCard({ item, disabled }: { item: LibraryItem; disabled: boolean }) {
  const st = useStudio.getState();
  const issue = item.problems.length ? item.problems.map((p) => SHORT_PROBLEM[p.code] ?? p.message).join(" · ") : "";
  const detail = item.problems.map((p) => p.message).join(" ");
  return (
    <li
      className={`lib-card${item.can_lift ? "" : " lib-card-warn"}`}
      draggable={!disabled}
      onDragStart={(e) => {
        libraryDrag.start(item);
        e.dataTransfer.setData(TEMPLATE_MIME, item.id);
        e.dataTransfer.setData("text/plain", item.name);
        e.dataTransfer.effectAllowed = "copy";
      }}
      onDragEnd={() => libraryDrag.end()}
      title={disabled ? "Editing is paused while a task runs" : detail ? `${detail} It can still stand on the table as an obstacle.` : "Drag onto the table, or press Add"}
    >
      <span className="lib-thumb" style={{ background: item.color }} aria-hidden="true">
        {item.asset_id ? "3D" : ""}
      </span>
      <span className="lib-text">
        <span className="lib-name">{item.name}</span>
        <span className="lib-meta">
          {dims(item.size)} · {kg(item.mass)}
        </span>
        {issue && <span className="lib-issue">Obstacle only: {issue.toLowerCase()}</span>}
      </span>
      <span className="lib-actions">
        <button className="btn btn-sm" disabled={disabled} onClick={() => st.addFromLibrary(item.id)} aria-label={`Add ${item.name} to the table`}>
          Add
        </button>
        {!item.builtin && (
          <button className="icon-btn" disabled={disabled} onClick={() => st.deleteLibraryItem(item.id)} aria-label={`Delete ${item.name} from the library`} title="Delete from the library">
            🗑
          </button>
        )}
      </span>
    </li>
  );
}

export function LibraryPanel() {
  const library = useStudio((s) => s.library);
  const scene = useStudio((s) => s.scene);
  const selectedId = useStudio((s) => s.selectedId);
  const running = useStudio((s) => isRunActive(s));
  const st = useStudio.getState();
  const builtin = library.filter((i) => i.builtin);
  const mine = library.filter((i) => !i.builtin);

  return (
    <aside className="panel panel-left" aria-label="Objects">
      <section className="panel-section">
        <div className="section-head">
          <h2>On the table</h2>
          <span className="count">{scene?.objects.length ?? 0}</span>
        </div>
        {scene && scene.objects.length > 0 ? (
          <ul className="scene-list">
            {scene.objects.map((o) => (
              <li key={o.id}>
                <button
                  className={`scene-item${o.id === selectedId ? " scene-item-selected" : ""}`}
                  onClick={() => st.select(o.id === selectedId ? null : o.id)}
                  onMouseEnter={() => st.setHover(o.id)}
                  onMouseLeave={() => st.setHover(null)}
                  aria-pressed={o.id === selectedId}
                >
                  <span className="swatch" style={{ background: o.color }} aria-hidden="true" />
                  <span className="scene-name">{o.name}</span>
                  {o.state === "tipped" || o.state === "fallen" ? (
                    <span className="chip chip-bad">{o.state === "tipped" ? "Tipped" : "Fell off"}</span>
                  ) : o.state === "stacked" ? (
                    <span className="chip">Stacked</span>
                  ) : !o.can_lift ? (
                    <span className="chip chip-warn">Obstacle</span>
                  ) : null}
                  <span className="scene-mass">{kg(o.mass)}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">The table is empty.</p>
        )}
        <div className="row-actions">
          <button className="btn btn-ghost btn-sm" disabled={running} onClick={() => st.resetScene("default")}>
            Sample layout
          </button>
          <button className="btn btn-ghost btn-sm" disabled={running || !scene?.objects.length} onClick={() => st.resetScene("empty")}>
            Clear table
          </button>
        </div>
      </section>

      <section className="panel-section panel-grow">
        <div className="section-head">
          <h2>Object library</h2>
        </div>
        <div className="add-buttons">
          <button className="btn btn-primary" disabled={running} onClick={() => st.openDialog("upload")}>
            Upload 3D model
          </button>
          <button className="btn" disabled={running} onClick={() => st.openDialog("custom")}>
            New box
          </button>
        </div>
        <p className="muted small">Drag an item onto the table, or press Add to place it in a free spot.</p>
        {mine.length > 0 && (
          <>
            <h3 className="sub-head">Your objects</h3>
            <ul className="lib-list">
              {mine.map((i) => (
                <ItemCard key={i.id} item={i} disabled={running} />
              ))}
            </ul>
          </>
        )}
        <h3 className="sub-head">Samples</h3>
        <ul className="lib-list">
          {builtin.map((i) => (
            <ItemCard key={i.id} item={i} disabled={running} />
          ))}
        </ul>
      </section>
    </aside>
  );
}
