import { useEffect } from "react";
import { CustomObjectDialog } from "./components/CustomObjectDialog";
import { LibraryPanel } from "./components/LibraryPanel";
import { TaskPanel } from "./components/TaskPanel";
import { Toasts } from "./components/Toasts";
import { TopBar } from "./components/TopBar";
import { UploadDialog } from "./components/UploadDialog";
import { Viewport } from "./components/Viewport";
import { useStudio } from "./store";

export default function App() {
  const config = useStudio((s) => s.config);
  const loadError = useStudio((s) => s.loadError);
  const dialog = useStudio((s) => s.dialog);

  useEffect(() => {
    useStudio.getState().init();
  }, []);

  if (!config) {
    return (
      <div className="splash" role="status">
        <span className="spinner" aria-hidden="true" />
        <p>{loadError ? "Waiting for the simulation server…" : "Starting ArmStrong…"}</p>
        {loadError && (
          <p className="muted small">
            {loadError} Start it with <code>studio.py</code> (see README).
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      <TopBar />
      <main className="layout">
        <LibraryPanel />
        <Viewport />
        <TaskPanel />
      </main>
      {dialog === "upload" && <UploadDialog />}
      {dialog === "custom" && <CustomObjectDialog />}
      <Toasts />
    </div>
  );
}
