import { useEffect } from "react";
import { CustomObjectDialog } from "./components/CustomObjectDialog";
import { HelpDialog } from "./components/HelpDialog";
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
  const helpOpen = useStudio((s) => s.helpOpen);

  useEffect(() => {
    useStudio.getState().init();
  }, []);

  if (!config) {
    return (
      <div className="splash" role="status">
        <span className="spinner" aria-hidden="true" />
        <p>{loadError ? "Waiting for the simulation server…" : "Starting Armstrong Studio…"}</p>
        {loadError && (
          <p className="muted small">
            {loadError} Start it with <code>Start.exe</code> or <code>start.bat</code> (see README).
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
      {helpOpen && !dialog && <HelpDialog />}
      <Toasts />
    </div>
  );
}
