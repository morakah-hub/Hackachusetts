import { useState } from "react";
import type { Vec3 } from "../api/types";
import { useStudio } from "../store";
import { Dialog } from "./Dialog";
import { ObjectChecks } from "./ObjectChecks";

/** Define a box-shaped object by its dimensions and weight. */
export function CustomObjectDialog() {
  const st = useStudio.getState();
  const close = () => st.openDialog(null);
  const [name, setName] = useState("My box");
  const [w, setW] = useState("50");
  const [d, setD] = useState("50");
  const [h, setH] = useState("80");
  const [mass, setMass] = useState("0.3");
  const [color, setColor] = useState("#6aa6d9");
  const [busy, setBusy] = useState(false);
  const size: Vec3 = [parseFloat(w) / 1000 || 0, parseFloat(d) / 1000 || 0, parseFloat(h) / 1000 || 0];
  const massNum = parseFloat(mass) || 0;

  const save = async (place: boolean) => {
    setBusy(true);
    const item = await st.addLibraryItem({ name: name.trim() || "My box", size, mass: massNum, color, asset_id: null, mesh: null });
    setBusy(false);
    if (item) {
      if (place) await st.addFromLibrary(item.id);
      close();
    }
  };

  return (
    <Dialog title="New box-shaped object" onClose={close}>
      <div className="form">
        <label className="field">
          <span>Name</span>
          <input value={name} maxLength={40} onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="field-row three">
          <label className="field">
            <span>Width (mm)</span>
            <input type="number" min={5} value={w} onChange={(e) => setW(e.target.value)} />
          </label>
          <label className="field">
            <span>Depth (mm)</span>
            <input type="number" min={5} value={d} onChange={(e) => setD(e.target.value)} />
          </label>
          <label className="field">
            <span>Height (mm)</span>
            <input type="number" min={5} value={h} onChange={(e) => setH(e.target.value)} />
          </label>
        </div>
        <div className="field-row">
          <label className="field">
            <span>Weight (kg)</span>
            <input type="number" min={0.01} step={0.05} value={mass} onChange={(e) => setMass(e.target.value)} />
          </label>
          <label className="field">
            <span>Colour</span>
            <input type="color" value={color} onChange={(e) => setColor(e.target.value)} />
          </label>
        </div>
        <ObjectChecks size={size} mass={massNum} />
        <div className="dialog-actions">
          <button className="btn" onClick={() => save(false)} disabled={busy}>
            Add to library
          </button>
          <button className="btn btn-primary" onClick={() => save(true)} disabled={busy}>
            Add and place on table
          </button>
        </div>
      </div>
    </Dialog>
  );
}
