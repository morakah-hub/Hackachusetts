import { useMemo, useRef, useState } from "react";
import type * as THREE from "three";
import { api, ApiError } from "../api/client";
import type { MeshPlacement, Vec3 } from "../api/types";
import { liftIssues } from "../lib/geom";
import { useStudio } from "../store";
import { fitBox, formatOf, guessUnits, parseMesh, type MeshFormat } from "../three/meshAssets";
import { Dialog } from "./Dialog";
import { MeshPreview } from "./MeshPreview";
import { ObjectChecks } from "./ObjectChecks";

const UNITS: [number, string][] = [
  [0.001, "millimetres"],
  [0.01, "centimetres"],
  [0.0254, "inches"],
  [1, "metres"],
];

interface Parsed {
  file: File;
  format: MeshFormat;
  object: THREE.Object3D;
}

export function UploadDialog() {
  const config = useStudio((s) => s.config);
  const st = useStudio.getState();
  const close = () => st.openDialog(null);
  const [parsed, setParsed] = useState<Parsed | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [units, setUnits] = useState(0.001);
  const [extraScale, setExtraScale] = useState("1");
  const [up, setUp] = useState<"y" | "z">("y");
  const [mass, setMass] = useState("0.3");
  const [color, setColor] = useState("#b8bec6");
  const [dragOver, setDragOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const maxMb = config?.limits.max_asset_mb ?? 30;

  const scale = units * (parseFloat(extraScale) > 0 ? parseFloat(extraScale) : 1);
  const fit = useMemo(() => (parsed ? fitBox(parsed.object, scale, up) : null), [parsed, scale, up]);
  const size: Vec3 = fit ? fit.size : [0.05, 0.05, 0.05];
  const placement: MeshPlacement | null = parsed && fit ? { format: parsed.format, scale, up, offset: fit.offset } : null;
  const massNum = parseFloat(mass);

  const onFile = async (file: File | undefined) => {
    setError("");
    if (!file) return;
    const format = formatOf(file.name);
    if (!format) return setError("Choose a .glb, .gltf, .obj or .stl file.");
    if (file.size > maxMb * 1024 * 1024) return setError(`The file is larger than ${maxMb} MB.`);
    setBusy(true);
    try {
      const object = await parseMesh(await file.arrayBuffer(), format, color);
      let meshes = 0;
      object.traverse((c) => {
        if ((c as THREE.Mesh).isMesh) meshes++;
      });
      if (meshes === 0) throw new Error("The file contains no meshes.");
      const raw = fitBox(object, 1, format === "stl" ? "z" : "y");
      setUnits(guessUnits(Math.max(...raw.size)));
      setUp(format === "stl" ? "z" : "y");
      setExtraScale("1");
      setName(file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").slice(0, 40) || "Uploaded object");
      setParsed({ file, format, object });
    } catch (e) {
      setParsed(null);
      setError((e as Error).message || "Could not read the file.");
    } finally {
      setBusy(false);
    }
  };

  const save = async (alsoPlace: boolean) => {
    if (!parsed || !placement || !config) return;
    if (!Number.isFinite(massNum) || massNum <= 0) return setError("Enter the object's weight in kg.");
    setBusy(true);
    setError("");
    try {
      const asset = await api.uploadAsset(parsed.file);
      const item = await st.addLibraryItem({ name: name.trim() || "Uploaded object", size, mass: massNum, color, asset_id: asset.id, mesh: placement });
      if (item) {
        st.toast("success", `${item.name} was added to the library.`);
        if (alsoPlace) await st.addFromLibrary(item.id);
        close();
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const issues = config ? liftIssues(size, massNum || 0, config.robot.payload_kg, { ...config.robot.limits, gap_m: config.gap_m }) : [];
  const tooLarge = config ? Math.max(size[0], size[1]) > config.robot.limits.max_footprint_m : false;

  return (
    <Dialog title="Upload a 3D model" onClose={close} wide>
      {!parsed ? (
        <div
          className={`dropzone${dragOver ? " dropzone-over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            onFile(e.dataTransfer.files[0]);
          }}
        >
          <p className="dropzone-title">Drop a model file here</p>
          <p className="muted">.glb, .gltf (self-contained), .obj or .stl · up to {maxMb} MB</p>
          <button className="btn btn-primary" onClick={() => input.current?.click()} disabled={busy}>
            {busy ? "Reading…" : "Choose a file"}
          </button>
          <input ref={input} type="file" accept=".glb,.gltf,.obj,.stl" hidden onChange={(e) => onFile(e.target.files?.[0])} />
          <button
            className="btn btn-ghost btn-sm"
            disabled={busy}
            onClick={async () => {
              try {
                const res = await fetch("/samples/drinks-can.obj");
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                await onFile(new File([await res.blob()], "drinks-can.obj", { type: "text/plain" }));
                setMass("0.35");
              } catch (e) {
                setError(`Could not load the sample: ${(e as Error).message}`);
              }
            }}
          >
            No file at hand? Try a sample drinks can
          </button>
          <p className="muted small">
            The simulation handles the object as a solid box around the model (its collision shape). Enter the real weight
            below; it is checked against the robot's payload.
          </p>
        </div>
      ) : (
        <div className="upload-grid">
          <div>
            <MeshPreview object={parsed.object} placement={placement} size={size} />
            <p className="muted small center">
              {parsed.file.name} · teal box = collision shape {Math.round(size[0] * 1000)} × {Math.round(size[1] * 1000)} × {Math.round(size[2] * 1000)} mm
            </p>
          </div>
          <div className="form">
            <label className="field">
              <span>Name</span>
              <input value={name} maxLength={40} onChange={(e) => setName(e.target.value)} />
            </label>
            <div className="field-row">
              <label className="field">
                <span>File units</span>
                <select value={units} onChange={(e) => setUnits(parseFloat(e.target.value))}>
                  {UNITS.map(([v, l]) => (
                    <option key={l} value={v}>
                      {l}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Extra scale</span>
                <input type="number" min={0.01} step={0.1} value={extraScale} onChange={(e) => setExtraScale(e.target.value)} />
              </label>
            </div>
            <fieldset className="field">
              <legend>Which way is up in the file?</legend>
              <div className="segmented" role="radiogroup">
                {(["y", "z"] as const).map((a) => (
                  <button key={a} role="radio" aria-checked={up === a} className={up === a ? "seg seg-on" : "seg"} onClick={() => setUp(a)}>
                    {a.toUpperCase()} axis up
                  </button>
                ))}
              </div>
            </fieldset>
            <div className="field-row">
              <label className="field">
                <span>Weight (kg)</span>
                <input type="number" min={0.01} step={0.05} value={mass} onChange={(e) => setMass(e.target.value)} />
              </label>
              {(parsed.format === "obj" || parsed.format === "stl") && (
                <label className="field">
                  <span>Colour</span>
                  <input type="color" value={color} onChange={(e) => setColor(e.target.value)} />
                </label>
              )}
            </div>
            <ObjectChecks size={size} mass={Number.isFinite(massNum) ? massNum : 0} />
            {tooLarge && <p className="text-bad">The model is larger than the table accepts. Check the units.</p>}
            {error && <p className="text-bad" role="alert">{error}</p>}
            <div className="dialog-actions">
              <button className="btn btn-ghost" onClick={() => setParsed(null)} disabled={busy}>
                Choose another file
              </button>
              <button className="btn" onClick={() => save(false)} disabled={busy || tooLarge}>
                Add to library
              </button>
              <button className="btn btn-primary" onClick={() => save(true)} disabled={busy || tooLarge}>
                {busy ? "Uploading…" : issues.length ? "Add as obstacle" : "Add and place on table"}
              </button>
            </div>
          </div>
        </div>
      )}
      {!parsed && error && <p className="text-bad" role="alert">{error}</p>}
    </Dialog>
  );
}
