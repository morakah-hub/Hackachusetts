import { useStudio } from "../store";
import { Dialog } from "./Dialog";

/** "How to use" guide: shown on the first visit and from the ? button in the top bar. */
export function HelpDialog() {
  const st = useStudio.getState();
  const close = () => st.setHelp(false);
  const tryIt = () => {
    st.setTask("assemble");
    close();
  };
  return (
    <Dialog title="Welcome to Armstrong Studio" onClose={close} wide>
      <p>
        A simulated <strong>Franka Emika Panda</strong> robot arm works on a table in front of you. It finds objects with a 3D camera, picks them up
        with real physics (grip and friction, nothing is glued to the hand) and puts them where you ask.
      </p>
      <ol className="help-steps">
        <li>
          <strong>Set up the table.</strong> The left panel lists what is on the table. Drag items from the <em>Object library</em> onto the table, or
          press <em>Assembly kit</em> for the parts of a bridge and a tower. Drag objects to move them; <kbd>Q</kbd>/<kbd>E</kbd> rotate the selected
          one, <kbd>Delete</kbd> removes it.
        </li>
        <li>
          <strong>Choose a task.</strong> In the right panel pick <em>Assemble</em>, <em>Move</em>, <em>Stack</em>, <em>Swap</em>, <em>Line up</em> or{" "}
          <em>Sort by weight</em>. Fill in what it asks for: click objects or spots on the table in the 3D view.
        </li>
        <li>
          <strong>Check, then Run.</strong> <em>Check</em> plans every move and shows it as arrows in the 3D view. <em>Run</em> starts the robot. You can
          pause, stop or change the speed, and <em>Undo</em> puts the objects back afterwards.
        </li>
      </ol>
      <p className="muted small">
        Tips: drag in the 3D view to orbit, scroll to zoom, right-drag to pan. The shaded area is where the arm can reach and the camera can see.
        The <em>Camera</em> panel shows what the robot sees.
      </p>
      <div className="dialog-actions">
        <button className="btn" onClick={close}>
          Explore on my own
        </button>
        <button className="btn btn-primary" onClick={tryIt}>
          Try it: assemble a bridge
        </button>
      </div>
    </Dialog>
  );
}
