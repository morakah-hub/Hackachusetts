// HTML drag-and-drop from the library onto the 3D view. Browsers hide dataTransfer contents during dragover,
// so the dragged item is kept here as well.
import type { LibraryItem } from "../api/types";

export const TEMPLATE_MIME = "application/x-workcell-template";

let dragging: LibraryItem | null = null;

export const libraryDrag = {
  start(item: LibraryItem) {
    dragging = item;
  },
  end() {
    dragging = null;
  },
  get current(): LibraryItem | null {
    return dragging;
  },
};
