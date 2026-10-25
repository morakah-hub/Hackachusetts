// Small line icons (20 x 20, currentColor) for the task cards.
const common = { width: 20, height: 20, viewBox: "0 0 20 20", fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };

export function TaskIcon({ id }: { id: string }) {
  switch (id) {
    case "move":
      return (
        <svg {...common}>
          <rect x="2.5" y="11" width="5" height="5" rx="1" />
          <path d="M9 6.5c2.5-2.5 5.5-2 7 0" />
          <path d="M14.5 4.2 16.3 6.6 13.6 7.3" />
          <rect x="12.5" y="11" width="5" height="5" rx="1" strokeDasharray="1.6 1.6" />
        </svg>
      );
    case "hand_over":
      return (
        <svg {...common}>
          <rect x="7.5" y="3" width="5" height="5" rx="1" />
          <path d="M3 13.5h4.5l2 1.5h3.5a1.5 1.5 0 0 0 0-3H10" />
          <path d="M3 17h9.5l4.5-3.5a1.4 1.4 0 0 0-1.9-2L12.5 13" />
        </svg>
      );
    case "swap":
      return (
        <svg {...common}>
          <path d="M4 7h12M13 4l3 3-3 3" />
          <path d="M16 13H4M7 10l-3 3 3 3" />
        </svg>
      );
    case "stack":
      return (
        <svg {...common}>
          <rect x="6.5" y="3" width="7" height="5" rx="1" />
          <rect x="4" y="10" width="12" height="7" rx="1" />
        </svg>
      );
    case "line_up":
      return (
        <svg {...common}>
          <rect x="2.5" y="8" width="4" height="8" rx="0.8" />
          <rect x="8" y="8" width="4" height="8" rx="0.8" />
          <rect x="13.5" y="8" width="4" height="8" rx="0.8" />
          <path d="M2.5 4.5h15" />
        </svg>
      );
    case "sort_weight":
      return (
        <svg {...common}>
          <rect x="2.5" y="12" width="4" height="4" rx="0.8" />
          <rect x="8" y="9.5" width="4" height="6.5" rx="0.8" />
          <rect x="13.5" y="6" width="4" height="10" rx="0.8" />
          <path d="M3 4.5h6M13 4.5h4M15 2.5v4" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="3" />
        </svg>
      );
  }
}
