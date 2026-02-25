import { useEffect, useState } from "react";

// Bump version so it shows again even if you dismissed v1 before
const KEY = "demo_notice_dismissed_v2";

export default function TopNotice({ onVisibilityChange }) {
  // Initialize immediately from localStorage (prevents flicker and “doesn’t show” confusion)
  const [hidden, setHidden] = useState(() => {
    try {
      return localStorage.getItem(KEY) === "1";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    onVisibilityChange?.(!hidden);
  }, [hidden, onVisibilityChange]);

  if (hidden) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        zIndex: 9999,
        width: "100%",
        background: "rgba(0,0,0,0.85)",
        color: "rgba(255,255,255,0.92)",
        borderBottom: "1px solid rgba(255,255,255,0.12)",
        backdropFilter: "blur(10px)",
      }}
    >
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          padding: "10px 14px",
          display: "flex",
          gap: 12,
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: 12,
          lineHeight: 1.3,
        }}
      >
        <div>
          <b>Note:</b> This demo runs on free-tier cloud infrastructure. If inactive,
          the server may take ~20–40 seconds to wake.
        </div>

        <button
          onClick={() => {
            try {
              localStorage.setItem(KEY, "1");
            } catch {}
            setHidden(true);
          }}
          style={{
            border: "1px solid rgba(255,255,255,0.18)",
            background: "rgba(255,255,255,0.06)",
            color: "rgba(255,255,255,0.92)",
            borderRadius: 999,
            padding: "6px 10px",
            cursor: "pointer",
            fontSize: 12,
            whiteSpace: "nowrap",
          }}
          title="Hide"
        >
          Got it
        </button>
      </div>
    </div>
  );
}