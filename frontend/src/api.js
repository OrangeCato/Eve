const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function eveChat({ question, objects = [], face_cues = [], history = [] }) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, objects, face_cues, history }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json(); // { answer }
}