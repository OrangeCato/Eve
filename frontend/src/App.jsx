import { useEffect, useMemo, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import TopNotice from "./components/TopNotice";

function useVoice() {
  const canSpeak = "speechSynthesis" in window;
  const speak = (text) => {
    if (!canSpeak) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
  };
  return { canSpeak, speak };
}

function useSpeechIn() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const canListen = !!SpeechRecognition;

  const listenOnce = () =>
    new Promise((resolve, reject) => {
      if (!canListen) return reject(new Error("SpeechRecognition not supported"));
      const r = new SpeechRecognition();
      r.lang = "en-US";
      r.interimResults = false;
      r.maxAlternatives = 1;

      r.onresult = (e) => resolve(e.results?.[0]?.[0]?.transcript || "");
      r.onerror = (e) => reject(new Error(e.error || "Speech recognition error"));
      r.onend = () => {};
      r.start();
    });

  return { canListen, listenOnce };
}

// MediaPipe indices
const IDX = { wrist: 0, thumbTip: 4, indexTip: 8, middleTip: 12, ringTip: 16, pinkyTip: 20 };

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}
function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}
function isPalm(lm) {
  const w = lm[IDX.wrist];
  const tips = [lm[IDX.indexTip], lm[IDX.middleTip], lm[IDX.ringTip], lm[IDX.pinkyTip]];
  const avg = tips.reduce((s, p) => s + dist(p, w), 0) / tips.length;
  return avg > 0.33;
}
function pinchStrength(lm) {
  const d = dist(lm[IDX.thumbTip], lm[IDX.indexTip]);
  return 1 - Math.min(1, Math.max(0, (d - 0.02) / (0.10 - 0.02)));
}

function formatTime() {
  return new Date().toLocaleTimeString();
}
function formatDay() {
  return new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}
function num(n, digits = 0) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}

// ---- Synthetic health generator ----
function seededRand(seed) {
  let t = seed + 0x6d2b79f5;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function makeHealthSynthetic() {
  const daySeed = Math.floor(Date.now() / (1000 * 60 * 60 * 24));
  const r1 = seededRand(daySeed + 11);
  const r2 = seededRand(daySeed + 22);
  const r3 = seededRand(daySeed + 33);
  const r4 = seededRand(daySeed + 44);

  const sleepHours = 6.0 + r1 * 2.6;
  const efficiency = 0.70 + r2 * 0.26;
  const restingHR = 54 + Math.round(r3 * 12);
  const currentHR = restingHR + 6 + Math.round(r4 * 22);
  const bedtime = 22 + Math.floor(r2 * 2);
  const wakeHour = 6 + Math.floor(r1 * 2);

  const hrDelta = Math.max(0, currentHR - restingHR);
  const sleepPenalty = (1 - efficiency) * 60;
  const stressScore = Math.round(clamp01((hrDelta / 30) * 0.55 + (sleepPenalty / 60) * 0.45) * 100);
  const stressLabel = stressScore < 30 ? "Low" : stressScore < 60 ? "Moderate" : stressScore < 80 ? "High" : "Very High";

  return { sleepHours, efficiency, restingHR, currentHR, wakeHour, bedtime, stressScore, stressLabel };
}

// ---- Weather (Open-Meteo) ----
async function getGeoPosition({ timeoutMs = 4500 } = {}) {
  if (!("geolocation" in navigator)) return null;
  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      () => resolve(null),
      { enableHighAccuracy: false, timeout: timeoutMs, maximumAge: 60_000 }
    );
  });
}
async function geocodeName(lat, lon) {
  // Lightweight: we just label it and keep timezone from browser
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "auto";
  if (lat != null && lon != null) return { name: "Local area", timezone: tz };
  return { name: "Local area", timezone: tz };
}
async function fetchWeather(lat, lon, timezone) {
  const tz = encodeURIComponent(timezone || "auto");
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
    `&current=temperature_2m,apparent_temperature,precipitation,wind_speed_10m` +
    `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=${tz}`;

  const r = await fetch(url);
  if (!r.ok) throw new Error("Weather fetch failed");
  return r.json();
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const idleCanvasRef = useRef(null);

  const { speak } = useVoice();
  const { canListen, listenOnce } = useSpeechIn();

  const [camError, setCamError] = useState("");
  const [ready, setReady] = useState(false);

  // notice bar (same pattern as EasyStonks)
  const [noticeVisible, setNoticeVisible] = useState(false);

  // gesture layer
  const [online, setOnline] = useState(false);
  const onlineRef = useRef(false);
  useEffect(() => {
    onlineRef.current = online;
  }, [online]);

  const [cursor, setCursor] = useState({ x: 0.5, y: 0.5 });
  const [pinchP, setPinchP] = useState(0);
  const [hoverId, setHoverId] = useState(null);

  const [toast, setToast] = useState("Stand by…");
  const [log, setLog] = useState([]);

  // app layer
  const [activePanel, setActivePanel] = useState(null);
  const [panelPinned, setPanelPinned] = useState(true);

  // Today
  const [weather, setWeather] = useState({ status: "idle", data: null, error: "" });
  const [place, setPlace] = useState({ name: "—", timezone: "auto", lat: null, lon: null });

  // Health
  const [health, setHealth] = useState(() => makeHealthSynthetic());

  // Scan
  const [scan, setScan] = useState({ status: "idle", items: [], error: "" });
  const cocoRef = useRef(null);

  // Mood
  const [mood, setMood] = useState({ status: "idle", label: "—", confidence: 0 });

  // Ask
  const [askInput, setAskInput] = useState("");
  const [askBusy, setAskBusy] = useState(false);
  const [history, setHistory] = useState([]);

  // gesture click system
  const draggingRef = useRef(false);
  const lastHoverDuringDragRef = useRef(null);
  const lastClickAtRef = useRef(0);
  const CLICK_COOLDOWN_MS = 450;

  const cursorRef = useRef({ x: 0.5, y: 0.5 });
  const cursorVelRef = useRef({ x: 0, y: 0 });

  const rectsRef = useRef([]);
  const [rectsEpoch, setRectsEpoch] = useState(0);

  // auto-brief guards (reset on every panel open)
  const briefedTodayRef = useRef(false);
  const briefedHealthRef = useRef(false);
  const briefedScanRef = useRef(false);
  const briefedMoodRef = useRef(false);

  function addLog(line) {
    setLog((prev) => [{ t: formatTime(), line }, ...prev].slice(0, 8));
  }
  function showToast(text) {
    setToast(text);
  }

  const API_BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

  const buttons = useMemo(
    () => [
      { id: "today", title: "Today's Info", subtitle: "Weather • reminders • status" },
      { id: "health", title: "Health Snapshot", subtitle: "Sleep • HR • stress (demo)" },
      { id: "scan", title: "Quick Scan", subtitle: "Object recognition (local)" },
      { id: "mood", title: "Mood Scan", subtitle: "Expression cues (demo)" },
      { id: "ask", title: "Ask EVE", subtitle: "Voice + chat" },
    ],
    []
  );

  function registerRect(id, el) {
    if (!el) return;
    const r = el.getBoundingClientRect();
    const rects = rectsRef.current.filter((x) => x.id !== id);
    rects.push({ id, left: r.left, top: r.top, right: r.right, bottom: r.bottom });
    rectsRef.current = rects;
  }

  function clearRects() {
    rectsRef.current = [];
    setRectsEpoch((x) => x + 1);
  }

  function hitTestCursor(cx01, cy01) {
    const rects = rectsRef.current || [];
    const x = cx01 * window.innerWidth;
    const y = cy01 * window.innerHeight;
    for (const r of rects) {
      if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return r.id;
    }
    return null;
  }

  function todayBriefingText() {
    const data = weather.data;
    const current = data?.current;
    const daily = data?.daily;

    const temp = current?.temperature_2m;
    const feels = current?.apparent_temperature;
    const wind = current?.wind_speed_10m;

    const hi = daily?.temperature_2m_max?.[0];
    const lo = daily?.temperature_2m_min?.[0];
    const rain = daily?.precipitation_sum?.[0];

    const reminders = ["Portfolio: verify links + live demos", "Recruiter mode: keep services pinned", "Next: connect vision + mood model"];

    return (
      `Briefing for ${formatDay()}. ` +
      `It's ${num(temp, 1)} degrees, feels like ${num(feels, 1)}. Wind ${num(wind, 0)} kilometers per hour. ` +
      `Forecast: high ${num(hi, 1)}, low ${num(lo, 1)}, rain ${num(rain, 1)} millimeters. ` +
      `Reminders: ${reminders.join(". ")}.`
    );
  }

  function healthBriefingText() {
    const eff = Math.round(health.efficiency * 100);
    const readiness = eff >= 85 ? "high" : eff >= 75 ? "normal" : "low";
    return (
      `Health summary. Sleep ${num(health.sleepHours, 1)} hours, efficiency ${eff} percent. ` +
      `Resting heart rate ${health.restingHR}. Current heart rate ${health.currentHR}. ` +
      `Stress ${health.stressLabel}, score ${health.stressScore} out of 100. Readiness is ${readiness}.`
    );
  }

  function scanBriefingText(items) {
    if (!items?.length) return "Scan complete. I did not detect any confident objects.";
    const top = items.slice(0, 4).map((x) => `${x.label} ${Math.round(x.score * 100)} percent`);
    return `Scan complete. I see: ${top.join(", ")}.`;
  }

  function moodBriefingText(m) {
    return `Mood scan complete. You appear ${m.label}. Confidence ${Math.round(m.confidence * 100)} percent.`;
  }

  async function runScanOnce() {
    setScan({ status: "loading", items: [], error: "" });

    try {
      if (!cocoRef.current) {
        cocoRef.current = await cocoSsd.load();
      }
      const m = cocoRef.current;
      const v = videoRef.current;
      if (!v || v.readyState < 2) throw new Error("Camera not ready");

      const preds = await m.detect(v);
      const top = preds
        .filter((p) => p.score >= 0.55)
        .slice(0, 8)
        .map((p) => ({ label: p.class, score: p.score, box: p.bbox }));

      setScan({ status: "ready", items: top, error: "" });
    } catch (e) {
      setScan({ status: "error", items: [], error: e?.message || "Scan failed" });
    }
  }

  function runMoodOnce() {
    const t = new Date();
    const seed = t.getHours() * 60 + t.getMinutes();
    const r = seededRand(seed + 999);

    const options = [
      { label: "focused", base: 0.72 },
      { label: "curious", base: 0.68 },
      { label: "calm", base: 0.66 },
      { label: "energized", base: 0.62 },
    ];
    const pick = options[Math.floor(r * options.length)];
    const conf = clamp01(pick.base + (r - 0.5) * 0.18);

    setMood({ status: "ready", label: pick.label, confidence: conf });
  }

  // ✅ REAL chat call → backend POST /api/chat
  async function askEve(text) {
    const q = (text || "").trim();
    if (!q) return;

    setAskBusy(true);
    const nextHistory = [...history, { role: "user", content: q }];
    setHistory(nextHistory);

    try {
      const r = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, history: nextHistory }),
      });

      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error(t || `HTTP ${r.status}`);
      }

      const res = await r.json();
      const answer = res.answer || res.message || "";

      const finalHistory = [...nextHistory, { role: "assistant", content: answer }];
      setHistory(finalHistory);
      speak(answer);
    } catch (e) {
      const msg = `Error: ${e.message || "Failed"}`;
      setHistory([...nextHistory, { role: "assistant", content: msg }]);
      speak(msg);
    } finally {
      setAskBusy(false);
    }
  }

  async function openPanel(id) {
    // reset auto speak flags
    briefedTodayRef.current = false;
    briefedHealthRef.current = false;
    briefedScanRef.current = false;
    briefedMoodRef.current = false;

    setActivePanel(id);
    clearRects(); // important: new UI = new rects

    if (id === "today") {
      speak("Opening today's briefing.");
      setWeather({ status: "loading", data: null, error: "" });

      const fallback = { lat: 55.6761, lon: 12.5683, name: "Copenhagen area", timezone: "Europe/Copenhagen" };
      const geo = await getGeoPosition();

      let lat = fallback.lat;
      let lon = fallback.lon;
      let name = fallback.name;
      let timezone = fallback.timezone;

      if (geo?.lat != null && geo?.lon != null) {
        lat = geo.lat;
        lon = geo.lon;
        const info = await geocodeName(lat, lon);
        name = info?.name || "Local area";
        timezone = info?.timezone || "auto";
      }

      setPlace({ name, timezone, lat, lon });

      try {
        const data = await fetchWeather(lat, lon, timezone);
        setWeather({ status: "ready", data, error: "" });
      } catch (e) {
        setWeather({ status: "error", data: null, error: e?.message || "Failed to load weather" });
      }
    }

    if (id === "health") {
      speak("Opening health snapshot.");
      setHealth(makeHealthSynthetic());
    }

    if (id === "scan") {
      speak("Initiating quick scan.");
      await runScanOnce();
    }

    if (id === "mood") {
      speak("Initiating mood scan.");
      runMoodOnce();
    }

    if (id === "ask") {
      speak("Ask your question.");
    }
  }

  function closePanel() {
    setActivePanel(null);
    clearRects();
    showToast("Stand by…");
    speak("Standing by.");
  }

  // unified action handler (menu + panel buttons)
  async function onAction(id) {
    const now = Date.now();
    if (now - lastClickAtRef.current < CLICK_COOLDOWN_MS) return;
    lastClickAtRef.current = now;

    if (id === "close") return closePanel();

    if (id === "pin") {
      setPanelPinned((p) => {
        const next = !p;
        speak(next ? "Pinned." : "Unpinned.");
        return next;
      });
      return;
    }

    if (activePanel === "today") {
      if (id === "today_read") return speak(todayBriefingText());
    }
    if (activePanel === "health") {
      if (id === "health_read") return speak(healthBriefingText());
    }
    if (activePanel === "scan") {
      if (id === "scan_rescan") {
        speak("Rescanning.");
        await runScanOnce();
        return;
      }
      if (id === "scan_read") return speak(scanBriefingText(scan.items));
    }
    if (activePanel === "mood") {
      if (id === "mood_rescan") {
        speak("Rescanning mood.");
        runMoodOnce();
        return;
      }
      if (id === "mood_read") return speak(moodBriefingText(mood));
    }
    if (activePanel === "ask") {
      if (id === "ask_send") {
        const t = askInput;
        setAskInput("");
        await askEve(t);
        return;
      }
      if (id === "ask_mic") {
        try {
          const t = await listenOnce();
          if (t) await askEve(t);
        } catch {
          speak("Voice input is not supported here. Please type your question.");
        }
        return;
      }
      if (id === "ask_clear") {
        setHistory([]);
        speak("Cleared.");
        return;
      }
    }

    const menu = buttons.find((b) => b.id === id);
    if (menu) {
      addLog(`OPEN → ${menu.title}`);
      showToast(menu.title);
      return openPanel(id);
    }
  }

  // Auto speak results when ready
  useEffect(() => {
    if (activePanel !== "today") return;
    if (weather.status !== "ready") return;
    if (briefedTodayRef.current) return;
    briefedTodayRef.current = true;
    speak(todayBriefingText());
  }, [activePanel, weather.status]);

  useEffect(() => {
    if (activePanel !== "health") return;
    if (briefedHealthRef.current) return;
    briefedHealthRef.current = true;
    const t = setTimeout(() => speak(healthBriefingText()), 200);
    return () => clearTimeout(t);
  }, [activePanel, health.stressScore]);

  useEffect(() => {
    if (activePanel !== "scan") return;
    if (scan.status !== "ready") return;
    if (briefedScanRef.current) return;
    briefedScanRef.current = true;
    speak(scanBriefingText(scan.items));
  }, [activePanel, scan.status]);

  useEffect(() => {
    if (activePanel !== "mood") return;
    if (mood.status !== "ready") return;
    if (briefedMoodRef.current) return;
    briefedMoodRef.current = true;
    speak(moodBriefingText(mood));
  }, [activePanel, mood.status]);

  // IDLE animation
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const c = idleCanvasRef.current;
      if (c) {
        const ctx = c.getContext("2d");
        const W = (c.width = c.clientWidth * devicePixelRatio);
        const H = (c.height = c.clientHeight * devicePixelRatio);
        const t = performance.now() * 0.001;

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "rgba(0,0,0,0.25)";
        ctx.fillRect(0, 0, W, H);

        const cx = W * 0.5;
        const cy = H * 0.45;
        const R = Math.min(W, H) * 0.18;

        ctx.save();
        ctx.globalAlpha = 0.85;
        ctx.strokeStyle = "rgba(59,130,246,0.35)";
        ctx.lineWidth = 6 * devicePixelRatio;
        ctx.beginPath();
        ctx.arc(cx, cy, R * (1.15 + 0.04 * Math.sin(t * 1.3)), 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        ctx.save();
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = "rgba(255,255,255,0.06)";
        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        for (let i = 0; i < 4; i++) {
          const phase = t * (0.6 + i * 0.12);
          const a0 = phase + i * 0.9;
          const a1 = a0 + 1.25 + 0.2 * Math.sin(t + i);
          ctx.strokeStyle = `rgba(16,185,129,${0.18 + 0.12 * Math.sin(t + i)})`;
          ctx.lineWidth = (2 + i) * devicePixelRatio;
          ctx.beginPath();
          ctx.arc(cx, cy, R * (0.65 + i * 0.13), a0, a1);
          ctx.stroke();
        }

        const idleVisible = !online && !activePanel;
        ctx.save();
        ctx.globalAlpha = idleVisible ? 0.9 : 0.0;
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.font = `${18 * devicePixelRatio}px ui-sans-serif, system-ui`;
        ctx.fillText("EVE", W * 0.5 - 18 * devicePixelRatio, H * 0.78);
        ctx.globalAlpha = idleVisible ? 0.65 : 0.0;
        ctx.font = `${13 * devicePixelRatio}px ui-sans-serif, system-ui`;
        ctx.fillText("Stand by…", W * 0.5 - 34 * devicePixelRatio, H * 0.82);
        ctx.restore();
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [online, activePanel]);

  // Start camera
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
        if (!active) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const v = videoRef.current;
        if (!v) return;
        v.srcObject = stream;
        await new Promise((res) => (v.onloadedmetadata = res));
        await v.play().catch(() => {});
        setReady(true);
      } catch (e) {
        console.error(e);
        setCamError("Camera blocked. Allow camera permissions and refresh.");
      }
    })();

    return () => {
      active = false;
      const v = videoRef.current;
      const s = v?.srcObject;
      if (s && typeof s.getTracks === "function") s.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Hand tracking loop
  useEffect(() => {
    if (!ready) return;
    let stop = false;

    (async () => {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const { FilesetResolver, HandLandmarker } = vision;

        const fileset = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const handLandmarker = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });

        addLog("Gesture system: ready");

        const tick = async () => {
          if (stop) return;

          const v = videoRef.current;
          const c = overlayRef.current;

          if (v && c && v.readyState >= 2) {
            c.width = v.videoWidth;
            c.height = v.videoHeight;
            const ctx = c.getContext("2d");
            const W = c.width;
            const H = c.height;

            ctx.clearRect(0, 0, W, H);

            const out = handLandmarker.detectForVideo(v, performance.now());
            const lm = out?.landmarks?.[0];

            let palmOn = false;
            let pStrength = 0;
            let pinchOn = false;
            let pointer = null;

            if (lm && lm.length >= 21) {
              palmOn = isPalm(lm);
              pStrength = pinchStrength(lm);
              pinchOn = pStrength > 0.72;
              pointer = lm[IDX.indexTip];
            }

            setPinchP(clamp01(pStrength));

            if (palmOn && !onlineRef.current) {
              setOnline(true);
              setToast("EVE ONLINE");
              addLog("EVE ONLINE");
            }

            if (!palmOn && onlineRef.current) {
              draggingRef.current = false;
              lastHoverDuringDragRef.current = null;
              setOnline(false);
              setHoverId(null);

              if (!activePanel || !panelPinned) {
                setToast("Stand by…");
                addLog("DISENGAGED");
              } else {
                addLog("DISENGAGED (panel pinned)");
              }
            }

            if (onlineRef.current) {
              const hoveredNow = hitTestCursor(cursorRef.current.x, cursorRef.current.y);
              setHoverId(hoveredNow);

              if (pinchOn && pointer) {
                draggingRef.current = true;
                const tx = 1 - clamp01(pointer.x);
                const ty = clamp01(pointer.y);

                const cur = cursorRef.current;
                const vel = cursorVelRef.current;

                const stiffness = 0.22;
                const damping = 0.70;

                vel.x = vel.x * damping + (tx - cur.x) * stiffness;
                vel.y = vel.y * damping + (ty - cur.y) * stiffness;

                cur.x = clamp01(cur.x + vel.x);
                cur.y = clamp01(cur.y + vel.y);

                cursorRef.current = cur;
                cursorVelRef.current = vel;

                setCursor({ x: cur.x, y: cur.y });
                lastHoverDuringDragRef.current = hitTestCursor(cur.x, cur.y);
              } else {
                if (draggingRef.current) {
                  draggingRef.current = false;
                  const target = hitTestCursor(cursorRef.current.x, cursorRef.current.y) || lastHoverDuringDragRef.current;
                  lastHoverDuringDragRef.current = null;
                  if (target) onAction(target);
                  else addLog("DROP → (no target)");
                }
              }

              // HUD overlay
              ctx.fillStyle = "rgba(0,0,0,0.08)";
              ctx.fillRect(0, 0, W, H);

              const cx = W * 0.5;
              const cy = H * 0.5;

              ctx.strokeStyle = "rgba(59,130,246,0.9)";
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.arc(cx, cy, 42, 0, Math.PI * 2);
              ctx.stroke();

              ctx.strokeStyle = "rgba(16,185,129,0.95)";
              ctx.lineWidth = 5;
              ctx.beginPath();
              ctx.arc(cx, cy, 58, -Math.PI / 2, -Math.PI / 2 + clamp01(pStrength) * Math.PI * 2);
              ctx.stroke();

              ctx.fillStyle = "rgba(255,255,255,0.88)";
              ctx.font = "14px ui-sans-serif, system-ui";
              ctx.fillText("EVE ONLINE", 16, 24);
              ctx.font = "12px ui-sans-serif, system-ui";
              ctx.fillText("Pinch+hold: move  •  Release pinch: select  •  Release palm: exit", 16, 44);
            }
          }

          setTimeout(tick, 90);
        };

        tick();
      } catch (e) {
        console.warn(e);
        addLog("Gesture system: failed to load");
      }
    })();

    return () => {
      stop = true;
    };
  }, [ready, activePanel, panelPinned, rectsEpoch]);

  useEffect(() => {
    const onResize = () => clearRects();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const cursorPx = useMemo(() => ({ x: cursor.x * window.innerWidth, y: cursor.y * window.innerHeight }), [cursor]);

  const panelCardStyle = {
    borderRadius: 16,
    border: "1px solid rgba(255,255,255,0.10)",
    background: "rgba(0,0,0,0.22)",
    padding: 12,
  };
  const statLabel = { fontSize: 12, opacity: 0.7 };
  const statValue = { fontSize: 22, fontWeight: 900, letterSpacing: "0.02em" };
  const smallBtn = {
    padding: "8px 10px",
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.18)",
    cursor: "pointer",
    fontSize: 12,
    opacity: 0.9,
    color: "rgba(255,255,255,0.92)",
    userSelect: "none",
  };

  function TodayPanel() {
    const data = weather.data;
    const current = data?.current;
    const daily = data?.daily;

    const reminders = ["Portfolio: verify links + live demos", "Recruiter mode: keep services pinned", "Next: connect DeepSeek vision + mood model"];

    return (
      <div style={{ display: "grid", gap: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
          <div style={{ fontWeight: 900, letterSpacing: "0.08em" }}>TODAY'S INFO</div>
          <div style={{ opacity: 0.75, fontSize: 12 }}>{formatDay()}</div>
        </div>

        <div style={panelCardStyle}>
          <div style={{ fontWeight: 800, opacity: 0.9 }}>
            {place.name}{" "}
            <span style={{ opacity: 0.65, fontSize: 12 }}>
              ({place.lat?.toFixed?.(2)}, {place.lon?.toFixed?.(2)})
            </span>
          </div>

          {weather.status === "loading" ? (
            <div style={{ opacity: 0.75, marginTop: 8 }}>Loading weather…</div>
          ) : weather.status === "error" ? (
            <div style={{ opacity: 0.75, marginTop: 8 }}>Weather error: {weather.error}</div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
              <div>
                <div style={statLabel}>Temp</div>
                <div style={statValue}>{num(current?.temperature_2m, 1)}°C</div>
              </div>
              <div>
                <div style={statLabel}>Feels</div>
                <div style={statValue}>{num(current?.apparent_temperature, 1)}°C</div>
              </div>
              <div>
                <div style={statLabel}>Wind</div>
                <div style={statValue}>{num(current?.wind_speed_10m, 0)} km/h</div>
              </div>
              <div>
                <div style={statLabel}>Precip</div>
                <div style={statValue}>{num(current?.precipitation, 1)} mm</div>
              </div>
            </div>
          )}
        </div>

        <div style={panelCardStyle}>
          <div style={{ fontWeight: 800, opacity: 0.9 }}>Forecast</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginTop: 10 }}>
            <div>
              <div style={statLabel}>High</div>
              <div style={statValue}>{num(daily?.temperature_2m_max?.[0], 1)}°C</div>
            </div>
            <div>
              <div style={statLabel}>Low</div>
              <div style={statValue}>{num(daily?.temperature_2m_min?.[0], 1)}°C</div>
            </div>
            <div>
              <div style={statLabel}>Rain</div>
              <div style={statValue}>{num(daily?.precipitation_sum?.[0], 1)} mm</div>
            </div>
          </div>
        </div>

        <div style={panelCardStyle}>
          <div style={{ fontWeight: 800, opacity: 0.9 }}>Reminders</div>
          <ul style={{ margin: "8px 0 0 18px", opacity: 0.82, lineHeight: 1.4 }}>
            {reminders.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button ref={(el) => registerRect("today_read", el)} onClick={() => onAction("today_read")} style={smallBtn}>
            🔊 Read out loud
          </button>
        </div>
      </div>
    );
  }

  function HealthPanel() {
    const eff = Math.round(health.efficiency * 100);
    const readiness = eff >= 85 ? "High" : eff >= 75 ? "Normal" : "Low";

    return (
      <div style={{ display: "grid", gap: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
          <div style={{ fontWeight: 900, letterSpacing: "0.08em" }}>HEALTH SNAPSHOT</div>
          <div style={{ opacity: 0.75, fontSize: 12 }}>Demo mode</div>
        </div>

        <div style={panelCardStyle}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div>
              <div style={statLabel}>Sleep</div>
              <div style={statValue}>{num(health.sleepHours, 1)}h</div>
              <div style={{ opacity: 0.7, fontSize: 12, marginTop: 4 }}>
                Bed {health.bedtime}:00 → Wake {health.wakeHour}:00
              </div>
            </div>
            <div>
              <div style={statLabel}>Efficiency</div>
              <div style={statValue}>{eff}%</div>
              <div style={{ opacity: 0.7, fontSize: 12, marginTop: 4 }}>Readiness: {readiness}</div>
            </div>
            <div>
              <div style={statLabel}>Resting HR</div>
              <div style={statValue}>{health.restingHR} bpm</div>
            </div>
            <div>
              <div style={statLabel}>Current HR</div>
              <div style={statValue}>{health.currentHR} bpm</div>
            </div>
            <div>
              <div style={statLabel}>Stress</div>
              <div style={statValue}>
                {health.stressScore} <span style={{ fontSize: 14, opacity: 0.75 }}>/100</span>
              </div>
              <div style={{ opacity: 0.7, fontSize: 12, marginTop: 4 }}>{health.stressLabel}</div>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button ref={(el) => registerRect("health_read", el)} onClick={() => onAction("health_read")} style={smallBtn}>
            🔊 Read out loud
          </button>
        </div>
      </div>
    );
  }

  function ScanPanel() {
    return (
      <div style={{ display: "grid", gap: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
          <div style={{ fontWeight: 900, letterSpacing: "0.08em" }}>QUICK SCAN</div>
          <div style={{ opacity: 0.75, fontSize: 12 }}>Local model</div>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button ref={(el) => registerRect("scan_rescan", el)} onClick={() => onAction("scan_rescan")} style={smallBtn}>
            ↻ Rescan
          </button>
          <button ref={(el) => registerRect("scan_read", el)} onClick={() => onAction("scan_read")} style={smallBtn}>
            🔊 Read results
          </button>
        </div>

        <div style={panelCardStyle}>
          {scan.status === "loading" ? (
            <div style={{ opacity: 0.8 }}>Scanning…</div>
          ) : scan.status === "error" ? (
            <div style={{ opacity: 0.8 }}>Scan error: {scan.error}</div>
          ) : scan.items.length ? (
            <div style={{ display: "grid", gap: 8 }}>
              {scan.items.slice(0, 8).map((x, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", opacity: 0.9 }}>
                  <div style={{ fontWeight: 800 }}>{x.label}</div>
                  <div style={{ opacity: 0.75 }}>{Math.round(x.score * 100)}%</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ opacity: 0.75 }}>No confident objects yet.</div>
          )}
        </div>
      </div>
    );
  }

  function MoodPanel() {
    return (
      <div style={{ display: "grid", gap: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
          <div style={{ fontWeight: 900, letterSpacing: "0.08em" }}>MOOD SCAN</div>
          <div style={{ opacity: 0.75, fontSize: 12 }}>Demo</div>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button ref={(el) => registerRect("mood_rescan", el)} onClick={() => onAction("mood_rescan")} style={smallBtn}>
            ↻ Rescan
          </button>
          <button ref={(el) => registerRect("mood_read", el)} onClick={() => onAction("mood_read")} style={smallBtn}>
            🔊 Read results
          </button>
        </div>

        <div style={panelCardStyle}>
          <div style={{ fontSize: 12, opacity: 0.7 }}>Detected</div>
          <div style={{ fontSize: 28, fontWeight: 900, letterSpacing: "0.04em" }}>{mood.label}</div>
          <div style={{ opacity: 0.7 }}>Confidence {Math.round(mood.confidence * 100)}%</div>
        </div>
      </div>
    );
  }

  function AskPanel() {
    return (
      <div style={{ display: "grid", gap: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
          <div style={{ fontWeight: 900, letterSpacing: "0.08em" }}>ASK EVE</div>
          <div style={{ opacity: 0.75, fontSize: 12 }}>{canListen ? "Mic ready" : "Mic unsupported"}</div>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={askInput}
            onChange={(e) => setAskInput(e.target.value)}
            placeholder='Try: "What do you see?"'
            style={{
              flex: 1,
              padding: 10,
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(0,0,0,0.18)",
              color: "rgba(255,255,255,0.92)",
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") onAction("ask_send");
            }}
          />
          <button ref={(el) => registerRect("ask_send", el)} onClick={() => onAction("ask_send")} disabled={askBusy} style={smallBtn}>
            {askBusy ? "…" : "Send"}
          </button>
          <button
            ref={(el) => registerRect("ask_mic", el)}
            onClick={() => onAction("ask_mic")}
            style={smallBtn}
            disabled={!canListen || askBusy}
            title="Voice question"
          >
            🎤
          </button>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button ref={(el) => registerRect("ask_clear", el)} onClick={() => onAction("ask_clear")} style={smallBtn}>
            Clear Chat
          </button>
        </div>

        <div style={{ ...panelCardStyle, maxHeight: 260, overflow: "auto" }}>
          {history.length === 0 ? (
            <div style={{ opacity: 0.75 }}>Ask a question. EVE will answer out loud.</div>
          ) : (
            history.map((m, i) => (
              <div key={i} style={{ marginTop: 10 }}>
                <div style={{ fontWeight: 800, opacity: 0.75 }}>{m.role === "user" ? "You" : "EVE"}</div>
                <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
              </div>
            ))
          )}
        </div>

        <div style={{ fontSize: 12, opacity: 0.65 }}>Backend: {API_BASE}</div>
      </div>
    );
  }

  const styles = {
    page: {
      minHeight: "100vh",
      padding: 16,
      fontFamily: "ui-sans-serif, system-ui",
      color: "rgba(255,255,255,0.92)",
      background:
        "radial-gradient(1200px 700px at 20% 10%, rgba(124,58,237,0.35), transparent 60%)," +
        "radial-gradient(900px 600px at 85% 25%, rgba(59,130,246,0.25), transparent 55%)," +
        "radial-gradient(900px 700px at 30% 90%, rgba(16,185,129,0.18), transparent 55%)," +
        "linear-gradient(180deg, #04050a, #070917 45%, #04050a)",
    },
    shell: { display: "grid", gridTemplateColumns: "1.25fr 0.95fr", gap: 16, alignItems: "start" },
    card: {
      borderRadius: 18,
      padding: 14,
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.10)",
      boxShadow: "0 20px 80px rgba(0,0,0,0.35)",
      backdropFilter: "blur(10px)",
    },
    titleRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 12, flexWrap: "wrap" },
    title: { margin: 0, fontSize: 32, letterSpacing: "0.18em", fontWeight: 900 },
    badge: {
      padding: "6px 10px",
      borderRadius: 999,
      border: "1px solid rgba(255,255,255,0.14)",
      background: "rgba(0,0,0,0.25)",
      fontSize: 12,
      opacity: 0.95,
    },
    camWrap: {
      position: "relative",
      borderRadius: 16,
      overflow: "hidden",
      border: "1px solid rgba(255,255,255,0.10)",
      background: "rgba(0,0,0,0.25)",
      minHeight: 420,
    },
    video: { width: "100%", display: "block" },
    overlay: { position: "absolute", inset: 0, width: "100%", height: "100%" },
    idleCanvas: {
      position: "absolute",
      inset: 0,
      width: "100%",
      height: "100%",
      opacity: online ? 0.0 : 1.0,
      transition: "opacity 350ms ease",
      pointerEvents: "none",
    },
    footer: { marginTop: 10, fontSize: 12, opacity: 0.68, lineHeight: 1.35 },

    panelHeaderRow: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, marginBottom: 10 },
    panelHeader: { fontWeight: 900, letterSpacing: "0.08em", opacity: 0.9 },
    btnRow: { display: "flex", gap: 8, alignItems: "center" },

    buttons: { display: "grid", gap: 10 },
    btn: (active) => ({
      padding: "12px 12px",
      borderRadius: 16,
      border: active ? "1px solid rgba(16,185,129,0.7)" : "1px solid rgba(255,255,255,0.12)",
      background: active ? "rgba(16,185,129,0.10)" : "rgba(0,0,0,0.18)",
      boxShadow: active ? "0 0 0 2px rgba(16,185,129,0.15) inset" : "none",
      cursor: "pointer",
      transition: "transform 120ms ease, background 120ms ease, border 120ms ease",
      transform: active ? "translateY(-1px)" : "translateY(0px)",
    }),
    btnTitle: { fontWeight: 900, fontSize: 14, letterSpacing: "0.04em" },
    btnSub: { fontSize: 12, opacity: 0.72, marginTop: 2, lineHeight: 1.25 },

    log: { marginTop: 12, display: "grid", gap: 8 },
    logRow: {
      padding: "8px 10px",
      borderRadius: 14,
      border: "1px solid rgba(255,255,255,0.10)",
      background: "rgba(0,0,0,0.22)",
      fontSize: 12,
      opacity: 0.9,
    },

    cursorDot: {
      position: "fixed",
      width: 14,
      height: 14,
      borderRadius: 999,
      border: "2px solid rgba(16,185,129,0.9)",
      background: "rgba(16,185,129,0.18)",
      boxShadow: "0 0 22px rgba(16,185,129,0.35)",
      transform: "translate(-50%, -50%)",
      pointerEvents: "none",
      zIndex: 50,
      opacity: online ? 1 : 0,
      transition: "opacity 200ms ease",
    },
    cursorTail: {
      position: "fixed",
      width: 36,
      height: 36,
      borderRadius: 999,
      border: "1px solid rgba(59,130,246,0.45)",
      transform: "translate(-50%, -50%)",
      pointerEvents: "none",
      zIndex: 49,
      opacity: online ? 1 : 0,
      transition: "opacity 200ms ease",
    },
    toast: {
      position: "fixed",
      left: 16,
      bottom: 16,
      padding: "10px 12px",
      borderRadius: 14,
      background: "rgba(0,0,0,0.30)",
      border: "1px solid rgba(255,255,255,0.14)",
      backdropFilter: "blur(10px)",
      zIndex: 60,
      fontSize: 12,
      opacity: 0.92,
    },
  };

  return (
    <div style={styles.page}>
      <TopNotice onVisibilityChange={setNoticeVisible} />
      {noticeVisible && <div style={{ height: 48 }} />}

      <div style={styles.titleRow}>
        <h1 style={styles.title}>EVE</h1>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <span style={styles.badge}>{online ? "ONLINE" : "IDLE"}</span>
          <span style={styles.badge}>Pinch: {Math.round(clamp01(pinchP) * 100)}%</span>
          <span style={styles.badge}>Hover: {hoverId || "—"}</span>
        </div>
      </div>

      <div style={styles.shell}>
        <div style={styles.card}>
          <div style={{ fontWeight: 900, letterSpacing: "0.08em", opacity: 0.9, marginBottom: 10 }}>Live Camera Feed</div>
          {camError ? (
            <div style={{ opacity: 0.92 }}>{camError}</div>
          ) : (
            <div style={styles.camWrap}>
              <canvas ref={idleCanvasRef} style={styles.idleCanvas} />
              <video ref={videoRef} style={styles.video} playsInline muted />
              <canvas ref={overlayRef} style={styles.overlay} />
            </div>
          )}
          <div style={styles.footer}>
            Instructions: Palm = wake up Eve. Pinch+hold = moves cursor. Release pinch = select. EVE speaks results for every feature.
          </div>
        </div>

        <div style={styles.card}>
          <div style={styles.panelHeaderRow}>
            <div style={styles.panelHeader}>{activePanel ? "SYSTEM" : "SYSTEM MENU"}</div>
            <div style={styles.btnRow}>
              {activePanel && (
                <button ref={(el) => registerRect("close", el)} style={smallBtn} onClick={() => onAction("close")}>
                  Close
                </button>
              )}
              <button
                ref={(el) => registerRect("pin", el)}
                style={smallBtn}
                onClick={() => onAction("pin")}
                title="Keep panel open when palm goes away"
              >
                {panelPinned ? "Pinned" : "Unpinned"}
              </button>
            </div>
          </div>

          {activePanel ? (
            <div>
              {activePanel === "today" && <TodayPanel />}
              {activePanel === "health" && <HealthPanel />}
              {activePanel === "scan" && <ScanPanel />}
              {activePanel === "mood" && <MoodPanel />}
              {activePanel === "ask" && <AskPanel />}
            </div>
          ) : (
            <div style={styles.buttons}>
              {buttons.map((b) => (
                <div key={b.id} ref={(el) => registerRect(b.id, el)} onClick={() => onAction(b.id)} style={styles.btn(hoverId === b.id)}>
                  <div style={styles.btnTitle}>{b.title}</div>
                  <div style={styles.btnSub}>{b.subtitle}</div>
                </div>
              ))}
            </div>
          )}

          <div style={styles.log}>
            {log.length === 0 ? (
              <div style={styles.logRow}>Stand by… Show your palm to bring EVE online.</div>
            ) : (
              log.map((x, i) => (
                <div key={i} style={styles.logRow}>
                  <b>{x.t}</b> — {x.line}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div style={{ ...styles.cursorTail, left: cursorPx.x, top: cursorPx.y }} />
      <div style={{ ...styles.cursorDot, left: cursorPx.x, top: cursorPx.y }} />
      <div style={styles.toast}>{toast}</div>
    </div>
  );
}