(() => {
  const $ = (id) => document.getElementById(id);
  const AXES = ["speed", "fractal", "symmetry", "color", "breath", "texture"];
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "-" : (+v).toFixed(d);
  const scores = [];  // [t, score|null]
  let lastState = null;

  function kv(el, obj) {
    el.innerHTML = "";
    for (const [k, v] of Object.entries(obj)) {
      const a = document.createElement("div"); a.textContent = k;
      const b = document.createElement("div"); b.textContent = v;
      el.append(a, b);
    }
  }
  function physical(p) {
    if (!p) return {};
    return {
      "motion": `${fmt(p.speed)} (${fmt(0.08 * p.speed, 3)} u/s)`,
      "fractal D": `${fmt(p.fractal)} (D ${fmt(1.1 + 0.6 * p.fractal)})`,
      "symmetry": `${fmt(p.symmetry)} (${Math.round(1 + 11 * p.symmetry)}-fold)`,
      "colour": `${fmt(p.color)} (${Math.round(2700 + 5300 * p.color)} K)`,
      "breath 0.1 Hz": `${fmt(p.breath)} (${fmt(5 * p.breath, 1)} %)`,
      "texture latent": fmt(p.texture),
    };
  }
  function fitClass(v) { return v <= 1 ? "good" : v <= 2 ? "mid" : "bad"; }

  function render(s) {
    lastState = s;
    $("session").textContent = `${s.wearer} / ${s.session_id}`;
    $("phase").textContent = s.phase.replace("calibration_", "cal: ");
    $("trial").textContent = s.phase === "idle" ? "-" : `${s.trial} / ${s.n_trials}`;
    $("remaining").textContent = s.phase === "idle" ? "-" : `${Math.round(s.phase_remaining)} s`;
    $("score").textContent = fmt(s.last_score);
    kv($("params"), s.params ? { "proposal": s.proposal_tag, ...physical(s.params) } : { "stimulus": s.phase === "idle" ? "none" : "uniform field" });
    $("message").textContent = s.message || "";
    $("btnStart").disabled = s.running;
    $("btnStop").disabled = !s.running;

    const names = ["TP9", "AF7", "AF8", "TP10"];
    const fits = s.fit || [];
    const kept = (s.quality && s.quality.kept_fraction) || {};
    $("fits").innerHTML = names.map((n, i) => {
      const f = fits[i] === undefined ? "-" : fits[i];
      const k = kept[n];
      return `<div class="fit ${fits[i] === undefined ? "" : fitClass(fits[i])}"><div>${n}</div><div class="muted">fit ${f}</div><div class="bar"><i style="width:${Math.round(100 * (k || 0))}%"></i></div><div class="muted">${k === undefined ? "" : Math.round(100 * k) + "% kept"}</div></div>`;
    }).join("");
    const q = s.quality || {};
    kv($("quality"), {
      "groups used": (q.groups_used || []).join(" + ") || "-",
      "temporal kept": q.temporal_kept === undefined ? "-" : `${Math.round(100 * q.temporal_kept)} %`,
      "frontal kept": q.frontal_kept === undefined ? "-" : `${Math.round(100 * q.frontal_kept)} %`,
      "segments": q.n_segments === undefined ? "-" : q.n_segments,
      "valid / rejected": `${s.scorer.valid} / ${s.scorer.rejected}`,
      "source": JSON.stringify(s.source),
    });
    $("fallback").textContent = q.fallback_frontal_only ? "temporal electrodes unusable - scoring frontal-only (fallback)" : "";

    const o = s.optimizer || {};
    kv($("opt"), {
      "acquisition": o.acquisition || "-",
      "observations": `${o.n_rewards || 0} rewards, ${o.n_pairwise || 0} pairwise`,
      "seeds remaining": o.seeds_remaining === undefined ? "-" : o.seeds_remaining,
      "ARD length scales": o.length_scales ? AXES.map(a => `${a.slice(0, 3)} ${fmt(o.length_scales[a], 1)}`).join("  ") : "-",
    });
    const b = o.best || {};
    const bx = b.x ? Object.fromEntries(AXES.map((a, i) => [a, b.x[i]])) : null;
    kv($("best"), bx ? { "source": b.source, "predicted": b.mean === null ? "-" : `${fmt(b.mean)} ± ${fmt(b.std)}`, ...physical(bx) } : { "best": "-" });

    const tb = $("trials").querySelector("tbody");
    tb.innerHTML = (s.trials || []).slice().reverse().map(t => {
      const p = t.params, q = t.stimulus_quality || {};
      const qtxt = `${(q.groups_used || []).map(g => g[0]).join("+")} ${q.temporal_kept === undefined ? "" : Math.round(100 * q.temporal_kept) + "%T"}${q.fallback_frontal_only ? " FB" : ""}`;
      return `<tr><td>${t.trial}</td><td><span class="tag">${t.tag}</span></td><td>${fmt(p.speed)}</td><td>${fmt(1.1 + 0.6 * p.fractal)}</td><td>${Math.round(1 + 11 * p.symmetry)}</td><td>${Math.round(2700 + 5300 * p.color)}</td><td>${fmt(p.breath)}</td><td>${fmt(p.texture)}</td><td>${fmt(t.baseline_window)}</td><td>${fmt(t.stimulus_window)}</td><td><b>${t.valid ? fmt(t.reward) : "invalid"}</b></td><td>${t.rating || ""}</td><td class="muted">${qtxt}</td></tr>`;
    }).join("");

    const c = s.calibration || {};
    const spec = s.spec || {};
    kv($("cal"), {
      "spec": `${spec.name || "-"} ${spec.version || ""} (window ${spec.window_s} s / hop ${spec.hop_s} s)`,
      "calibrated": s.calibrated ? "yes" : "no",
      "profile": c.profile || spec.profile || "-",
      "separation d": c.separation_d === undefined ? "-" : fmt(c.separation_d),
      "signs": c.signs ? Object.entries(c.signs).filter(([k]) => (spec.features || []).some(f => f.name === k && f.weight !== 0)).map(([k, v]) => `${k.replace("_log", "").replace("temporal_", "T:").replace("frontal_", "F:")}${v > 0 ? "+" : "-"}`).join("  ") : "-",
      "notes": (c.notes || []).join(" | ") || "-",
    });
  }

  // sparkline
  function drawSpark() {
    const cv = $("spark"), ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    if (scores.length < 2) return;
    const tNow = scores[scores.length - 1][0], tMin = tNow - 600;
    const pts = scores.filter(p => p[0] >= tMin);
    const vals = pts.map(p => p[1]).filter(v => v !== null);
    if (!vals.length) return;
    const lo = Math.min(-2, ...vals), hi = Math.max(2, ...vals);
    const x = (t) => ((t - tMin) / 600) * W, y = (v) => H - ((v - lo) / (hi - lo)) * (H - 10) - 5;
    ctx.strokeStyle = "#2a2f38"; ctx.beginPath(); ctx.moveTo(0, y(0)); ctx.lineTo(W, y(0)); ctx.stroke();
    ctx.strokeStyle = "#7fb3d5"; ctx.lineWidth = 1.5; ctx.beginPath(); let pen = false;
    for (const [t, v] of pts) { if (v === null) { pen = false; continue; } if (!pen) { ctx.moveTo(x(t), y(v)); pen = true; } else ctx.lineTo(x(t), y(v)); }
    ctx.stroke();
    ctx.fillStyle = "#d26a5c";
    for (const [t, v] of pts) if (v === null) ctx.fillRect(x(t) - 1, H - 4, 2, 4);
  }

  function log(msg) {
    const el = $("log");
    el.textContent += (el.textContent ? "\n" : "") + msg;
    el.scrollTop = el.scrollHeight;
  }

  let ws;
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/dashboard`);
    ws.onopen = () => { $("conn").textContent = "connected"; };
    ws.onclose = () => { $("conn").textContent = "disconnected"; setTimeout(connect, 1500); };
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "state") render(m);
      else if (m.type === "score") { scores.push([m.t, m.valid ? m.score : null]); if (scores.length > 2000) scores.shift(); drawSpark(); }
      else if (m.type === "scores") { scores.length = 0; for (const p of m.points) scores.push(p); drawSpark(); }
      else if (m.type === "log") log(`[${(m.t || 0).toFixed(0).padStart(5)}] ${m.msg}`);
      else if (m.type === "trial") { /* state follows */ }
      else if (m.type === "diag") log(`visuals diag: ${JSON.stringify(m.patches.map(p => ({ band: +(100 * p.band_3_30_share).toFixed(1), peak: p.peak_hz })))}`);
    };
  }
  connect();
  const send = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));
  $("btnStart").onclick = () => send({ type: "start" });
  $("btnStop").onclick = () => send({ type: "stop" });
  $("btnBetter").onclick = () => send({ type: "rate", better: true });
  $("btnWorse").onclick = () => send({ type: "rate", better: false });
})();
