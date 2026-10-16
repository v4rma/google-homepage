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
  function fitClass(v) { return v === null || v === undefined ? "unknown" : v <= 1 ? "good" : v <= 2 ? "mid" : "bad"; }

  function renderStream(m) {
    const panel = $("streamPanel");
    const s = m && m.summary;
    $("streamStatus").textContent = m ? (m.error ? "failed" : !m.seconds ? "live (session statistics)" : m.running ? `listening ${Math.round(m.elapsed_s)} / ${Math.round(m.seconds)} s` : `done (${(s && s.duration_s) || 0} s)`) : "";
    if (!s) {
      $("streamDiag").className = "diag " + (m && m.error ? "fatal" : "");
      $("streamDiag").textContent = m && m.error ? m.error : "no stream check yet - press Check stream";
      for (const id of ["streamAddr", "streamEeg", "streamSeg"]) $(id).innerHTML = "";
      return;
    }
    const d = s.diagnosis || {};
    $("streamDiag").className = `diag ${d.severity || ""}`;
    $("streamDiag").innerHTML = `<b>${(d.severity || "").toUpperCase()}</b> ${d.cause || ""}` + (d.advice ? `<div class="advice">${d.advice}</div>` : "");
    const dur = Math.max(s.duration_s || 0, 1e-9);
    kv($("streamAddr"), Object.fromEntries(Object.entries(s.addresses || {}).slice(0, 12).map(([a, n]) => [a, `${n} (${(n / dur).toFixed(1)}/s)`])));
    const e = s.eeg || {}, r = e.raw || {};
    const hs = s.horseshoe;
    kv($("streamEeg"), {
      "samples / rate": `${e.n || 0} / ${e.rate_hz || 0} Hz (nominal ${e.nominal_hz || 256})`,
      "rate per channel": (e.rate_per_channel_hz || []).map(v => v === null ? "-" : Math.round(v)).join("  "),
      "raw median": r.median ? r.median.map(v => (+v).toFixed(0)).join("  ") : "-",
      "raw min / max": r.min ? `${r.min.map(v => Math.round(v)).join(" ")} / ${r.max.map(v => Math.round(v)).join(" ")}` : "-",
      "units": r.units_guess || "-",
      "horseshoe": hs ? `${hs.n} msgs, last ${hs.last.join(" ")}` : "none received (fit unknown)",
      "band msgs": Object.keys((s.bands || {}).n || {}).length ? JSON.stringify(s.bands.n) : "none",
      "blink / jaw": `${(s.markers || {}).blink || 0} / ${(s.markers || {}).jaw_clench || 0}`,
    });
    const g = s.segments || {}, w = s.windows || {};
    kv($("streamSeg"), {
      "segments": g.n || 0,
      "kept per channel": g.kept_fraction ? Object.entries(g.kept_fraction).map(([c, f]) => `${c} ${Math.round(100 * f)}%`).join("  ") : "-",
      "usable groups": (g.usable_groups || []).join(" + ") || "none",
      "reasons": Object.keys(g.reasons || {}).length ? Object.entries(g.reasons).map(([k, n]) => `${k} ${n}`).join("  ") : "none",
      "per channel": g.reasons_per_channel ? Object.entries(g.reasons_per_channel).filter(([, v]) => Object.keys(v).length).map(([c, v]) => `${c}: ${Object.entries(v).map(([k, n]) => `${k} ${n}`).join(",")}`).join("  ") || "-" : "-",
      "windows valid": `${w.valid || 0} / ${w.n || 0}${w.frontal_only ? ` (${w.frontal_only} frontal-only)` : ""}`,
      "spec": `${(s.spec || {}).name || ""}${(s.spec || {}).tolerant ? " (fit-tolerant)" : ""}`,
    });
  }

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
    $("btnCheck").disabled = s.running;

    const cp = s.calibration_progress;
    if (cp) {
      const pct = Math.min(100, Math.round(100 * (cp.elapsed_s || 0) / Math.max(cp.block_s || 1, 1)));
      const ok = cp.valid_windows >= cp.needed;
      $("calprog").innerHTML = `<div class="muted">calibration block ${cp.block} / ${cp.of} - ${cp.condition.replace("calibration_", "")}${cp.extension ? ` (extension ${cp.extension})` : ""}</div><div style="font-size:18px" class="${ok ? "" : "warn"}">valid windows <b>${cp.valid_windows}</b> / ${cp.needed} needed</div><div class="progress"><i style="width:${pct}%"></i></div>`;
    } else $("calprog").innerHTML = "";

    const names = ["TP9", "AF7", "AF8", "TP10"];
    const fits = s.fit;  // null when no horseshoe message has arrived: unknown, not bad
    const kept = (s.quality && s.quality.kept_fraction) || {};
    $("fits").innerHTML = names.map((n, i) => {
      const f = fits ? fits[i] : null;
      const k = kept[n];
      return `<div class="fit ${fitClass(f)}"><div>${n}</div><div class="muted">fit ${f === null ? "?" : f}</div><div class="bar"><i style="width:${Math.round(100 * (k || 0))}%"></i></div><div class="muted">${k === undefined ? "" : Math.round(100 * k) + "% kept"}</div></div>`;
    }).join("");
    const q = s.quality || {};
    const st = s.stream || {};
    kv($("quality"), {
      "groups used": (q.groups_used || []).join(" + ") || "-",
      "temporal kept": q.temporal_kept === undefined ? "-" : `${Math.round(100 * q.temporal_kept)} %`,
      "frontal kept": q.frontal_kept === undefined ? "-" : `${Math.round(100 * q.frontal_kept)} %`,
      "segments": q.n_segments === undefined ? "-" : q.n_segments,
      "EEG rate": st.eeg && st.eeg.rate_hz ? `${st.eeg.rate_hz} Hz` : "-",
      "raw offset / scale": q.raw_offset ? `${q.raw_offset.map(v => Math.round(v)).join(" ")} / x${q.raw_scale}` : "-",
      "valid / rejected": `${s.scorer.valid} / ${s.scorer.rejected}`,
      "source": JSON.stringify(s.source),
    });
    const warn = [];
    if (q.fallback_frontal_only) warn.push("temporal electrodes unusable - scoring frontal-only (fallback)");
    if (s.running && s.fit_known === false && q.n_segments) warn.push("no horseshoe messages: fit unknown (not treated as bad)");
    if (st.diagnosis && st.diagnosis.severity && st.diagnosis.severity !== "ok" && s.running) warn.push(`${st.diagnosis.severity}: ${st.diagnosis.cause}`);
    $("fallback").textContent = warn.join(" | ");
    if (s.stream && s.stream.addresses) renderStream({ running: s.running, elapsed_s: 0, seconds: 0, summary: s.stream });

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
      else if (m.type === "stream_check") renderStream(m);
      else if (m.type === "diag") log(`visuals diag: ${JSON.stringify(m.patches.map(p => ({ band: +(100 * p.band_3_30_share).toFixed(1), peak: p.peak_hz })))}`);
    };
  }
  connect();
  const send = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));
  $("btnStart").onclick = () => send({ type: "start" });
  $("btnCheck").onclick = () => send({ type: "check_stream" });
  $("btnStop").onclick = () => send({ type: "stop" });
  $("btnBetter").onclick = () => send({ type: "rate", better: true });
  $("btnWorse").onclick = () => send({ type: "rate", better: false });
})();
