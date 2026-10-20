/* Avatar UI. Plain fetch + DOM; no build step. */
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = { view: "episodes", episodes: [], selectedEpisode: null, assumptions: [], showRejected: false, episodeTitles: {} };

function toast(msg, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), ms);
}

async function api(method, path, body) {
  const res = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

async function busy(btn, fn) {
  if (btn) btn.disabled = true;
  try { return await fn(); }
  catch (e) { toast("Error: " + e.message, 6000); }
  finally { if (btn) btn.disabled = false; }
}

const epLabel = (id) => (state.episodeTitles[id] ? `${state.episodeTitles[id]} (${id})` : id);
const epLink = (id) => `<a href="#" data-open-episode="${esc(id)}">${esc(epLabel(id))}</a>`;
const statusBadge = (s) => `<span class="badge ${s === "rejected" ? "rejected" : s === "revised" ? "revised" : s === "under examination" ? "under" : ""}">${esc(s)}</span>`;

// ------------------------------------------------------------------ status + nav
async function refreshStatus() {
  const s = await api("GET", "/api/status");
  const c = s.counts;
  $("#status").innerHTML = `provider <b>${esc(s.provider)}${s.model ? " / " + esc(s.model) : ""}</b> · ${c.episodes} episodes (${c.reserved_episodes} reserved) · ${c.active_assumptions} active candidates · ${c.interview_turns} questions`;
}

document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
function showView(v) {
  state.view = v;
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  document.querySelectorAll(".view").forEach((s) => s.classList.toggle("active", s.id === "view-" + v));
  ({ episodes: loadEpisodes, assumptions: loadAssumptions, interview: loadInterview, holdout: loadHoldout })[v]();
  refreshStatus();
}

document.body.addEventListener("click", (e) => {
  const a = e.target.closest("[data-open-episode]");
  if (a) { e.preventDefault(); showView("episodes"); openEpisode(a.dataset.openEpisode); }
});

// ------------------------------------------------------------------ episodes
async function loadEpisodes() {
  state.episodes = await api("GET", "/api/episodes");
  state.episodeTitles = Object.fromEntries(state.episodes.map((e) => [e.id, e.title || e.situation.slice(0, 40)]));
  $("#episode-list").innerHTML = state.episodes.map((e) => `
    <li data-id="${esc(e.id)}" class="${state.selectedEpisode === e.id ? "selected" : ""}">
      <div>${esc(e.title || e.situation.slice(0, 60))}</div>
      <div class="meta"><span>${esc(e.id)}</span>${e.reserved ? '<span class="badge reserved">reserved</span>' : ""}${e.turn_count ? `<span>${e.turn_count} q</span>` : ""}</div>
    </li>`).join("") || '<li class="hint">No episodes yet.</li>';
  document.querySelectorAll("#episode-list li[data-id]").forEach((li) => li.addEventListener("click", () => openEpisode(li.dataset.id)));
}

$("#btn-new-episode").addEventListener("click", () => {
  state.selectedEpisode = null;
  const tpl = $("#episode-form").content.cloneNode(true);
  $("#episode-detail").innerHTML = "";
  $("#episode-detail").appendChild(tpl);
  $("#cancel-episode").addEventListener("click", () => ($("#episode-detail").innerHTML = '<p class="hint">Cancelled.</p>'));
  $("#new-episode-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    const body = Object.fromEntries(fd.entries());
    body.reserved = fd.get("reserved") === "on";
    await busy(ev.submitter, async () => {
      const ep = await api("POST", "/api/episodes", body);
      toast("Episode saved. The original account is now locked.");
      await loadEpisodes();
      openEpisode(ep.id);
    });
  });
});

$("#btn-seed").addEventListener("click", (ev) => busy(ev.target, async () => {
  const r = await api("POST", "/api/demo/seed");
  toast(r.created.length ? `Loaded ${r.created.length} synthetic demo episodes.` : "Demo data already loaded.");
  loadEpisodes(); refreshStatus();
}));

async function openEpisode(id) {
  state.selectedEpisode = id;
  document.querySelectorAll("#episode-list li").forEach((li) => li.classList.toggle("selected", li.dataset.id === id));
  const ep = await api("GET", `/api/episodes/${id}`);
  const obs = ep.observations.filter((o) => o.source === "extract");
  const byKind = {};
  obs.forEach((o) => (byKind[o.kind] = byKind[o.kind] || []).push(o));
  const kinds = ["event", "quotation", "expectation", "emotion", "action", "missing"];
  const field = (label, v) => `<div class="field"><div class="label">${label}</div><div class="value">${esc(v) || "<span class='hint'>—</span>"}</div></div>`;
  $("#episode-detail").innerHTML = `
    <h2>${esc(ep.title || "Episode")} <span class="hint">${esc(ep.id)}</span> ${ep.reserved ? '<span class="badge reserved">reserved for prediction test</span>' : ""}</h2>
    <div class="hint">Written ${esc(ep.created_at)}</div>
    <h3>Original account (immutable)</h3>
    <div class="locked">
      ${field("Situation", ep.situation)}${field("Immediate thought", ep.immediate_thought)}${field("Emotion and bodily feeling", ep.emotion_body)}
      ${field("Action and outcome", ep.action_outcome)}${field("Comparable occasion that went differently", ep.comparable_occasion)}
    </div>
    <div class="actions">
      ${ep.reserved ? "" : `<button class="primary" id="ep-process" title="Extract, hypothesise and critique">Analyse</button><button id="ep-extract">Extract only</button><button id="ep-ask">Ask a question about this episode</button>`}
      <button id="ep-reserve">${ep.reserved ? "Un-reserve" : "Reserve for prediction test"}</button>
      <button class="danger" id="ep-delete">Delete episode</button>
    </div>
    <h3>Observations <span class="hint">(extracted from your words; quotations marked verified appear verbatim in your account)</span></h3>
    ${obs.length ? kinds.filter((k) => byKind[k]).map((k) => `<div class="field"><div class="label">${k}</div><ul class="plain">${byKind[k].map((o) => `<li>${esc(o.content)} ${k === "missing" ? "" : `<span class="badge ${o.verified ? "verified" : "unverified"}">${o.verified ? "verified" : "paraphrase"}</span>`}</li>`).join("")}</ul></div>`).join("") : '<p class="hint">Not extracted yet.</p>'}
    <h3>Interview on this episode</h3>
    ${ep.turns.length ? ep.turns.map((t) => `<div class="field"><div class="label">Q</div><div>${esc(t.question)}</div><div class="label">A</div><div>${t.answer == null ? "<span class='hint'>(waiting for your answer — see Interview tab)</span>" : esc(t.answer) + (t.dont_know ? ' <span class="badge">don\'t know</span>' : "")}</div></div>`).join("") : '<p class="hint">No questions yet.</p>'}
    <h3>Cited as evidence by</h3>
    ${ep.evidence.length ? ep.evidence.map((e) => `<div class="quote">“${esc(e.quotation)}” <span class="src">${e.role} · ${esc(e.proposition)} ${statusBadge(e.status)} ${e.verified ? "" : '<span class="badge unverified">unverified quotation</span>'}</span></div>`).join("") : '<p class="hint">No assumption cites this episode yet.</p>'}
    ${ep.predictions.length ? `<h3>Prediction tests</h3><p class="hint">See the Prediction test tab for the comparison table.</p>` : ""}
    <details><summary>Model interpretations for this episode (${ep.interpretations.length}) — never used as evidence</summary>
      ${ep.interpretations.map((i) => `<div class="interp"><span class="tag">${esc(i.stage)}</span> <code>${esc(JSON.stringify(i.content)).slice(0, 600)}</code></div>`).join("")}
    </details>`;
  $("#ep-process")?.addEventListener("click", (e) => busy(e.target, async () => {
    const r = await api("POST", `/api/episodes/${id}/process`);
    const n = r.hypothesize.assumptions.length;
    toast(r.hypothesize.cannot_tell ? "The model found nothing pointing to a recurring assumption in this episode." : `Analysed: ${n} candidate(s) proposed or updated, ${r.critiques.length} critiqued.`);
    openEpisode(id); refreshStatus();
  }));
  $("#ep-extract")?.addEventListener("click", (e) => busy(e.target, async () => { await api("POST", `/api/episodes/${id}/extract`); openEpisode(id); }));
  $("#ep-ask")?.addEventListener("click", (e) => busy(e.target, async () => {
    const r = await api("POST", "/api/interview/next", { episode_id: id });
    if (r.stop) toast(r.reason, 6000); else showView("interview");
  }));
  $("#ep-reserve").addEventListener("click", (e) => busy(e.target, async () => { await api("PATCH", `/api/episodes/${id}/reserved`, { reserved: !ep.reserved }); loadEpisodes(); openEpisode(id); }));
  $("#ep-delete").addEventListener("click", (e) => busy(e.target, async () => {
    if (!confirm("Delete this episode and everything derived from it?")) return;
    await api("DELETE", `/api/episodes/${id}`); state.selectedEpisode = null;
    $("#episode-detail").innerHTML = '<p class="hint">Deleted.</p>'; loadEpisodes(); refreshStatus();
  }));
}

// ------------------------------------------------------------------ assumptions
$("#show-rejected").addEventListener("change", (e) => { state.showRejected = e.target.checked; loadAssumptions(); });

async function loadAssumptions() {
  if (!state.episodes.length) await loadEpisodes().catch(() => {});
  state.assumptions = await api("GET", "/api/assumptions");
  const list = state.assumptions.filter((a) => state.showRejected || a.status !== "rejected");
  $("#assumption-list").innerHTML = list.length ? list.map(renderAssumption).join("") : '<p class="hint">No candidate assumptions yet. Analyse a few episodes first.</p>';
  list.forEach(bindAssumption);
}

const li = (items) => (items && items.length ? `<ul class="plain">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<span class="hint">—</span>');
const evidence = (items) => (items.length ? items.map((e) => `<div class="quote">“${esc(e.quotation)}” <span class="src">${epLink(e.episode_id)} ${e.verified ? '<span class="badge verified">verified</span>' : '<span class="badge unverified">not found in your words — not counted</span>'} ${e.note ? "· " + esc(e.note) : ""} <a href="#" data-del-evidence="${esc(e.id)}" title="Remove this link">✕</a></span></div>`).join("") : '<span class="hint">none</span>');

function renderAssumption(a) {
  const corrections = a.user_corrections.length ? a.user_corrections.map((c) => `<li>${esc(c.text)} <span class="hint">(${esc(c.at)}, v${c.version})</span></li>`).join("") : "";
  return `<div class="card ${a.status === "rejected" ? "rejected" : ""}" data-id="${esc(a.id)}">
    <div class="meta">${statusBadge(a.status)} <span class="badge">${esc(a.kind.replace("_", "/"))}</span> <span>${esc(a.id)}</span> <span>v${a.version}</span>
      <span>${a.supporting_episode_ids.length} supporting episode(s) · ${a.contradicting_episode_ids.length} contradicting</span></div>
    <h2>${esc(a.proposition)}</h2>
    <div class="grid2">
      <div><div class="field"><div class="label">Situations where it may apply</div>${li(a.situations)}</div>
           <div class="field"><div class="label">Supporting episodes and exact quotations</div>${evidence(a.supporting)}</div>
           <div class="field"><div class="label">Contradicting episodes</div>${evidence(a.contradicting)}</div></div>
      <div><div class="field"><div class="label">Alternative explanations <span class="badge">interpretation</span></div>${li(a.alternative_explanations)}</div>
           <div class="field"><div class="label">Prediction for a new situation <span class="badge">interpretation</span></div><div>${esc(a.prediction_new_situation) || '<span class="hint">—</span>'}</div></div>
           <div class="field"><div class="label">What would weaken this explanation</div>${li(a.weakeners)}</div>
           <div class="field"><div class="label">Your corrections</div>${corrections ? `<ul class="plain">${corrections}</ul>` : '<span class="hint">none yet</span>'}</div></div>
    </div>
    <div class="actions">
      <button data-act="critique">Critique</button>
      <button data-act="correct">Correct…</button>
      <button data-act="reject" ${a.status === "rejected" ? "disabled" : ""}>Reject</button>
      <select data-act="status"><option value="">set status…</option>${["proposed", "under examination", "revised", "rejected"].map((s) => `<option value="${s}" ${s === a.status ? "disabled" : ""}>${s}</option>`).join("")}</select>
      <button class="danger" data-act="delete">Delete</button>
    </div>
    <div class="correct-box" style="display:none"><textarea rows="2" placeholder="Say what is wrong or how you would put it (e.g. 'It is only when someone else sets the schedule')"></textarea>
      <div class="actions"><button class="primary" data-act="correct-send">Send correction</button><button class="ghost" data-act="correct-cancel">Cancel</button></div></div>
    <details><summary>Revision history (${a.revisions.length})</summary>${a.revisions.map((r) => `<div class="interp"><span class="tag">v${r.version} → v${r.version + 1} · ${esc(r.trigger)}</span> ${esc(r.reason)}<br><span class="hint">was: ${esc(r.previous.proposition)} [${esc(r.previous.status)}]</span></div>`).join("") || '<span class="hint">none</span>'}</details>
    <details class="interp-box"><summary>Model critique and reasoning (interpretation, never evidence)</summary><div class="interp-content hint">open to load…</div></details>
  </div>`;
}

function bindAssumption(a) {
  const card = document.querySelector(`.card[data-id="${a.id}"]`);
  const q = (s) => card.querySelector(s);
  q('[data-act="critique"]').addEventListener("click", (e) => busy(e.target, async () => { await api("POST", `/api/assumptions/${a.id}/critique`); toast("Critique recorded."); loadAssumptions(); }));
  q('[data-act="correct"]').addEventListener("click", () => (q(".correct-box").style.display = "block"));
  q('[data-act="correct-cancel"]').addEventListener("click", () => (q(".correct-box").style.display = "none"));
  q('[data-act="correct-send"]').addEventListener("click", (e) => busy(e.target, async () => {
    const text = q(".correct-box textarea").value.trim();
    if (!text) return toast("Write the correction first.");
    const r = await api("POST", `/api/assumptions/${a.id}/correct`, { text });
    toast("Revised: " + r.reason, 5000); loadAssumptions();
  }));
  q('[data-act="reject"]').addEventListener("click", (e) => busy(e.target, async () => {
    const text = prompt("Optional: why does this not fit? (your words are kept with the record)") ?? "";
    const r = await api("POST", `/api/assumptions/${a.id}/reject`, { text });
    toast(r.alternative ? "Rejected. The model offered an alternative candidate." : "Rejected.", 5000); loadAssumptions(); refreshStatus();
  }));
  q('[data-act="status"]').addEventListener("change", (e) => busy(null, async () => { if (e.target.value) { await api("PATCH", `/api/assumptions/${a.id}/status`, { status: e.target.value }); loadAssumptions(); } }));
  q('[data-act="delete"]').addEventListener("click", (e) => busy(e.target, async () => { if (confirm("Delete this candidate assumption and its evidence links?")) { await api("DELETE", `/api/assumptions/${a.id}`); loadAssumptions(); refreshStatus(); } }));
  card.querySelectorAll("[data-del-evidence]").forEach((x) => x.addEventListener("click", (e) => busy(null, async () => { e.preventDefault(); await api("DELETE", `/api/evidence/${x.dataset.delEvidence}`); loadAssumptions(); })));
  q(".interp-box").addEventListener("toggle", async (e) => {
    if (!e.target.open) return;
    const full = await api("GET", `/api/assumptions/${a.id}`);
    q(".interp-content").innerHTML = full.interpretations.map((i) => `<div class="interp"><span class="tag">${esc(i.stage)}</span> ${renderInterp(i.content)}</div>`).join("") || "none";
  });
}

function renderInterp(c) {
  if (c.rationale !== undefined) return esc(c.rationale);
  if (c.contradictions !== undefined) return `<b>verdict:</b> ${esc(c.verdict)}<br><b>unsupported leaps:</b> ${li(c.unsupported_leaps)}<b>resolving questions:</b> ${li(c.resolving_questions.map((r) => r.question))}`;
  return `<code>${esc(JSON.stringify(c)).slice(0, 500)}</code>`;
}

// ------------------------------------------------------------------ interview
async function loadInterview() {
  if (!state.episodes.length) await loadEpisodes().catch(() => {});
  const turns = await api("GET", "/api/interview");
  const pending = turns.find((t) => t.answer == null);
  const assumptions = await api("GET", "/api/assumptions");
  const propOf = (id) => assumptions.find((a) => a.id === id)?.proposition || id;
  const box = $("#interview-current");
  if (pending) {
    const ep = state.episodes.find((e) => e.id === pending.episode_id);
    box.innerHTML = `
      <h2>One question</h2>
      <div class="hint">About ${epLink(pending.episode_id)}${ep ? `: <em>${esc(ep.situation.slice(0, 220))}</em>` : ""}</div>
      <p class="question">${esc(pending.question)}</p>
      ${pending.discriminates_between.length ? `<details><summary>Why this question (model rationale)</summary><div class="interp">${esc(pending.rationale)}<ul class="plain">${pending.discriminates_between.map((id) => `<li>${esc(propOf(id))}</li>`).join("")}</ul></div></details>` : ""}
      <textarea id="answer" rows="4" placeholder="Answer in your own words. 'I don't know' is a fine answer."></textarea>
      <div class="actions"><button class="primary" id="send-answer">Answer</button><button id="dont-know">I don't know</button><button class="ghost" id="skip-q">Skip this question</button></div>`;
    const send = (text) => (e) => busy(e.target, async () => {
      const r = await api("POST", `/api/interview/${pending.id}/answer`, { answer: text ?? $("#answer").value });
      const summary = r.updates.filter((u) => u.action !== "keep").map((u) => `${u.action}: ${propOf(u.assumption_id).slice(0, 50)}`).join("; ");
      toast(r.note || (summary ? "Updated — " + summary : "Recorded; no candidate changed."), 7000);
      if (!r.stop) await api("POST", "/api/interview/next", {}).catch(() => {});
      loadInterview(); refreshStatus();
    });
    $("#send-answer").addEventListener("click", send(null));
    $("#dont-know").addEventListener("click", send("I don't know."));
    $("#skip-q").addEventListener("click", (e) => busy(e.target, async () => { await api("DELETE", `/api/interview/${pending.id}`); loadInterview(); }));
  } else {
    box.innerHTML = `<h2>Interview</h2><p class="hint">The model asks one question at a time, grounded in one of your episodes and chosen to tell competing candidates apart. Each answer is used to revise the candidates; a rejection prompts revision, never a label.</p>
      <div class="actions"><button class="primary" id="ask-next">Ask the next question</button></div><p id="ask-note" class="hint"></p>`;
    $("#ask-next").addEventListener("click", (e) => busy(e.target, async () => {
      const r = await api("POST", "/api/interview/next", {});
      if (r.stop) $("#ask-note").textContent = "The model stopped: " + r.reason; else loadInterview();
    }));
  }
  $("#turn-list").innerHTML = turns.filter((t) => t.answer != null).reverse().map((t) => `<li><div><b>Q</b> ${esc(t.question)}</div><div><b>A</b> ${esc(t.answer)}${t.dont_know ? ' <span class="badge">don\'t know</span>' : ""}</div><div class="meta">${epLink(t.episode_id)} · ${esc(t.answered_at)}</div></li>`).join("") || '<li class="hint">No answered questions yet.</li>';
}

// ------------------------------------------------------------------ holdout
async function loadHoldout() {
  if (!state.episodes.length) await loadEpisodes().catch(() => {});
  const [h, assumptions] = await Promise.all([api("GET", "/api/holdout"), api("GET", "/api/assumptions")]);
  const propOf = (id) => assumptions.find((a) => a.id === id)?.proposition || `(deleted ${id})`;
  const fmt = (x) => (x == null ? "—" : Number(x).toFixed(2));
  $("#holdout-list").innerHTML = h.reserved.length ? h.reserved.map((ep) => {
    const preds = h.predictions.filter((p) => p.episode_id === ep.id);
    const batches = [...new Set(preds.map((p) => p.batch_id))];
    const latest = batches.length ? preds.filter((p) => p.batch_id === batches[batches.length - 1]) : [];
    const row = (p) => `<tr class="${p.assumption_id ? "" : "baseline"}"><td>${p.assumption_id ? esc(propOf(p.assumption_id)) : "<em>Generic baseline (no assumption)</em>"}</td><td>${esc(p.predicted_thought)}</td><td>${esc(p.predicted_emotion)}</td><td>${esc(p.predicted_action)}</td><td>${fmt(p.confidence)}</td><td class="score">${fmt(p.match_score)}<br><span class="hint">${esc(p.comparison)}</span></td>
      <td>${["accurate", "partly", "wrong"].map((r) => `<button data-rate="${r}" data-pid="${p.id}" class="${p.user_rating === r ? "primary" : ""}">${r}</button>`).join(" ")}</td></tr>`;
    return `<div class="card">
      <div class="meta"><span class="badge reserved">reserved</span> <span>${esc(ep.id)}</span></div>
      <h2>${esc(ep.title || "Reserved episode")}</h2>
      <div class="field"><div class="label">Situation given to the model</div><div>${esc(ep.situation)}</div></div>
      <div class="actions"><button class="primary" data-run="${esc(ep.id)}">${latest.length ? "Run again" : "Run prediction test"}</button></div>
      ${latest.length ? `<table><thead><tr><th>Candidate assumption</th><th>Predicted thought</th><th>Predicted feeling</th><th>Predicted action</th><th>Conf.</th><th>Match</th><th>Your rating</th></tr></thead><tbody>${latest.map(row).join("")}</tbody></table>
      <h3>What you actually recorded</h3><div class="locked grid2"><div><div class="label">Immediate thought</div>${esc(ep.immediate_thought)}</div><div><div class="label">Emotion and body</div>${esc(ep.emotion_body)}</div><div><div class="label">Action and outcome</div>${esc(ep.action_outcome)}</div></div>` : ""}
    </div>`;
  }).join("") : '<p class="hint">No reserved episodes. Mark an episode as reserved (or write a new one with the checkbox ticked).</p>';
  document.querySelectorAll("[data-run]").forEach((b) => b.addEventListener("click", (e) => busy(e.target, async () => { await api("POST", `/api/holdout/${b.dataset.run}/run`); loadHoldout(); })));
  document.querySelectorAll("[data-rate]").forEach((b) => b.addEventListener("click", (e) => busy(e.target, async () => { await api("PATCH", `/api/predictions/${b.dataset.pid}/rating`, { rating: b.dataset.rate }); loadHoldout(); })));
}

showView("episodes");
