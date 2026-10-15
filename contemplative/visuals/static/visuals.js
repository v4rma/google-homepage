/*
 * Contemplative visuals: a continuous parameterised shader over the 6-axis
 * design space (report Section 2).  Parameters arrive over a websocket from the
 * backend and every change is a smooth morph (>= 10 s, server-specified).
 *
 * Fixed: mean luminance, contrast, field size.  Variable: speed, fractal D,
 * symmetry order (+ concentricity), colour temperature, 0.1 Hz breath
 * pulsation amplitude, texture latent.
 *
 * Temporal safety: the only periodic term is the 0.1 Hz pulsation; the noise
 * field evolves continuously (3-D noise, time as the third axis) with no
 * frame-rate-locked or 3-30 Hz components.  `?diag=1` measures the luminance
 * spectrum of the actual rendering (see docs/temporal-spectrum-check.md).
 */
(() => {
  const AXES = ["speed", "fractal", "symmetry", "color", "breath", "texture"];
  const DEFAULTS = { speed: 0.2, fractal: 0.33, symmetry: 0.5, color: 0.5, breath: 0.5, texture: 0.5 };
  const MEAN_LUM = 0.10;   // linear luminance of the field and the surround (fixed)
  const CONTRAST = 0.16;   // peak texture modulation around MEAN_LUM (fixed)
  const FIELD_R = 0.46;    // field radius in units of the shorter screen side (fixed)

  const canvas = document.getElementById("gl");
  const gl = canvas.getContext("webgl", { antialias: false, alpha: false, preserveDrawingBuffer: true });
  if (!gl) { document.body.innerHTML = "<p style='color:#888;font:16px sans-serif;padding:2em'>WebGL is not available.</p>"; return; }

  const VS = `attribute vec2 aPos; void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }`;
  const FS = `
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform vec4 uA;      // speed, fractal, symmetry, color
uniform vec2 uB;      // breath, texture
uniform float uDrift; // accumulated motion (integral of speed), time axis of the noise
uniform float uBaseline; // 0 = stimulus, 1 = uniform field
uniform float uMeanLum, uContrast, uFieldR;

const float PI = 3.14159265359;

// ---- 3-D value noise ------------------------------------------------------
float hash3(vec3 p){ p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3)); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vnoise(vec3 p){
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float n000 = hash3(i), n100 = hash3(i + vec3(1,0,0)), n010 = hash3(i + vec3(0,1,0)), n110 = hash3(i + vec3(1,1,0));
  float n001 = hash3(i + vec3(0,0,1)), n101 = hash3(i + vec3(1,0,1)), n011 = hash3(i + vec3(0,1,1)), n111 = hash3(i + vec3(1,1,1));
  float x00 = mix(n000, n100, f.x), x10 = mix(n010, n110, f.x), x01 = mix(n001, n101, f.x), x11 = mix(n011, n111, f.x);
  return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z) * 2.0 - 1.0;
}
// fBm with Hurst exponent H: per-octave gain 2^-H.  Fractal dimension of the
// texture's iso-contours is D = 2 - H (Hagerhall's silhouettes: D 1.1-1.7).
float fbm(vec3 p, float H){
  float g = exp2(-H), a = 1.0, s = 0.0, norm = 0.0;
  for (int i = 0; i < 6; i++){ s += a * vnoise(p); norm += a; p = p * 2.03 + vec3(7.1, 3.7, 1.9); a *= g; }
  return s / norm;
}
float ridged(vec3 p, float H){
  float g = exp2(-H), a = 1.0, s = 0.0, norm = 0.0;
  for (int i = 0; i < 6; i++){ s += a * (1.0 - 2.0 * abs(vnoise(p))); norm += a; p = p * 2.03 + vec3(1.3, 9.2, 4.4); a *= g; }
  return s / norm;
}
float warped(vec3 p, float H){
  vec3 q = vec3(fbm(p + vec3(0.0, 0.0, 1.7), H), fbm(p + vec3(5.2, 1.3, 2.9), H), 0.0);
  return fbm(p + 1.2 * q, H);
}
// texture latent: 1-D path fbm -> ridged -> warped, plus a seed offset
float texture_value(vec3 p, float H, float latent){
  float wA = max(0.0, 1.0 - 2.0 * latent), wB = 1.0 - abs(2.0 * latent - 1.0), wC = max(0.0, 2.0 * latent - 1.0);
  vec3 off = vec3(latent * 13.7, latent * 5.3, 0.0);
  float v = 0.0;
  if (wA > 0.001) v += wA * fbm(p + off, H);
  if (wB > 0.001) v += wB * ridged(p + off, H);
  if (wC > 0.001) v += wC * warped(p + off, H);
  return v;
}
// kaleidoscopic fold of order n (n >= 2); n == 1 returns the input unchanged
vec2 fold(vec2 p, float n){
  if (n < 1.5) return p;
  float r = length(p), th = atan(p.y, p.x), sector = 2.0 * PI / n;
  th = abs(mod(th + sector * 0.5, sector) - sector * 0.5);
  return r * vec2(cos(th), sin(th));
}
vec2 sample_coords(vec2 p, float n, float conc){
  vec2 q = fold(p, n);
  float r = length(q), th = atan(q.y, q.x);
  // concentricity: blend towards polar coordinates so structure follows rings
  vec2 polar = vec2(r * 1.6, th * max(r, 0.05) * 1.2);
  return mix(q, polar, conc);
}
vec3 tint(float colorTemp){
  vec3 warm = vec3(1.0, 0.70, 0.38);   // ~2700 K
  vec3 cool = vec3(0.70, 0.82, 1.0);   // ~8000 K
  vec3 c = mix(warm, cool, colorTemp);
  float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
  return c / y;                        // unit luminance: colour never changes brightness
}
void main(){
  vec2 p = (gl_FragCoord.xy - 0.5 * uRes) / min(uRes.x, uRes.y);
  float speed = uA.x, fractal = uA.y, symmetry = uA.z, colorTemp = uA.w, breath = uB.x, latent = uB.y;
  // 0.1 Hz breath pulsation: a gentle radial zoom of at most 5 %
  float pulse = 1.0 + 0.05 * breath * sin(2.0 * PI * 0.1 * uTime);
  vec2 pp = p / pulse;
  float H = 2.0 - (1.1 + 0.6 * fractal);
  float n = 1.0 + 11.0 * symmetry;
  float nLo = floor(n), nHi = nLo + 1.0, nf = n - nLo;
  float conc = 0.55 * symmetry;
  vec3 zAxis = vec3(0.0, 0.0, uDrift);
  float scale = 2.6;
  float vLo = texture_value(vec3(sample_coords(pp, nLo, conc) * scale, 0.0) + zAxis, H, latent);
  float v = vLo;
  if (nf > 0.001) {
    float vHi = texture_value(vec3(sample_coords(pp, nHi, conc) * scale, 0.0) + zAxis, H, latent);
    v = mix(vLo, vHi, smoothstep(0.0, 1.0, nf));
  }
  // fixed field size: texture contrast fades to zero at the field edge, surround = mean grey
  float r = length(p);
  float win = 1.0 - smoothstep(uFieldR * 0.82, uFieldR, r);
  float lum = uMeanLum + uContrast * v * win;
  lum = mix(lum, uMeanLum, uBaseline);
  lum = clamp(lum, 0.0, 1.0);
  vec3 rgb = lum * mix(tint(colorTemp), vec3(1.0), uBaseline);
  // faint fixation mark (constant, negligible luminance)
  float dot_ = smoothstep(0.006, 0.004, r);
  rgb = mix(rgb, vec3(uMeanLum * 0.55), dot_ * 0.6);
  gl_FragColor = vec4(pow(rgb, vec3(1.0 / 2.2)), 1.0);
}`;

  function compile(type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog, "aPos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  const U = {};
  for (const n of ["uRes", "uTime", "uA", "uB", "uDrift", "uBaseline", "uMeanLum", "uContrast", "uFieldR"]) U[n] = gl.getUniformLocation(prog, n);

  // ---- parameter morphing --------------------------------------------------------
  const state = {
    cur: { ...DEFAULTS }, from: { ...DEFAULTS }, to: { ...DEFAULTS },
    baseCur: 1, baseFrom: 1, baseTo: 1,
    tStart: performance.now() / 1000, dur: 10, drift: 0, mode: "idle",
  };
  function smooth(u) { u = Math.min(1, Math.max(0, u)); return u * u * (3 - 2 * u); }
  function setTarget(params, mode, morphS) {
    const now = performance.now() / 1000;
    const u = smooth((now - state.tStart) / Math.max(state.dur, 1e-3));
    for (const a of AXES) { state.from[a] = state.from[a] + (state.to[a] - state.from[a]) * u; }
    state.baseFrom = state.baseFrom + (state.baseTo - state.baseFrom) * u;
    if (params) for (const a of AXES) if (a in params) state.to[a] = Math.min(1, Math.max(0, +params[a]));
    state.baseTo = mode === "stimulus" ? 0 : 1;
    state.mode = mode;
    state.tStart = now;
    // never shorter than 10 s, except the initial sync on connect (morphS === 0)
    state.dur = morphS === 0 ? 0 : Math.max(+morphS || 10, 10);
  }
  function update(dt) {
    const now = performance.now() / 1000;
    const u = state.dur <= 0 ? 1 : smooth((now - state.tStart) / state.dur);
    for (const a of AXES) state.cur[a] = state.from[a] + (state.to[a] - state.from[a]) * u;
    state.baseCur = state.baseFrom + (state.baseTo - state.baseFrom) * u;
    // motion: integral of speed so speed changes are themselves smooth
    state.drift += 0.08 * state.cur.speed * dt;
  }

  // ---- cues (calibration only; faint text, fades in/out) ------------------------
  const cueEl = document.getElementById("cue");
  let cueTimer = null;
  function cue(text) {
    cueEl.textContent = text; cueEl.classList.add("on");
    clearTimeout(cueTimer); cueTimer = setTimeout(() => cueEl.classList.remove("on"), 4000);
  }
  const CUES = { calibration_fixation: "rest your gaze on the field", calibration_breath: "follow the breath" };

  // ---- websocket -----------------------------------------------------------------
  let ws = null;
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/visuals`);
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "visuals") {
        setTarget(m.params, m.mode, m.immediate ? 0 : m.morph_s);
        if (CUES[m.mode]) cue(CUES[m.mode]);
      }
    };
    ws.onclose = () => setTimeout(connect, 1500);
  }
  connect();
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

  // ---- diagnostics: luminance spectrum of the actual rendering -------------------
  const diag = { on: new URLSearchParams(location.search).get("diag") === "1", t: [], y: [], patches: [], secs: 20 };
  const diagEl = document.getElementById("diag");
  const px = new Uint8Array(8 * 8 * 4);
  function diagSample(now) {
    if (!diag.on) return;
    if (diag.patches.length === 0) {
      const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
      diag.patches = [[w / 2, h / 2], [w / 2 + w * 0.12, h / 2], [w / 2, h / 2 + h * 0.15], [w / 2 - w * 0.2, h / 2 - h * 0.1], [w / 2 + w * 0.05, h / 2 - h * 0.22]];
      diag.y = diag.patches.map(() => []);
      diagEl.classList.add("on");
    }
    diag.t.push(now);
    diag.patches.forEach(([x, y], i) => {
      gl.readPixels(Math.round(x) - 4, Math.round(y) - 4, 8, 8, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let s = 0; for (let k = 0; k < 64; k++) s += 0.2126 * px[4 * k] + 0.7152 * px[4 * k + 1] + 0.0722 * px[4 * k + 2];
      diag.y[i].push(s / 64);
    });
    if (diag.t[diag.t.length - 1] - diag.t[0] >= diag.secs) diagReport();
  }
  function diagReport() {
    const t = diag.t, N = t.length, T = t[N - 1] - t[0], fsHz = (N - 1) / T;
    const lines = [`luminance spectrum, ${N} frames, ${T.toFixed(1)} s, ${fsHz.toFixed(1)} fps, mode=${state.mode}`];
    const report = { type: "diag", fps: fsHz, seconds: T, mode: state.mode, params: state.cur, patches: [] };
    diag.y.forEach((y, i) => {
      const mean = y.reduce((a, b) => a + b, 0) / N;
      const yd = y.map(v => v - mean);
      // naive DFT on 0.1 Hz grid up to 40 Hz (N is a few hundred; fine)
      let pTot = 1e-12, pBand = 0, fPeak = 0, pPeak = 0, pLow = 0;
      for (let f = 0.1; f <= Math.min(40, fsHz / 2); f += 0.1) {
        let re = 0, im = 0;
        for (let k = 0; k < N; k++) { const ph = 2 * Math.PI * f * (t[k] - t[0]); re += yd[k] * Math.cos(ph); im -= yd[k] * Math.sin(ph); }
        const p = (re * re + im * im) / (N * N);
        pTot += p;
        if (f < 1.0) pLow += p;
        if (f >= 3 && f <= 30) { pBand += p; if (p > pPeak) { pPeak = p; fPeak = f; } }
      }
      const frac = pBand / pTot;
      const line = `patch ${i}: mean ${mean.toFixed(1)}/255, sd ${(Math.sqrt(yd.reduce((a, b) => a + b * b, 0) / N)).toFixed(2)}, 3-30 Hz share ${(100 * frac).toFixed(1)} %, peak ${fPeak.toFixed(1)} Hz (${(100 * pPeak / pTot).toFixed(1)} % of power), <1 Hz share ${(100 * pLow / pTot).toFixed(1)} %`;
      lines.push(line);
      report.patches.push({ mean, band_3_30_share: frac, peak_hz: fPeak, peak_share: pPeak / pTot, low_share: pLow / pTot });
    });
    diagEl.textContent = lines.join("\n");
    console.log(lines.join("\n"));
    send(report);
    diag.t = []; diag.y = diag.patches.map(() => []);
  }

  // ---- UI: fullscreen, cursor, ratings ----------------------------------------
  const hint = document.getElementById("hint");
  setTimeout(() => hint.classList.add("hidden"), 8000);
  let cursorTimer = null;
  function showCursor() {
    document.body.classList.add("show-cursor"); hint.classList.remove("hidden");
    clearTimeout(cursorTimer);
    cursorTimer = setTimeout(() => { document.body.classList.remove("show-cursor"); hint.classList.add("hidden"); }, 2500);
  }
  document.addEventListener("mousemove", showCursor);
  function goFullscreen() { const el = document.documentElement; if (!document.fullscreenElement && el.requestFullscreen) el.requestFullscreen().catch(() => {}); }
  canvas.addEventListener("click", goFullscreen);
  document.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (k === "f") goFullscreen();
    else if (k === "b") { send({ type: "rate", better: true }); cue("noted: better"); }
    else if (k === "w") { send({ type: "rate", better: false }); cue("noted: worse"); }
    else if (k === "d") { diag.on = !diag.on; diagEl.classList.toggle("on", diag.on); diag.t = []; diag.patches = []; }
  });

  // ---- render loop -----------------------------------------------------------------
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h); }
  }
  let last = performance.now() / 1000;
  const t0 = last;
  function frame() {
    const now = performance.now() / 1000;
    const dt = Math.min(0.1, now - last); last = now;
    resize();
    update(dt);
    const c = state.cur;
    gl.uniform2f(U.uRes, canvas.width, canvas.height);
    gl.uniform1f(U.uTime, now - t0);
    gl.uniform4f(U.uA, c.speed, c.fractal, c.symmetry, c.color);
    gl.uniform2f(U.uB, c.breath, c.texture);
    gl.uniform1f(U.uDrift, state.drift);
    gl.uniform1f(U.uBaseline, state.baseCur);
    gl.uniform1f(U.uMeanLum, MEAN_LUM);
    gl.uniform1f(U.uContrast, CONTRAST);
    gl.uniform1f(U.uFieldR, FIELD_R);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    diagSample(now);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
