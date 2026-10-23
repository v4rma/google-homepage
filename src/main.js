import { FluidMorph, DEFAULT_PARAMS } from './morph.js';
import { isSupportedImage } from './textures.js';

const $ = (sel) => document.querySelector(sel);

const canvas = $('#view');
const morph = new FluidMorph(canvas);

// ---------------------------------------------------------------------------
// Deterministic clock. Everything (idle motion and playback) derives from a
// single time value in seconds. `?t=` and `?capture` can freeze/step it.
// ---------------------------------------------------------------------------
const query = new URLSearchParams(location.search);
const clock = {
  origin: performance.now(),
  fixed: query.has('t') ? parseFloat(query.get('t')) : null,
  now() {
    return this.fixed ?? (performance.now() - this.origin) / 1000;
  },
};

const transport = {
  playing: false,
  direction: 1,
  duration: 20,
  lastTime: 0,
};

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
const els = {
  progress: $('#progress'),
  progressOut: $('#progress-out'),
  play: $('#play'),
  duration: $('#duration'),
  durationOut: $('#duration-out'),
  stats: $('#stats'),
  nameA: $('#name-a'),
  nameB: $('#name-b'),
};

function setProgress(value, { fromSlider = false } = {}) {
  const p = Math.min(1, Math.max(0, value));
  morph.setParams({ progress: p });
  if (!fromSlider) els.progress.value = p;
  els.progressOut.textContent = `${Math.round(p * 100)}%`;
}

els.progress.addEventListener('input', () => setProgress(parseFloat(els.progress.value), { fromSlider: true }));

function setPlaying(playing) {
  transport.playing = playing;
  transport.lastTime = clock.now();
  if (playing) {
    const p = morph.params.progress;
    if (p >= 1) transport.direction = -1;
    else if (p <= 0) transport.direction = 1;
  }
  els.play.textContent = playing ? 'Pause' : 'Play';
}
els.play.addEventListener('click', () => setPlaying(!transport.playing));

els.duration.addEventListener('input', () => {
  transport.duration = parseFloat(els.duration.value);
  els.durationOut.textContent = `${transport.duration} s`;
});

function bindRange(id, key, format = (v) => v.toFixed(2)) {
  const input = $(`#${id}`);
  const out = $(`#${id}-out`);
  const apply = () => {
    const v = parseFloat(input.value);
    morph.setParams({ [key]: v });
    out.textContent = format(v);
  };
  input.addEventListener('input', apply);
  input.value = DEFAULT_PARAMS[key];
  apply();
}

bindRange('intensity', 'intensity');
bindRange('softness', 'softness');
bindRange('noiseScale', 'noiseScale');
bindRange('warpAmplitude', 'warpAmplitude', (v) => v.toFixed(3));
bindRange('flowSpeed', 'flowSpeed');
bindRange('transitionWidth', 'transitionWidth');
bindRange('maskScale', 'maskScale');

function bindSegmented(attr, key, parse = (v) => v) {
  const buttons = [...document.querySelectorAll(`[data-${attr}]`)];
  const select = (value) => {
    morph.setParams({ [key]: value });
    buttons.forEach((b) => b.classList.toggle('active', parse(b.dataset[attr]) === value));
  };
  buttons.forEach((b) => b.addEventListener('click', () => select(parse(b.dataset[attr]))));
  return select;
}

const selectMode = bindSegmented('mode', 'mode');
const selectFit = bindSegmented('fit', 'fit');
const selectRender = bindSegmented('render', 'renderScale', parseFloat);

// Hide / show / fullscreen
const setControlsHidden = (hidden) => document.body.classList.toggle('controls-hidden', hidden);
$('#hide-controls').addEventListener('click', () => setControlsHidden(true));
$('#show-controls').addEventListener('click', () => setControlsHidden(false));
$('#fullscreen').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
});
// With controls hidden, the "controls" tab only appears while the mouse moves.
let mouseTimer;
window.addEventListener('pointermove', () => {
  document.body.classList.add('mouse-active');
  clearTimeout(mouseTimer);
  mouseTimer = setTimeout(() => document.body.classList.remove('mouse-active'), 3500);
});
window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.key === 'h' || e.key === 'H') setControlsHidden(!document.body.classList.contains('controls-hidden'));
  if (e.key === ' ') {
    e.preventDefault();
    setPlaying(!transport.playing);
  }
});

// ---------------------------------------------------------------------------
// Images: file inputs + drag & drop
// ---------------------------------------------------------------------------
let toastTimer;
function toast(message) {
  let el = $('#toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('visible'), 3200);
}

async function loadInto(slot, source, label) {
  try {
    const info = await morph.setImage(slot, source, label);
    (slot === 'A' ? els.nameA : els.nameB).textContent = label;
    if (info.downscaled) {
      toast(`${label} was downscaled to ${info.width}×${info.height} to fit this GPU's texture limit`);
    }
    updateStats();
  } catch (err) {
    console.error(err);
    toast(`Could not load ${label}`);
  }
}

$('#file-a').addEventListener('change', (e) => {
  const f = e.target.files?.[0];
  if (f && isSupportedImage(f)) loadInto('A', f, f.name);
});
$('#file-b').addEventListener('change', (e) => {
  const f = e.target.files?.[0];
  if (f && isSupportedImage(f)) loadInto('B', f, f.name);
});

const dropHalves = document.querySelectorAll('.drop-half');
let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth++;
  document.body.classList.add('dragging');
});
window.addEventListener('dragover', (e) => {
  e.preventDefault();
  const left = e.clientX < window.innerWidth / 2;
  dropHalves[0].classList.toggle('hot', left);
  dropHalves[1].classList.toggle('hot', !left);
});
window.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    document.body.classList.remove('dragging');
  }
});
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  const files = [...(e.dataTransfer?.files || [])].filter(isSupportedImage);
  if (files.length === 0) {
    toast('Drop PNG, JPG or WebP images');
    return;
  }
  if (files.length >= 2) {
    loadInto('A', files[0], files[0].name);
    loadInto('B', files[1], files[1].name);
  } else {
    const slot = e.clientX < window.innerWidth / 2 ? 'A' : 'B';
    loadInto(slot, files[0], files[0].name);
  }
});

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
let frameCount = 0;
let fpsTime = performance.now();
let fps = 0;

function updateStats() {
  const a = morph.images.A;
  const b = morph.images.B;
  const size = morph.uniforms.uResolution.value;
  const fmt = (img) => (img ? `${img.width}×${img.height}` : '—');
  els.stats.textContent =
    `A ${fmt(a)} · B ${fmt(b)} · canvas ${size.x}×${size.y} · ` +
    `max tex ${morph.capabilities.maxTextureSize} · ${fps.toFixed(0)} fps`;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
function frame() {
  const now = clock.now();

  if (transport.playing) {
    const dt = now - transport.lastTime;
    transport.lastTime = now;
    let p = morph.params.progress + (transport.direction * dt) / transport.duration;
    if (p >= 1) {
      p = 1;
      transport.direction = -1;
    } else if (p <= 0) {
      p = 0;
      transport.direction = 1;
    }
    setProgress(p);
  }

  morph.render(now);

  frameCount++;
  const t = performance.now();
  if (t - fpsTime > 1000) {
    fps = (frameCount * 1000) / (t - fpsTime);
    frameCount = 0;
    fpsTime = t;
    updateStats();
  }
  requestAnimationFrame(frame);
}

window.addEventListener('resize', () => {
  morph.resize();
  updateStats();
});

// ---------------------------------------------------------------------------
// Boot: sample artworks, optional URL overrides for deterministic captures
// ---------------------------------------------------------------------------
async function boot() {
  if (query.has('p')) setProgress(parseFloat(query.get('p')));
  if (query.get('controls') === '0') setControlsHidden(true);
  if (query.has('mode')) selectMode(query.get('mode'));
  if (query.has('fit')) selectFit(query.get('fit'));
  if (query.has('render')) selectRender(parseFloat(query.get('render')));

  const urlA = query.get('a') || '/samples/imageA.webp';
  const urlB = query.get('b') || '/samples/imageB.webp';
  await Promise.all([
    loadInto('A', urlA, urlA.split('/').pop()),
    loadInto('B', urlB, urlB.split('/').pop()),
  ]);

  if (query.get('play') === '1') setPlaying(true);

  // Debug / capture hook: freeze the clock at `time` and render one frame.
  window.__fluidMorph = {
    morph,
    seek(time, progress) {
      clock.fixed = time;
      if (progress !== undefined) setProgress(progress);
      morph.render(time);
    },
  };
  document.body.dataset.ready = '1';
  requestAnimationFrame(frame);
}

boot();
