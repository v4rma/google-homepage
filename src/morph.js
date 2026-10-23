import * as THREE from 'three';
import vertexShader from './shaders/morph.vert?raw';
import fragmentShader from './shaders/morph.frag?raw';
import { loadTextureFromBlob, loadTextureFromUrl } from './textures.js';

export const DEFAULT_PARAMS = {
  progress: 0,
  intensity: 1,
  softness: 0.5,
  mode: 'fluid', // 'fluid' | 'crossfade'
  fit: 'cover', // 'cover' | 'contain'
  noiseScale: 1,
  warpAmplitude: 0.075,
  flowSpeed: 1,
  transitionWidth: 0.8,
  maskScale: 1,
  renderScale: 1,
};

export class FluidMorph {
  constructor(canvas) {
    this.canvas = canvas;
    this.params = { ...DEFAULT_PARAMS };

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.autoClear = false;

    const placeholder = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    placeholder.colorSpace = THREE.SRGBColorSpace;
    placeholder.needsUpdate = true;

    this.uniforms = {
      uTexA: { value: placeholder },
      uTexB: { value: placeholder },
      uSizeA: { value: new THREE.Vector2(1, 1) },
      uSizeB: { value: new THREE.Vector2(1, 1) },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uProgress: { value: 0 },
      uTime: { value: 0 },
      uMode: { value: 1 },
      uFit: { value: 1 },
      uIntensity: { value: 1 },
      uWarpAmplitude: { value: 0.075 },
      uNoiseScale: { value: 1 },
      uFlowSpeed: { value: 1 },
      uSoftness: { value: 0.5 },
      uTransitionWidth: { value: 0.8 },
      uMaskScale: { value: 1 },
    };

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader,
      fragmentShader,
      depthTest: false,
      depthWrite: false,
    });

    // A single oversized triangle covers the viewport without a diagonal seam.
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
    );
    geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2));

    this.scene = new THREE.Scene();
    this.scene.add(new THREE.Mesh(geometry, this.material));
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this.images = { A: null, B: null };
    this.resize();
  }

  get capabilities() {
    return this.renderer.capabilities;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 3) * this.params.renderScale;
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.uniforms.uResolution.value.copy(size);
  }

  setParams(partial) {
    Object.assign(this.params, partial);
    if ('renderScale' in partial) this.resize();
  }

  async setImage(slot, source, label) {
    const loaded =
      source instanceof Blob
        ? await loadTextureFromBlob(source, this.renderer)
        : await loadTextureFromUrl(source, this.renderer);

    const previous = this.images[slot];
    this.images[slot] = { ...loaded, label };
    this.uniforms[`uTex${slot}`].value = loaded.texture;
    this.uniforms[`uSize${slot}`].value.set(loaded.width, loaded.height);
    if (previous) {
      previous.texture.image?.close?.();
      previous.texture.dispose();
    }
    return this.images[slot];
  }

  render(time) {
    const p = this.params;
    const u = this.uniforms;
    u.uTime.value = time;
    u.uProgress.value = p.progress;
    u.uMode.value = p.mode === 'fluid' ? 1 : 0;
    u.uFit.value = p.fit === 'cover' ? 1 : 0;
    u.uIntensity.value = p.intensity;
    u.uWarpAmplitude.value = p.warpAmplitude;
    u.uNoiseScale.value = p.noiseScale;
    u.uFlowSpeed.value = p.flowSpeed;
    u.uSoftness.value = p.softness;
    u.uTransitionWidth.value = p.transitionWidth;
    u.uMaskScale.value = p.maskScale;
    this.renderer.render(this.scene, this.camera);
  }
}
