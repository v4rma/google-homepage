// Fluid morph: multiscale curl-noise advection + spatially varying blend.
//
// The whole frame is treated as a single incompressible fluid. Image A is
// advected forward along the velocity field (sampled by tracing backwards),
// image B is sampled by tracing forwards so its pigment arrives *along the
// same currents* and settles into its rest position as progress -> 1.
// Source textures are sampled exactly once each per frame; nothing is fed
// back, so detail is never resampled or blurred over time.

precision highp float;

uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform vec2 uResolution;   // canvas size in physical pixels
uniform vec2 uSizeA;        // image A size in pixels
uniform vec2 uSizeB;        // image B size in pixels

uniform float uProgress;    // 0 = A, 1 = B
uniform float uTime;        // seconds, deterministic
uniform float uMode;        // 0 = crossfade, 1 = fluid morph
uniform float uFit;         // 0 = contain, 1 = cover

uniform float uIntensity;        // user "fluid intensity" multiplier
uniform float uWarpAmplitude;    // max advection distance in frame-height units
uniform float uNoiseScale;       // base frequency of the flow field
uniform float uFlowSpeed;        // temporal evolution rate of the field
uniform float uSoftness;         // feather of the transition front (0..1)
uniform float uTransitionWidth;  // how staggered regions are in progress (0..1)
uniform float uMaskScale;        // base frequency of the transition mask

varying vec2 vUv;

#define FLOW_STEPS 4

// ---------------------------------------------------------------------------
// psrdnoise (2D, non-periodic variant), Stefan Gustavson & Ian McEwan, MIT.
// Returns simplex-style gradient noise with an analytic gradient. `alpha`
// rotates the gradient vectors, which gives smooth temporal evolution
// ("flow noise") without re-hashing the lattice.
// ---------------------------------------------------------------------------
float psrdnoise(vec2 x, float alpha, out vec2 gradient) {
  vec2 uv = vec2(x.x + x.y * 0.5, x.y);
  vec2 i0 = floor(uv);
  vec2 f0 = fract(uv);
  float cmp = step(f0.y, f0.x);
  vec2 o1 = vec2(cmp, 1.0 - cmp);
  vec2 i1 = i0 + o1;
  vec2 i2 = i0 + 1.0;
  vec2 v0 = vec2(i0.x - i0.y * 0.5, i0.y);
  vec2 v1 = vec2(v0.x + o1.x - o1.y * 0.5, v0.y + o1.y);
  vec2 v2 = vec2(v0.x + 0.5, v0.y + 1.0);
  vec2 x0 = x - v0;
  vec2 x1 = x - v1;
  vec2 x2 = x - v2;
  vec3 iu = vec3(i0.x, i1.x, i2.x);
  vec3 iv = vec3(i0.y, i1.y, i2.y);
  vec3 hash = mod(iu, 289.0);
  hash = mod((hash * 51.0 + 2.0) * hash + iv, 289.0);
  hash = mod((hash * 34.0 + 10.0) * hash, 289.0);
  vec3 psi = hash * 0.07482 + alpha;
  vec3 gx = cos(psi);
  vec3 gy = sin(psi);
  vec2 g0 = vec2(gx.x, gy.x);
  vec2 g1 = vec2(gx.y, gy.y);
  vec2 g2 = vec2(gx.z, gy.z);
  vec3 w = 0.8 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2));
  w = max(w, 0.0);
  vec3 w2 = w * w;
  vec3 w4 = w2 * w2;
  vec3 gdotx = vec3(dot(g0, x0), dot(g1, x1), dot(g2, x2));
  float n = dot(w4, gdotx);
  vec3 w3 = w2 * w;
  vec3 dw = -8.0 * w3 * gdotx;
  vec2 dn0 = w4.x * g0 + dw.x * x0;
  vec2 dn1 = w4.y * g1 + dw.y * x1;
  vec2 dn2 = w4.z * g2 + dw.z * x2;
  gradient = 10.9 * (dn0 + dn1 + dn2);
  return 10.9 * n;
}

// Divergence-free 2D field: rotate the gradient of a scalar potential by 90°.
vec2 curl(vec2 x, float alpha) {
  vec2 g;
  psrdnoise(x, alpha, g);
  return vec2(g.y, -g.x);
}

// ---------------------------------------------------------------------------
// Velocity field. Three octaves: broad slow currents, medium folds / eddies
// that ride on the currents (domain-advected), and small surface disturbance.
// Each octave is divided by its frequency so the weights express displacement
// amplitude directly. Finer scales evolve faster, as in real turbulence.
// ---------------------------------------------------------------------------
vec2 flow(vec2 p, float t, float tideWeight) {
  float s = uNoiseScale;

  float f0 = 0.9 * s;
  float f1 = 2.1 * s;
  float f2 = 5.0 * s;
  float f3 = 10.5 * s;

  // |curl| of an octave is ~3/f, so the raw octaves are ~3.3, 1.4, 0.6, 0.29.
  vec2 c0 = curl(p * f0 + vec2(3.1, 7.7), t * 0.08) / f0;
  vec2 c1 = curl((p + 0.03 * c0) * f1 + vec2(-5.2, 1.3), t * 0.14 + 1.7) / f1;
  vec2 c2 = curl((p + 0.06 * c1) * f2 + vec2(9.4, -2.6), t * 0.24 + 4.0) / f2;
  vec2 c3 = curl((p + 0.10 * c2) * f3 + vec2(-1.7, 6.3), t * 0.40 + 8.0) / f3;

  // A slow tide: one unified direction that veers over minutes, so the eddies
  // are carried along instead of stirring in place. Its weight is supplied by
  // the caller so settled regions stay pinned and its speed varies regionally.
  float a = 0.55 + 0.35 * sin(t * 0.03 + 1.0);
  vec2 tide = 0.3 * tideWeight * vec2(cos(a), sin(a));

  // Weighted so the displacement spectrum peaks at the medium folds rather
  // than at broad currents (rubber-sheeting) or fine ripples (marbling).
  // Result is roughly unit RMS in frame-height units.
  return tide + c0 * 0.20 + c1 * 0.70 + c2 * 0.55 + c3 * 0.22;
}

// Trace a streamline through the (slowly evolving) field. `tau` is the
// advection "time"; `dir` = -1 traces backwards (forward advection of the
// image), +1 traces forwards (image arriving along the currents).
vec2 trace(vec2 p, vec2 v0, float tau, float dir, float t, float tideWeight) {
  float h = tau / float(FLOW_STEPS);
  vec2 q = p + dir * h * v0;
  for (int i = 1; i < FLOW_STEPS; i++) {
    q += dir * h * flow(q, t, tideWeight);
  }
  return q;
}

// ---------------------------------------------------------------------------
// Low-frequency threshold field for the spatially varying transition.
// ---------------------------------------------------------------------------
float maskField(vec2 p, float t) {
  float m = uMaskScale;
  vec2 g;
  float n0 = psrdnoise(p * 0.9 * m + vec2(11.0, -3.0), t * 0.045, g);
  float n1 = psrdnoise(p * 2.1 * m + vec2(-7.0, 5.0), t * 0.08 + 2.0, g);
  float n2 = psrdnoise(p * 5.3 * m + vec2(4.0, 9.0), t * 0.14 + 5.0, g);
  float n = n0 * 0.58 + n1 * 0.28 + n2 * 0.14;
  // simplex noise rarely exceeds ±0.75; stretch so the field spans ~[0,1]
  return clamp(n * 0.95 + 0.5, 0.0, 1.0);
}

// Map from frame-normalised flow space (x in ±aspect/2, y in ±0.5) to an
// image's texture coordinates, honouring contain/cover.
vec2 imageScale(vec2 imgSize) {
  float screenAspect = uResolution.x / uResolution.y;
  float imgAspect = imgSize.x / imgSize.y;
  float r = screenAspect / imgAspect;
  vec2 coverScale = (r > 1.0) ? vec2(1.0, 1.0 / r) : vec2(r, 1.0);
  vec2 containScale = (r > 1.0) ? vec2(r, 1.0) : vec2(1.0, 1.0 / r);
  return mix(containScale, coverScale, uFit);
}

vec2 toImageUv(vec2 p, vec2 scale, float screenAspect) {
  vec2 uv = p / vec2(screenAspect, 1.0) * scale + 0.5;
  return vec2(uv.x, 1.0 - uv.y);
}

float inside(vec2 uv) {
  vec2 b = step(vec2(0.0), uv) * step(uv, vec2(1.0));
  return b.x * b.y;
}

float luma(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float smootherstep(float e0, float e1, float x) {
  float t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Interleaved-gradient dither: hides 8-bit banding in the dark gradients
// of a blended frame. Faded out at the endpoints so the sources stay exact.
float dither(vec2 fragCoord) {
  return fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715)))) - 0.5;
}

void main() {
  float screenAspect = uResolution.x / uResolution.y;
  vec2 p = (vUv - 0.5) * vec2(screenAspect, 1.0);

  float progress = clamp(uProgress, 0.0, 1.0);
  float t = uTime * uFlowSpeed;

  // Advection "time" for each image. A keeps flowing away as progress grows,
  // B arrives from its displaced state and is exactly at rest at progress 1.
  // Feather of the transition front, in progress units, and the progress
  // remapped so the mask is exactly 0 at progress 0 and 1 at progress 1.
  float w = mix(0.04, 0.45, pow(uSoftness, 1.5));
  float pp = progress * (1.0 + 2.0 * w) - w;

  // Where is this region in its own transition? Churn is concentrated around
  // the moment the front passes; far from it both images sit almost still.
  float field0 = maskField(p, t);
  float thr0 = clamp(0.5 + (field0 - 0.5) * uTransitionWidth, 0.0, 1.0);
  float d = pp - thr0;
  float local = exp(-d * d / (2.0 * 0.25 * 0.25));
  float churn = mix(0.3, 1.0, local);

  // B arrives a little less displaced than A departs so its bright matter,
  // which surfaces first, is never rubber-sheeted.
  float ampBase = uWarpAmplitude * uIntensity * uMode;
  float amp = ampBase * churn;
  float tauA = amp * progress;
  float tauB = amp * 0.8 * (1.0 - progress);

  vec2 qA = p;
  vec2 qB = p;
  if (ampBase > 0.0) {
    float tideWeight = local * (0.6 + 0.8 * field0);
    vec2 v0 = flow(p, t, tideWeight); // the first step is shared by both traces
    qA = trace(p, v0, tauA, -1.0, t, tideWeight);
    qB = trace(p, v0, tauB, 1.0, t, tideWeight);
  }

  vec2 scaleA = imageScale(uSizeA);
  vec2 scaleB = imageScale(uSizeB);

  vec2 uvA = toImageUv(qA, scaleA, screenAspect);
  vec2 uvB = toImageUv(qB, scaleB, screenAspect);

  // The frame rectangle (contain mode) is defined by the *undisplaced* position
  // so the border stays perfectly still while the paint inside it moves.
  vec2 uvA0 = toImageUv(p, scaleA, screenAspect);
  vec2 uvB0 = toImageUv(p, scaleB, screenAspect);

  vec3 colA = texture2D(uTexA, uvA).rgb * inside(uvA0);
  vec3 colB = texture2D(uTexB, uvB).rgb * inside(uvB0);

  // --- spatially varying transition -----------------------------------------
  // Sample the threshold field at the advected coordinate so the front itself
  // is folded by the currents, then let the images nudge it: bright matter of
  // B surfaces first through dark regions of A, dark masses of B arrive last.
  vec2 pm = mix(p, qA, 0.75);
  float field = maskField(pm, t);
  float lumA = luma(colA);
  float lumB = luma(colB);
  float lumBias = (lumB - lumA) * 0.28;
  float threshold = 0.5 + (field - 0.5) * uTransitionWidth - lumBias * uTransitionWidth;
  threshold = clamp(threshold, 0.0, 1.0);

  float fluidMask = smootherstep(threshold - w, threshold + w, pp);

  // Inside the overlap band the darker pigment wins locally, so the two
  // images interleave like ink rather than ghosting through each other.
  float band = 4.0 * fluidMask * (1.0 - fluidMask);
  fluidMask = clamp(fluidMask + (lumA - lumB) * 0.6 * band, 0.0, 1.0);

  float m = mix(progress, fluidMask, uMode);

  vec3 col = mix(colA, colB, m);

  // Where the two pigments interpenetrate they shade each other very slightly;
  // zero at the endpoints by construction (m(1-m) = 0).
  col *= 1.0 - 0.10 * m * (1.0 - m) * uMode;

  gl_FragColor = vec4(col, 1.0);
  gl_FragColor = linearToOutputTexel(gl_FragColor);

  float edgeFade = smoothstep(0.0, 0.015, progress) * smoothstep(1.0, 0.985, progress);
  gl_FragColor.rgb += dither(gl_FragCoord.xy) * (0.75 / 255.0) * edgeFade;
}
