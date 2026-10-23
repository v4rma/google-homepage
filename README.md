# Posterity — Fluid Morph

A real-time, browser-based experiment that morphs between two high-resolution
artworks with custom GLSL shaders. Instead of crossfading, the frame is treated
as a single slow, incompressible fluid: the pigment of image A is carried away
along evolving currents while the pigment of image B arrives along the same
currents and settles into place.

Built with [Three.js](https://threejs.org) and [Vite](https://vitejs.dev). No
other runtime dependencies, no backend.

## Run it locally

```bash
npm install
npm run dev
```

Open the printed URL (default <http://localhost:5173>). The two sample artworks
load automatically; drag the **Progress** slider or press **Play**.

`npm run build` produces a static site in `dist/` (`npm run preview` serves it).

## Controls

| Control | What it does |
| --- | --- |
| Progress | Scrub between image A (0 %) and image B (100 %). |
| Play / Pause | Ping-pong 0 → 1 → 0 automatically (space bar also toggles). |
| Duration | 5–60 s per transition. |
| Fluid intensity | How strongly the pigment is displaced. |
| Transition softness | How gradually B emerges through A (feather of the transition front). |
| Mode | Compare the fluid morph against a plain crossfade. |
| Fit | Cover (fill the screen, crop) or Contain (letterbox). |
| A / B | Load your own PNG, JPG or WebP. Drag & drop also works: left half → A, right half → B, two files → A + B. |
| hide / `H` | Hide the panel for fullscreen viewing; move the mouse to get it back. |
| fullscreen | Toggle browser fullscreen. |

**Advanced** exposes the underlying parameters: noise scale, warp amplitude,
flow speed, transition width (how staggered different regions are in time),
mask scale, and a render-scale switch for weaker GPUs.

URL parameters for reproducible frames: `?p=0.5` (progress), `?t=12` (freeze the
clock at 12 s), `?controls=0`, `?mode=crossfade`, `?fit=contain`, `?play=1`,
`?a=<url>&b=<url>` (alternative images served from `public/`).

## How the morph works

Everything happens in one fragment shader (`src/shaders/morph.frag`) that
samples the two *source* textures once each per frame. There is no feedback
buffer, so detail is never resampled or blurred over time.

1. **Flow field.** Three octaves of gradient noise with analytic derivatives
   ([psrdnoise](https://github.com/stegu/psrdnoise)) are turned into
   divergence-free *curl* fields: broad slow currents, medium folds and eddies
   that are domain-advected by the currents, and small surface disturbance.
   Finer octaves evolve faster. Rotating the noise gradients over time
   ("flow noise") makes the field evolve smoothly without re-hashing.
2. **Bidirectional advection.** For every pixel a short streamline is traced
   through the field. Image A is sampled by tracing *backwards* for an
   advection time proportional to `progress` (its pigment flows away); image B
   is sampled by tracing *forwards* for a time proportional to `1 − progress`
   (its pigment arrives along the same currents and is exactly at rest at
   `progress = 1`). Both displacements are therefore zero at the endpoint
   where that image is shown, so A and B are reproduced pixel-exactly.
3. **Spatially varying transition.** A low-frequency threshold field decides
   when each region switches from A to B. It is sampled at the advected
   coordinate so the front itself is folded by the currents, and it is nudged
   by the images: bright matter in B surfaces first through dark regions of A
   and dark masses of B arrive last. The front is feathered with a smooth
   step whose width is the *softness* control, constructed so the mask is
   exactly 0 at `progress = 0` and 1 at `progress = 1` everywhere.
4. **Detail preservation.** Textures are uploaded as sRGB with trilinear
   mipmaps and maximum anisotropic filtering, blended in linear light and
   re-encoded on output; the renderer runs at device pixel ratio; images
   larger than the GPU's `MAX_TEXTURE_SIZE` are downscaled once at upload
   (you get a notice). A ±0.75 LSB dither hides 8-bit banding in blended
   frames and is faded out at the endpoints.

When progress sits anywhere strictly between 0 and 1 the field keeps evolving,
so the painting breathes gently even while the slider is still.

### What this is — and is not

This is **noise-driven texture warping plus spatially varying blending**. A
shader has no idea what the shapes in two unrelated images are, so this is not
semantic morphing: a mountain in A does not "become" a wave in B. It is the
most convincing fluid reorganisation available without correspondence data.
A second method based on optical-flow / feature correspondence preprocessing
is a natural next experiment and is deliberately kept out of this prototype.

## Sample artworks

`public/samples/imageA.webp` and `imageB.webp` are the two reference artworks
at full resolution (2688 × 1792). They were converted from the original ~7 MB
PNGs to WebP at quality 95 (about 1.4 MB each) to keep the repository light;
the difference is visually imperceptible. Any PNG/JPG/WebP can be loaded in
their place through the panel or by drag and drop.

## Project layout

```
index.html              markup for the canvas and the control panel
src/main.js             UI wiring, deterministic clock, playback, drag & drop
src/morph.js            renderer, fullscreen triangle, uniforms
src/textures.js         decoding, texture-limit handling, texture setup
src/shaders/morph.frag  the fluid morph
src/shaders/morph.vert
src/style.css
public/samples/         default artworks
```
