import * as THREE from 'three';

const ACCEPTED = ['image/png', 'image/jpeg', 'image/webp'];

export function isSupportedImage(file) {
  return ACCEPTED.includes(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name || '');
}

/**
 * Decode a Blob into an ImageBitmap at the largest size the GPU accepts.
 * Downscaling happens only when the source exceeds MAX_TEXTURE_SIZE, and only
 * once, at upload. Colour-space conversion and alpha premultiplication are
 * disabled so the raw sRGB bytes reach the GPU untouched.
 */
export async function decodeImage(blob, maxTextureSize) {
  const base = { colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'from-image' };
  let bitmap = await createImageBitmap(blob, base);
  const longest = Math.max(bitmap.width, bitmap.height);
  let downscaled = false;
  if (longest > maxTextureSize) {
    const k = maxTextureSize / longest;
    const resized = await createImageBitmap(bitmap, {
      ...base,
      resizeWidth: Math.max(1, Math.round(bitmap.width * k)),
      resizeHeight: Math.max(1, Math.round(bitmap.height * k)),
      resizeQuality: 'high',
    });
    bitmap.close();
    bitmap = resized;
    downscaled = true;
  }
  return { bitmap, downscaled };
}

/**
 * Wrap an ImageBitmap in a texture configured for faithful, stable sampling:
 * sRGB decode in hardware, trilinear mipmaps + full anisotropy for minified
 * detail, mirrored wrap so displaced samples beyond the border stay plausible.
 */
export function createTexture(bitmap, renderer) {
  const tex = new THREE.Texture(bitmap);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = false; // ImageBitmap ignores UNPACK_FLIP_Y; the shader flips v instead
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.MirroredRepeatWrapping;
  tex.wrapT = THREE.MirroredRepeatWrapping;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  tex.needsUpdate = true;
  return tex;
}

export async function loadTextureFromBlob(blob, renderer) {
  const { bitmap, downscaled } = await decodeImage(blob, renderer.capabilities.maxTextureSize);
  const texture = createTexture(bitmap, renderer);
  return { texture, width: bitmap.width, height: bitmap.height, downscaled };
}

export async function loadTextureFromUrl(url, renderer) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return loadTextureFromBlob(await res.blob(), renderer);
}
