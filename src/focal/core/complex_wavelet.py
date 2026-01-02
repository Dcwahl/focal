"""
Complex Daubechies wavelet transform for focus stacking.

Ported from py-focus-stack, which ported from focus-stack C++ implementation.
Reference: "Image Processing with Complex Daubechies Wavelets", J.M. Lina, 1997
"""
import numpy as np

# Complex Daubechies wavelet filter coefficients
# Each pair is (real, imaginary)
LOPASS = np.array([
    [-0.0662912607, -0.0855816496],
    [ 0.1104854346, -0.0855816496],
    [ 0.6629126074,  0.1711632992],
    [ 0.6629126074,  0.1711632992],
    [ 0.1104854346, -0.0855816496],
    [-0.0662912607, -0.0855816496],
], dtype=np.float32)

HIPASS = np.array([
    [-0.0662912607,  0.0855816496],
    [-0.1104854346, -0.0855816496],
    [ 0.6629126074, -0.1711632992],
    [-0.6629126074,  0.1711632992],
    [ 0.1104854346,  0.0855816496],
    [ 0.0662912607, -0.0855816496],
], dtype=np.float32)

FILTER_LEN = 6
MIN_LEVELS = 5
MAX_LEVELS = 10


def levels_for_size(shape: tuple[int, int]) -> int:
    """Determine number of decomposition levels for given image size."""
    min_dimension = min(shape)
    max_dimension = max(shape)

    # Cap levels to ensure smallest region is at least 4x4
    # (needed for filter boundary handling in compose)
    max_safe_levels = int(np.log2(min_dimension / 4)) if min_dimension >= 4 else 1
    max_safe_levels = max(1, max_safe_levels)

    # Start at MIN_LEVELS but don't exceed safe maximum
    levels = min(MIN_LEVELS, max_safe_levels)

    # Increase if we have room and regions would still be > 8
    while (max_dimension >> levels) > 8 and levels < MAX_LEVELS and levels < max_safe_levels:
        levels += 1

    return levels


def expand_to_valid_size(shape: tuple[int, int], levels: int) -> tuple[int, int]:
    """Return shape expanded to be divisible by 2^levels."""
    divider = 1 << levels
    h, w = shape
    if h % divider != 0:
        h += divider - (h % divider)
    if w % divider != 0:
        w += divider - (w % divider)
    return (h, w)


def _decompose_1d(src: np.ndarray, vertical: bool) -> np.ndarray:
    """1D wavelet decomposition along one axis."""
    if vertical:
        length = src.shape[0]
        count = src.shape[1]
    else:
        length = src.shape[1]
        count = src.shape[0]

    halflen = length // 2
    dest = np.zeros_like(src)

    for x in range(count):
        for y in range(0, length, 2):
            re_lo, im_lo = 0.0, 0.0
            re_hi, im_hi = 0.0, 0.0

            for j in range(FILTER_LEN):
                pos = y + j - FILTER_LEN // 2
                if pos < 0:
                    pos = length + pos
                if pos >= length:
                    pos = pos - length

                if vertical:
                    val = src[pos, x]
                else:
                    val = src[x, pos]

                lo = LOPASS[j]
                hi = HIPASS[j]

                re_lo += val[0] * lo[0] - val[1] * lo[1]
                im_lo += val[1] * lo[0] + val[0] * lo[1]
                re_hi += val[0] * hi[0] - val[1] * hi[1]
                im_hi += val[1] * hi[0] + val[0] * hi[1]

            if vertical:
                dest[y // 2, x] = [re_lo, im_lo]
                dest[y // 2 + halflen, x] = [re_hi, im_hi]
            else:
                dest[x, y // 2] = [re_lo, im_lo]
                dest[x, y // 2 + halflen] = [re_hi, im_hi]

    return dest


def _compose_1d(src: np.ndarray, vertical: bool) -> np.ndarray:
    """1D wavelet composition (inverse of decompose_1d)."""
    if vertical:
        length = src.shape[0]
        count = src.shape[1]
    else:
        length = src.shape[1]
        count = src.shape[0]

    halflen = length // 2
    dest = np.zeros_like(src)

    for x in range(count):
        for y in range(length):
            re, im = 0.0, 0.0

            for j in range((y + FILTER_LEN // 2) % 2, FILTER_LEN, 2):
                pos = (y - j + FILTER_LEN // 2) // 2
                if pos < 0:
                    pos = halflen + pos
                if pos >= halflen:
                    pos = pos - halflen

                if vertical:
                    val_lo = src[pos, x]
                    val_hi = src[pos + halflen, x]
                else:
                    val_lo = src[x, pos]
                    val_hi = src[x, pos + halflen]

                lo = LOPASS[j]
                hi = HIPASS[j]

                re += val_lo[0] * lo[0] + val_hi[0] * hi[0]
                re += val_lo[1] * lo[1] + val_hi[1] * hi[1]
                im += val_lo[1] * lo[0] + val_hi[1] * hi[0]
                im -= val_lo[0] * lo[1] + val_hi[0] * hi[1]

            if vertical:
                dest[y, x] = [re, im]
            else:
                dest[x, y] = [re, im]

    return dest


def _decompose_level(input_arr: np.ndarray) -> np.ndarray:
    """Single level 2D decomposition."""
    tmp = _decompose_1d(input_arr, vertical=True)
    return _decompose_1d(tmp, vertical=False)


def _compose_level(input_arr: np.ndarray) -> np.ndarray:
    """Single level 2D composition."""
    tmp = _compose_1d(input_arr, vertical=True)
    return _compose_1d(tmp, vertical=False)


def decompose(image: np.ndarray, levels: int) -> np.ndarray:
    """
    Multi-level wavelet decomposition.

    Args:
        image: Grayscale image as float32 array (H, W)
        levels: Number of decomposition levels

    Returns:
        Complex wavelet coefficients as (H, W, 2) array
    """
    h, w = image.shape

    # Convert to complex (real, imag=0)
    output = np.zeros((h, w, 2), dtype=np.float32)
    output[:, :, 0] = image

    for i in range(levels):
        size = h >> i, w >> i
        region = output[:size[0], :size[1]].copy()
        output[:size[0], :size[1]] = _decompose_level(region)

    return output


def compose(wavelet: np.ndarray, levels: int) -> np.ndarray:
    """
    Multi-level wavelet composition (inverse transform).

    Args:
        wavelet: Complex wavelet coefficients as (H, W, 2) array
        levels: Number of decomposition levels

    Returns:
        Reconstructed grayscale image as float32 array (H, W)
    """
    h, w = wavelet.shape[:2]
    output = wavelet.copy()

    for i in range(levels - 1, -1, -1):
        size = h >> i, w >> i
        region = output[:size[0], :size[1]].copy()
        output[:size[0], :size[1]] = _compose_level(region)

    return output[:, :, 0]


def get_sq_absval(complex_arr: np.ndarray) -> np.ndarray:
    """Compute squared absolute value of complex array."""
    return complex_arr[:, :, 0] ** 2 + complex_arr[:, :, 1] ** 2


def merge_wavelets(
    wavelets: list[np.ndarray],
    consistency: int = 2,
) -> np.ndarray:
    """
    Merge multiple wavelet images by selecting highest magnitude coefficients.

    Args:
        wavelets: List of wavelet coefficient arrays (H, W, 2)
        consistency: Denoising level 0-2

    Returns:
        Merged wavelet coefficients
    """
    result = wavelets[0].copy()
    max_absval = get_sq_absval(result)

    for wavelet in wavelets[1:]:
        absval = get_sq_absval(wavelet)
        mask = absval > max_absval
        result[mask] = wavelet[mask]
        max_absval = np.maximum(max_absval, absval)

    if consistency >= 1:
        _denoise_subbands(result, wavelets)

    if consistency >= 2:
        _denoise_neighbours(result, wavelets)

    return result


def _denoise_subbands(result: np.ndarray, wavelets: list[np.ndarray]) -> None:
    """Two-out-of-three voting across H/V/D subbands at each level."""
    levels = levels_for_size(result.shape[:2])

    for level in range(levels):
        w = result.shape[1] >> level
        h = result.shape[0] >> level
        w2 = w // 2
        h2 = h // 2

        for y in range(h2):
            for x in range(w2):
                # Get which source image each subband came from
                absvals_h = [get_sq_absval(wv)[y, w2 + x] for wv in wavelets]
                absvals_d = [get_sq_absval(wv)[h2 + y, w2 + x] for wv in wavelets]
                absvals_v = [get_sq_absval(wv)[h2 + y, x] for wv in wavelets]

                src_h = np.argmax(absvals_h)
                src_d = np.argmax(absvals_d)
                src_v = np.argmax(absvals_v)

                if src_h == src_d == src_v:
                    continue
                elif src_d == src_v and src_h != src_d:
                    result[y, w2 + x] = wavelets[src_d][y, w2 + x]
                elif src_h == src_v and src_d != src_h:
                    result[h2 + y, w2 + x] = wavelets[src_h][h2 + y, w2 + x]
                elif src_h == src_d and src_v != src_h:
                    result[h2 + y, x] = wavelets[src_h][h2 + y, x]


def _denoise_neighbours(result: np.ndarray, wavelets: list[np.ndarray]) -> None:
    """Remove outlier pixels where all 4 neighbors agree on different source."""
    h, w = result.shape[:2]

    # Compute source indices for each pixel
    all_absvals = np.stack([get_sq_absval(wv) for wv in wavelets], axis=0)
    sources = np.argmax(all_absvals, axis=0)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = sources[y, x]
            left = sources[y, x - 1]
            right = sources[y, x + 1]
            top = sources[y - 1, x]
            bottom = sources[y + 1, x]

            if ((center > top and center > bottom and
                 center > left and center > right) or
                (center < top and center < bottom and
                 center < left and center < right)):
                avg = (int(top) + int(bottom) + int(left) + int(right) + 2) // 4
                avg = min(avg, len(wavelets) - 1)
                result[y, x] = wavelets[avg][y, x]
