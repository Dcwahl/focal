"""
Complex Daubechies wavelet transform for focus stacking.

Ported from py-focus-stack, which ported from focus-stack C++ implementation.
Reference: "Image Processing with Complex Daubechies Wavelets", J.M. Lina, 1997
"""
import numpy as np
from numba import njit

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
    """
    Determine number of decomposition levels for given image size.

    Aims for ~8 pixel image at lowest level.
    """
    dimension = max(shape)
    levels = MIN_LEVELS
    while (dimension >> levels) > 8 and levels < MAX_LEVELS:
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


@njit(cache=True)
def _decompose_1d_jit(src: np.ndarray, vertical: bool, lopass: np.ndarray, hipass: np.ndarray) -> np.ndarray:
    """1D wavelet decomposition along one axis (JIT-compiled)."""
    filter_len = 6

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

            for j in range(filter_len):
                pos = y + j - filter_len // 2
                if pos < 0:
                    pos = length + pos
                if pos >= length:
                    pos = pos - length

                if vertical:
                    val_re = src[pos, x, 0]
                    val_im = src[pos, x, 1]
                else:
                    val_re = src[x, pos, 0]
                    val_im = src[x, pos, 1]

                lo_re = lopass[j, 0]
                lo_im = lopass[j, 1]
                hi_re = hipass[j, 0]
                hi_im = hipass[j, 1]

                re_lo += val_re * lo_re - val_im * lo_im
                im_lo += val_im * lo_re + val_re * lo_im
                re_hi += val_re * hi_re - val_im * hi_im
                im_hi += val_im * hi_re + val_re * hi_im

            if vertical:
                dest[y // 2, x, 0] = re_lo
                dest[y // 2, x, 1] = im_lo
                dest[y // 2 + halflen, x, 0] = re_hi
                dest[y // 2 + halflen, x, 1] = im_hi
            else:
                dest[x, y // 2, 0] = re_lo
                dest[x, y // 2, 1] = im_lo
                dest[x, y // 2 + halflen, 0] = re_hi
                dest[x, y // 2 + halflen, 1] = im_hi

    return dest


def _decompose_1d(src: np.ndarray, vertical: bool) -> np.ndarray:
    """1D wavelet decomposition along one axis."""
    return _decompose_1d_jit(src, vertical, LOPASS, HIPASS)


@njit(cache=True)
def _compose_1d_jit(src: np.ndarray, vertical: bool, lopass: np.ndarray, hipass: np.ndarray) -> np.ndarray:
    """1D wavelet composition (inverse of decompose_1d) (JIT-compiled)."""
    filter_len = 6

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

            for j in range((y + filter_len // 2) % 2, filter_len, 2):
                pos = (y - j + filter_len // 2) // 2
                if pos < 0:
                    pos = halflen + pos
                if pos >= halflen:
                    pos = pos - halflen

                if vertical:
                    val_lo_re = src[pos, x, 0]
                    val_lo_im = src[pos, x, 1]
                    val_hi_re = src[pos + halflen, x, 0]
                    val_hi_im = src[pos + halflen, x, 1]
                else:
                    val_lo_re = src[x, pos, 0]
                    val_lo_im = src[x, pos, 1]
                    val_hi_re = src[x, pos + halflen, 0]
                    val_hi_im = src[x, pos + halflen, 1]

                lo_re = lopass[j, 0]
                lo_im = lopass[j, 1]
                hi_re = hipass[j, 0]
                hi_im = hipass[j, 1]

                re += val_lo_re * lo_re + val_hi_re * hi_re
                re += val_lo_im * lo_im + val_hi_im * hi_im
                im += val_lo_im * lo_re + val_hi_im * hi_re
                im -= val_lo_re * lo_im + val_hi_re * hi_im

            if vertical:
                dest[y, x, 0] = re
                dest[y, x, 1] = im
            else:
                dest[x, y, 0] = re
                dest[x, y, 1] = im

    return dest


def _compose_1d(src: np.ndarray, vertical: bool) -> np.ndarray:
    """1D wavelet composition (inverse of decompose_1d)."""
    return _compose_1d_jit(src, vertical, LOPASS, HIPASS)


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
    magnitude_threshold: float = 0.0,
) -> np.ndarray:
    """
    Merge multiple wavelet images by selecting highest magnitude coefficients.

    Args:
        wavelets: List of wavelet coefficient arrays (H, W, 2)
        consistency: Denoising level 0-2
        magnitude_threshold: Minimum squared magnitude to allow frame switching.
            Below this threshold, reference frame (first) is used. Prevents
            noisy selection in low-contrast regions.

    Returns:
        Merged wavelet coefficients
    """
    h, w = wavelets[0].shape[:2]

    result = wavelets[0].copy()
    depth_map = np.zeros((h, w), dtype=np.uint16)
    max_absval = get_sq_absval(result)

    # Select maximum magnitude wavelet at each position
    for i, wavelet in enumerate(wavelets[1:], 1):
        absval = get_sq_absval(wavelet)
        # Only switch frames if magnitude is above threshold AND greater than current max
        mask = (absval > max_absval) & (absval > magnitude_threshold)

        result[mask] = wavelet[mask]
        depth_map[mask] = i
        max_absval = np.maximum(max_absval, absval)

    if consistency >= 1:
        _denoise_subbands(result, depth_map, wavelets)

    if consistency >= 2:
        _denoise_neighbours(result, depth_map, wavelets)

    return result


def _denoise_subbands(
    result: np.ndarray,
    depth_map: np.ndarray,
    wavelets: list[np.ndarray],
) -> None:
    """
    Two-out-of-three voting across H/V/D subbands at each level.
    Modifies result and depth_map in place.
    """
    levels = levels_for_size(result.shape[:2])

    for level in range(levels):
        w = result.shape[1] >> level
        h = result.shape[0] >> level
        w2 = w // 2
        h2 = h // 2

        # Three subbands: horizontal, diagonal, vertical
        for y in range(h2):
            for x in range(w2):
                v1 = depth_map[y, w2 + x]           # Horizontal
                v2 = depth_map[h2 + y, w2 + x]      # Diagonal
                v3 = depth_map[h2 + y, x]           # Vertical

                if v1 == v2 == v3:
                    continue
                elif v2 == v3 and v1 != v2:
                    depth_map[y, w2 + x] = v2
                    result[y, w2 + x] = wavelets[v2][y, w2 + x]
                elif v1 == v3 and v2 != v1:
                    depth_map[h2 + y, w2 + x] = v1
                    result[h2 + y, w2 + x] = wavelets[v1][h2 + y, w2 + x]
                elif v1 == v2 and v3 != v1:
                    depth_map[h2 + y, x] = v1
                    result[h2 + y, x] = wavelets[v1][h2 + y, x]


def _denoise_neighbours(
    result: np.ndarray,
    depth_map: np.ndarray,
    wavelets: list[np.ndarray],
) -> None:
    """
    Remove outlier pixels where all 4 neighbors agree on different value.
    Modifies result and depth_map in place.
    """
    h, w = depth_map.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            left = depth_map[y, x - 1]
            right = depth_map[y, x + 1]
            top = depth_map[y - 1, x]
            bottom = depth_map[y + 1, x]
            center = depth_map[y, x]

            # Check if center is outlier (all neighbors above or below)
            if ((center > top and center > bottom and
                 center > left and center > right) or
                (center < top and center < bottom and
                 center < left and center < right)):

                # Replace with average of neighbors
                avg = (int(top) + int(bottom) + int(left) + int(right) + 2) // 4
                avg = min(avg, len(wavelets) - 1)
                depth_map[y, x] = avg
                result[y, x] = wavelets[avg][y, x]
