"""Color reassignment from merged grayscale back to original colors."""
import numpy as np
from numba import njit
from dataclasses import dataclass


@dataclass
class ColorEntry:
    """Stores a grayscale value and its corresponding color."""
    gray: int
    color: np.ndarray  # BGR color


def build_color_map(
    grayscale_imgs: list[np.ndarray],
    color_imgs: list[np.ndarray],
) -> list[list[ColorEntry]]:
    """
    Build per-pixel mapping from grayscale values to colors.

    For each pixel position, stores all unique grayscale values
    seen across the stack with their corresponding colors.

    Args:
        grayscale_imgs: List of grayscale images
        color_imgs: List of corresponding color images

    Returns:
        2D list of ColorEntry lists, one per pixel
    """
    h, w = grayscale_imgs[0].shape
    color_map: list[list[ColorEntry]] = []

    for y in range(h):
        for x in range(w):
            entries: list[ColorEntry] = []
            seen_grays: set[int] = set()

            for gray_img, color_img in zip(grayscale_imgs, color_imgs):
                gray_val = int(gray_img[y, x])
                if gray_val not in seen_grays:
                    seen_grays.add(gray_val)
                    entries.append(ColorEntry(
                        gray=gray_val,
                        color=color_img[y, x].copy()
                    ))

            color_map.append(entries)

    return color_map


def reassign_colors(
    merged_gray: np.ndarray,
    color_map: list[list[ColorEntry]],
) -> np.ndarray:
    """
    Reassign colors to merged grayscale image.

    For each pixel, finds the color entry with closest grayscale value.

    Args:
        merged_gray: Merged grayscale image
        color_map: Per-pixel color mapping from build_color_map

    Returns:
        Color image with reassigned pixels
    """
    h, w = merged_gray.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    idx = 0
    for y in range(h):
        for x in range(w):
            entries = color_map[idx]
            idx += 1

            if not entries:
                continue

            target_gray = int(merged_gray[y, x])

            # Find closest match
            best_entry = entries[0]
            best_error = abs(best_entry.gray - target_gray)

            for entry in entries[1:]:
                error = abs(entry.gray - target_gray)
                if error < best_error:
                    best_error = error
                    best_entry = entry
                if error == 0:
                    break

            result[y, x] = best_entry.color

    return result


# =============================================================================
# Fast Numba-accelerated versions
# =============================================================================

@njit(cache=True)
def _build_color_map_fast(gray_stack: np.ndarray, color_stack: np.ndarray) -> tuple:
    """
    Build color map as arrays for fast lookup.

    Returns:
        (gray_values, colors) - arrays of shape (H, W, N) and (H, W, N, 3)
        where N is number of source images
    """
    n_imgs, h, w = gray_stack.shape

    # Store all gray values and colors per pixel
    # Shape: (H, W, N) for grays, (H, W, N, 3) for colors
    gray_values = np.zeros((h, w, n_imgs), dtype=np.uint8)
    colors = np.zeros((h, w, n_imgs, 3), dtype=np.uint8)

    for i in range(n_imgs):
        for y in range(h):
            for x in range(w):
                gray_values[y, x, i] = gray_stack[i, y, x]
                colors[y, x, i, 0] = color_stack[i, y, x, 0]
                colors[y, x, i, 1] = color_stack[i, y, x, 1]
                colors[y, x, i, 2] = color_stack[i, y, x, 2]

    return gray_values, colors


@njit(cache=True)
def _reassign_colors_fast(
    merged_gray: np.ndarray,
    gray_values: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """
    Reassign colors using precomputed arrays (JIT-compiled).

    Args:
        merged_gray: Merged grayscale image (H, W)
        gray_values: Gray values per pixel per image (H, W, N)
        colors: Colors per pixel per image (H, W, N, 3)

    Returns:
        Color image (H, W, 3)
    """
    h, w = merged_gray.shape
    n_imgs = gray_values.shape[2]
    result = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            target = merged_gray[y, x]

            # Find closest match
            # Note: Must use np.int64 to avoid unsigned overflow in Numba
            # (uint8 subtraction wraps around instead of going negative)
            best_idx = 0
            best_error = abs(np.int64(gray_values[y, x, 0]) - np.int64(target))

            for i in range(1, n_imgs):
                error = abs(np.int64(gray_values[y, x, i]) - np.int64(target))
                if error < best_error:
                    best_error = error
                    best_idx = i
                if error == 0:
                    break

            result[y, x, 0] = colors[y, x, best_idx, 0]
            result[y, x, 1] = colors[y, x, best_idx, 1]
            result[y, x, 2] = colors[y, x, best_idx, 2]

    return result


def build_color_map_fast(
    grayscale_imgs: list[np.ndarray],
    color_imgs: list[np.ndarray],
) -> tuple:
    """
    Build color map as stacked arrays for fast Numba lookup.

    Args:
        grayscale_imgs: List of grayscale images
        color_imgs: List of corresponding color images

    Returns:
        Tuple of (gray_stack, color_stack) arrays
    """
    gray_stack = np.stack(grayscale_imgs, axis=0).astype(np.uint8)
    color_stack = np.stack(color_imgs, axis=0).astype(np.uint8)
    return _build_color_map_fast(gray_stack, color_stack)


def reassign_colors_fast(
    merged_gray: np.ndarray,
    gray_values: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """
    Reassign colors using fast Numba implementation.

    Args:
        merged_gray: Merged grayscale image
        gray_values: From build_color_map_fast
        colors: From build_color_map_fast

    Returns:
        Color image with reassigned pixels
    """
    return _reassign_colors_fast(
        merged_gray.astype(np.uint8),
        gray_values,
        colors,
    )
