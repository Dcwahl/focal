"""Color reassignment from merged grayscale back to original colors."""
import numpy as np
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
