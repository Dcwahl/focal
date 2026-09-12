"""Image alignment using OpenCV ECC algorithm."""
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """
    Invert a 2x3 affine transformation matrix.

    Args:
        transform: 2x3 affine matrix (src -> ref)

    Returns:
        2x3 inverse affine matrix (ref -> src)
    """
    return cv2.invertAffineTransform(transform)


def sample_aligned_region(
    source: np.ndarray,
    transform: np.ndarray | None,
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a result-space rectangle and mark pixels covered by the source.

    bounds is (y_start, y_end, x_start, x_end). Only the requested rectangle is
    allocated, so brush painting does not warp an entire high-resolution image.
    """
    y0, y1, x0, x1 = bounds
    if transform is None:
        transform = np.eye(2, 3, dtype=np.float32)
    local_transform = transform.copy()
    local_transform[:, 2] -= (x0, y0)
    region = cv2.warpAffine(source, local_transform, (x1 - x0, y1 - y0),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    inverse = invert_transform(transform)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    sx = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
    sy = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
    valid = (sx >= 0) & (sx <= source.shape[1] - 1) & (sy >= 0) & (sy <= source.shape[0] - 1)
    return region, valid


def compute_transform(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    max_resolution: int = 2048,
    rough: bool = False,
    initial_transform: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute affine transformation to align src to ref.

    Uses OpenCV's findTransformECC for subpixel accuracy.

    Args:
        ref_gray: Reference grayscale image
        src_gray: Source grayscale image to align
        max_resolution: Max resolution for alignment (downscales if larger)
        rough: If True, use fewer iterations (for initial alignment)
        initial_transform: Source-to-reference affine estimate at full resolution.

    Returns:
        2x3 affine transformation matrix
    """
    # Resize if needed
    resolution = max(ref_gray.shape)
    scale = 1.0
    if resolution > max_resolution:
        scale = max_resolution / resolution
        ref_scaled = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        src_scaled = cv2.resize(src_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        ref_scaled = ref_gray
        src_scaled = src_gray

    # Convert the full-resolution estimate into the actual resized coordinates,
    # including rounding of the resized width and height.
    initial = np.eye(3, dtype=np.float64)
    if initial_transform is not None:
        initial[:2] = initial_transform
    resize = np.diag([ref_scaled.shape[1] / ref_gray.shape[1],
                      ref_scaled.shape[0] / ref_gray.shape[0], 1.0])
    warp_matrix = (resize @ initial @ np.linalg.inv(resize))[:2].astype(np.float32)
    if rough and initial_transform is None:
        # Translation initialization gives ECC a useful starting point when
        # motion is too large for its local optimization from identity.
        src_float = src_scaled.astype(np.float32)
        ref_float = ref_scaled.astype(np.float32)
        window = cv2.createHanningWindow((ref_scaled.shape[1], ref_scaled.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(src_float, ref_float, window)
        if response > 0.1 and np.isfinite(shift).all():
            warp_matrix[:, 2] = shift
    fallback = warp_matrix.copy()

    # Set termination criteria
    if rough:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01)
        gauss_filt_size = 1
    else:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        gauss_filt_size = 3

    try:
        _, warp_matrix = cv2.findTransformECC(
            src_scaled.astype(np.float32),
            ref_scaled.astype(np.float32),
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            gauss_filt_size
        )
        if not np.isfinite(warp_matrix).all() or np.linalg.det(warp_matrix[:, :2]) <= 0:
            logger.warning("ECC returned an invalid affine transform; keeping the initial estimate")
            warp_matrix = fallback
    except cv2.error as error:
        # ECC mutates its input before raising: never return that partial solve.
        logger.warning("ECC alignment failed; keeping the initial estimate: %s", error)
        warp_matrix = fallback

    # Convert the refined affine transform back to full-resolution coordinates.
    full = np.eye(3, dtype=np.float64)
    full[:2] = warp_matrix
    return (np.linalg.inv(resize) @ full @ resize)[:2].astype(np.float32)


def align_image(
    ref_gray: np.ndarray,
    ref_color: np.ndarray,
    src_gray: np.ndarray,
    src_color: np.ndarray,
    transform: np.ndarray | None = None,
    return_transform: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Align source image to reference.

    Args:
        ref_gray: Reference grayscale image
        ref_color: Reference color image (for size)
        src_gray: Source grayscale image
        src_color: Source color image to warp
        transform: Optional pre-computed transform matrix
        return_transform: If True, return (aligned_image, transform) tuple

    Returns:
        Aligned color image, or (aligned_image, transform) if return_transform=True
    """
    if transform is None:
        # Rough alignment first
        transform = compute_transform(ref_gray, src_gray, max_resolution=256, rough=True)
        # Then fine alignment
        transform = compute_transform(ref_gray, src_gray, max_resolution=2048,
                                      rough=False, initial_transform=transform)

    # Apply transform
    h, w = ref_color.shape[:2]
    aligned = cv2.warpAffine(
        src_color,
        transform,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    if return_transform:
        return aligned, transform
    return aligned
