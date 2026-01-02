"""Image alignment using OpenCV ECC algorithm."""
import numpy as np
import cv2


def compute_transform(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    max_resolution: int = 2048,
    rough: bool = False,
) -> np.ndarray:
    """
    Compute affine transformation to align src to ref.

    Uses OpenCV's findTransformECC for subpixel accuracy.

    Args:
        ref_gray: Reference grayscale image
        src_gray: Source grayscale image to align
        max_resolution: Max resolution for alignment (downscales if larger)
        rough: If True, use fewer iterations (for initial alignment)

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

    # Initialize transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)

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
    except cv2.error:
        # If ECC fails, return identity
        pass

    # Scale translation back to original resolution
    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale

    return warp_matrix


def align_image(
    ref_gray: np.ndarray,
    ref_color: np.ndarray,
    src_gray: np.ndarray,
    src_color: np.ndarray,
    transform: np.ndarray | None = None,
) -> np.ndarray:
    """
    Align source image to reference.

    Args:
        ref_gray: Reference grayscale image
        ref_color: Reference color image (for size)
        src_gray: Source grayscale image
        src_color: Source color image to warp
        transform: Optional pre-computed transform matrix

    Returns:
        Aligned color image
    """
    if transform is None:
        # Rough alignment first
        transform = compute_transform(ref_gray, src_gray, max_resolution=256, rough=True)
        # Then fine alignment
        transform = compute_transform(ref_gray, src_gray, max_resolution=2048, rough=False)

    # Apply transform
    h, w = ref_color.shape[:2]
    aligned = cv2.warpAffine(
        src_color,
        transform,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    return aligned
