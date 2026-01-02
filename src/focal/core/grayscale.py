"""PCA-based grayscale conversion for maximum information preservation."""
import numpy as np
import cv2


def compute_pca_weights(
    image: np.ndarray,
    samples: int = 64
) -> np.ndarray:
    """
    Compute optimal grayscale weights using PCA.

    Samples the image and finds the color channel combination
    with maximum variance.

    Args:
        image: BGR image (H, W, 3)
        samples: Number of samples per dimension

    Returns:
        Weights array (3,) for B, G, R channels
    """
    h, w = image.shape[:2]

    # Sample pixels from image
    ys = np.linspace(0, h - 1, samples, dtype=int)
    xs = np.linspace(0, w - 1, samples, dtype=int)

    sample_data = []
    for y in ys:
        for x in xs:
            sample_data.append(image[y, x].astype(np.float32))

    samples_mat = np.array(sample_data, dtype=np.float32)

    # Compute PCA
    mean, eigenvectors = cv2.PCACompute(samples_mat, mean=None, maxComponents=2)

    # Get first principal component direction
    weights = eigenvectors[0]

    # Normalize so weights sum to 1
    weights = weights / weights.sum()

    return weights


def to_grayscale(
    image: np.ndarray,
    weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: Input image (H, W) or (H, W, 3) in BGR format
        weights: Optional weights for B, G, R channels.
                 If None, computes PCA weights.

    Returns:
        Grayscale image as uint8
    """
    if image.ndim == 2:
        return image

    if weights is None:
        weights = compute_pca_weights(image)

    # Apply weights to channels
    b, g, r = cv2.split(image)
    gray = (
        b.astype(np.float32) * weights[0] +
        g.astype(np.float32) * weights[1] +
        r.astype(np.float32) * weights[2]
    )

    return np.clip(gray, 0, 255).astype(np.uint8)
