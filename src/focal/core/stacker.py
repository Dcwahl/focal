"""Focus stacking algorithm."""

from pathlib import Path
from typing import Callable
import numpy as np


class FocusStacker:
    """Laplacian pyramid focus stacking."""

    def stack(
        self,
        image_paths: list[Path],
        progress_callback: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """
        Stack images using Laplacian pyramid method.

        Args:
            image_paths: List of paths to source images
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            Stacked result as numpy array (H, W, C) in uint8
        """
        # TODO: Implement Laplacian pyramid stacking
        # Reference: pyfocusstack implementation
        raise NotImplementedError("Stacking not yet implemented")
