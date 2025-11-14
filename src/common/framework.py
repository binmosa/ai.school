"""Framework selection and management utilities.

This module provides utilities for managing the Keras backend and framework selection
between TensorFlow and PyTorch. Since Keras 3 supports multiple backends, most code
can remain framework-agnostic.
"""

import os
import sys
from typing import Literal

Framework = Literal["tensorflow", "torch", "jax"]


def get_current_framework() -> Framework:
    """Get the currently configured Keras backend.

    Returns:
        The current framework: 'tensorflow', 'torch', or 'jax'
    """
    backend = os.getenv("KERAS_BACKEND", "tensorflow")
    if backend not in ["tensorflow", "torch", "jax"]:
        msg = (
            f"Invalid KERAS_BACKEND: {backend}. "
            "Must be 'tensorflow', 'torch', or 'jax'"
        )
        raise ValueError(msg)
    return backend


def set_framework(framework: Framework) -> None:
    """Set the Keras backend for the current process.

    Warning: This must be called BEFORE importing keras!

    Args:
        framework: The framework to use ('tensorflow', 'torch', or 'jax')
    """
    if "keras" in sys.modules:
        msg = (
            "Cannot change framework after Keras has been imported. "
            "Set KERAS_BACKEND before importing keras."
        )
        raise RuntimeError(msg)

    os.environ["KERAS_BACKEND"] = framework


def get_framework_info() -> dict[str, str]:
    """Get information about available frameworks and current selection.

    Returns:
        Dictionary with framework versions and current backend
    """
    import importlib.metadata

    info = {
        "current_backend": get_current_framework(),
        "keras_version": importlib.metadata.version("keras"),
    }

    # Check TensorFlow availability
    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
        info["tensorflow_gpu"] = str(len(tf.config.list_physical_devices("GPU")) > 0)
    except ImportError:
        info["tensorflow_version"] = "not installed"

    # Check PyTorch availability
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda"] = str(torch.cuda.is_available())
    except ImportError:
        info["torch_version"] = "not installed"

    return info


def validate_framework() -> None:
    """Validate that the selected framework is properly installed.

    Raises:
        RuntimeError: If the selected framework is not available
    """
    framework = get_current_framework()

    try:
        if framework == "tensorflow":
            import tensorflow  # noqa: F401
        elif framework == "torch":
            import torch  # noqa: F401
    except ImportError as e:
        msg = (
            f"Framework '{framework}' is not installed. "
            f"Please install it with: uv add {framework}"
        )
        raise RuntimeError(msg) from e


if __name__ == "__main__":
    # Command-line interface for framework info
    import json

    info = get_framework_info()
    print(json.dumps(info, indent=2))
