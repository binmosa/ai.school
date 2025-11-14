"""Tests for framework selection and switching."""

import os

import pytest


def test_framework_detection():
    """Test that framework detection works correctly."""
    from common.framework import get_current_framework

    # Should default to tensorflow or whatever is set
    framework = get_current_framework()
    assert framework in ["tensorflow", "torch", "jax"]


def test_framework_info():
    """Test that framework info is collected correctly."""
    from common.framework import get_framework_info

    info = get_framework_info()

    assert "current_backend" in info
    assert "keras_version" in info
    assert info["current_backend"] in ["tensorflow", "torch", "jax"]


@pytest.mark.tensorflow
def test_tensorflow_available():
    """Test that TensorFlow is available."""
    import tensorflow as tf

    assert tf.__version__


@pytest.mark.pytorch
def test_pytorch_available():
    """Test that PyTorch is available."""
    import torch

    assert torch.__version__


def test_keras_import():
    """Test that Keras can be imported with current backend."""
    import keras

    assert keras.__version__


def test_framework_validation():
    """Test framework validation."""
    from common.framework import validate_framework

    # Should not raise if framework is properly installed
    validate_framework()


def test_invalid_backend():
    """Test that invalid backend raises error."""
    from common.framework import get_current_framework

    # Save current backend
    original_backend = os.getenv("KERAS_BACKEND")

    try:
        os.environ["KERAS_BACKEND"] = "invalid_backend"
        with pytest.raises(ValueError, match="Invalid KERAS_BACKEND"):
            get_current_framework()
    finally:
        # Restore original backend
        if original_backend:
            os.environ["KERAS_BACKEND"] = original_backend
        else:
            os.environ.pop("KERAS_BACKEND", None)


def test_keras_model_creation():
    """Test that a simple Keras model can be created with current backend."""
    import keras
    import numpy as np

    # Create a simple model
    model = keras.Sequential(
        [
            keras.layers.Dense(10, activation="relu", input_shape=(5,)),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )

    # Test forward pass
    x = np.random.randn(2, 5).astype(np.float32)
    output = model(x)

    assert output.shape == (2, 3)


def test_model_compilation():
    """Test that a model can be compiled with current backend."""
    import keras

    model = keras.Sequential(
        [
            keras.layers.Dense(10, activation="relu", input_shape=(5,)),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )

    # Should compile without errors
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    assert model.optimizer is not None
