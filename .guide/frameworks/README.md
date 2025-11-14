# Framework Selection: TensorFlow vs PyTorch

This project supports both **TensorFlow** and **PyTorch** as backend frameworks, powered by **Keras 3's multi-backend support**.

## Quick Start

### Check Current Framework

```bash
just framework-info
```

Output:
```
Current ML Framework Configuration:
======================================
KERAS_BACKEND: tensorflow
ML Framework Information:
==================================================
Current Backend: tensorflow
Keras Version: 3.11.3

TensorFlow: 2.18.1
  GPU Support: False

PyTorch: 2.5.0
  CUDA Support: False
```

### Switch to TensorFlow

```bash
just use-tensorflow
source .env  # Apply changes
```

### Switch to PyTorch

```bash
just use-pytorch
source .env  # Apply changes
```

## Training with Different Backends

### Train with TensorFlow

```bash
just train-tensorflow
```

### Train with PyTorch

```bash
just train-pytorch
```

### Compare Both Frameworks

```bash
just train-compare
```

This will train the same model with both backends and log results to MLflow for comparison.

## How It Works

Keras 3 provides a unified API that works across multiple backends:

- **TensorFlow**: Production-ready, mature ecosystem
- **PyTorch**: Popular in research, dynamic computation
- **JAX**: (Optional) High-performance, functional programming

The same Keras code runs on all backends with minimal changes!

## Code Example

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or "tensorflow"

from keras import layers, models

# This code works on BOTH backends!
model = models.Sequential([
    layers.Dense(10, activation="relu"),
    layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

## When to Use Each Framework

### Use TensorFlow when:

- Deploying to production (TensorFlow Serving, TFLite)
- Using TensorBoard extensively
- Working with TensorFlow ecosystem tools
- Need mature deployment options

### Use PyTorch when:

- Rapid prototyping and research
- Need dynamic computation graphs
- Prefer PyTorch's debugging experience
- Working with PyTorch-first libraries

## Available Commands

### Framework Management

```bash
# Show framework information
just framework-info

# Switch to TensorFlow
just use-tensorflow

# Switch to PyTorch
just use-pytorch

# Validate current framework
just framework-validate
```

### Training Commands

```bash
# Train with current framework
just train

# Train with specific framework
just train-with tensorflow
just train-with torch

# Train with TensorFlow
just train-tensorflow

# Train with PyTorch
just train-pytorch

# Compare both frameworks
just train-compare
```

### Testing Commands

```bash
# Run all tests with current framework
just test

# Test specific framework
just test-framework tensorflow
just test-framework torch

# Test both frameworks
just test-all
```

### Serving Commands

```bash
# Serve with current framework
just serve

# Serve with specific framework
just serve-with tensorflow
just serve-with torch
```

### Dependency Management

```bash
# Sync all dependencies (both frameworks)
just sync

# Install all frameworks
just install-all

# Install only TensorFlow
just install-tensorflow

# Install only PyTorch
just install-pytorch
```

## Testing

Run tests for both frameworks:

```bash
just test-all
```

Run tests for specific framework:

```bash
just test-framework tensorflow
just test-framework torch
```

## Troubleshooting

### Framework not found

```bash
just framework-validate
```

If validation fails, try reinstalling:

```bash
just install-all
```

### Wrong framework is active

Check your `.env` file:

```bash
cat .env
```

Update it manually or use:

```bash
just use-tensorflow  # or use-pytorch
source .env
```

### Module import errors

Make sure you've synced dependencies:

```bash
just sync
```

### Keras shows wrong backend

The backend is determined when Keras is first imported. Restart your Python process after changing `KERAS_BACKEND`.

## Advanced: Manual Switching

You can also set the backend programmatically:

```python
from common.framework import set_framework

# Must be called BEFORE importing keras
set_framework("torch")

import keras
# Now Keras will use PyTorch backend
```

Or via environment variable:

```bash
export KERAS_BACKEND=torch
uv run src/pipelines/training.py run
```

## Performance Comparison

Both backends should produce similar results, but performance may vary:

- **TensorFlow**: Often faster on CPUs, optimized for production
- **PyTorch**: Often preferred for research, better debugging

Use `just train-compare` to compare training time and accuracy on your hardware.

## Model Compatibility

Models trained with one backend can be loaded with another:

1. Train with TensorFlow:
   ```bash
   KERAS_BACKEND=tensorflow just train
   ```

2. Serve with PyTorch:
   ```bash
   KERAS_BACKEND=torch just serve
   ```

Keras handles the conversion automatically!

## Additional Resources

- [Keras 3 Multi-Backend Guide](https://keras.io/keras_3/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

## Getting Help

If you encounter issues:

1. Check framework is installed: `just framework-validate`
2. Review logs for errors
3. Try with both backends to isolate framework-specific issues
4. Consult the documentation links above
