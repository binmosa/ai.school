"""Command-line tool for framework management."""

import os
import sys
from pathlib import Path

import click


@click.group()
def cli():
    """Manage ML framework selection (TensorFlow vs PyTorch)."""


@cli.command()
def info():
    """Display information about available frameworks."""
    from common.framework import get_framework_info

    info_dict = get_framework_info()

    click.echo("ML Framework Information:")
    click.echo("=" * 50)
    click.echo(f"Current Backend: {info_dict['current_backend']}")
    click.echo(f"Keras Version: {info_dict['keras_version']}")
    click.echo()
    click.echo(f"TensorFlow: {info_dict.get('tensorflow_version', 'not installed')}")
    if info_dict.get("tensorflow_gpu"):
        click.echo(f"  GPU Support: {info_dict['tensorflow_gpu']}")
    click.echo()
    click.echo(f"PyTorch: {info_dict.get('torch_version', 'not installed')}")
    if info_dict.get("torch_cuda"):
        click.echo(f"  CUDA Support: {info_dict['torch_cuda']}")


@cli.command()
@click.argument("framework", type=click.Choice(["tensorflow", "torch", "pytorch"]))
def use(framework):
    """Set the default framework for the current session.

    This updates the .env file with the selected framework.
    """
    # Normalize pytorch -> torch
    if framework == "pytorch":
        framework = "torch"

    env_file = Path(".env")

    # Read existing .env or create new
    if env_file.exists():
        lines = env_file.read_text().splitlines()
    else:
        lines = []

    # Update or add KERAS_BACKEND
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("KERAS_BACKEND="):
            lines[i] = f"KERAS_BACKEND={framework}"
            updated = True
            break

    if not updated:
        lines.append(f"KERAS_BACKEND={framework}")

    # Also update ML_FRAMEWORK if it exists
    ml_framework_updated = False
    for i, line in enumerate(lines):
        if line.startswith("ML_FRAMEWORK="):
            lines[i] = f"ML_FRAMEWORK={framework}"
            ml_framework_updated = True
            break

    if not ml_framework_updated:
        lines.append(f"ML_FRAMEWORK={framework}")

    # Write back to .env
    env_file.write_text("\n".join(lines) + "\n")

    # Also set for current process
    os.environ["KERAS_BACKEND"] = framework
    os.environ["ML_FRAMEWORK"] = framework

    framework_name = "TensorFlow" if framework == "tensorflow" else "PyTorch"
    click.echo(f"✓ Framework set to: {framework_name}")
    click.echo("✓ Updated .env file")
    click.echo()
    click.echo("Note: Restart your shell or run 'source .env' to apply changes.")


@cli.command()
def validate():
    """Validate that the selected framework is properly installed."""
    from common.framework import get_current_framework, validate_framework

    try:
        validate_framework()
        framework = get_current_framework()
        framework_name = "TensorFlow" if framework == "tensorflow" else "PyTorch"
        click.echo(f"✓ {framework_name} is properly configured and available.")
    except RuntimeError as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
