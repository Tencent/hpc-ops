from __future__ import annotations

import os
import subprocess

__all__ = ["dynamic_metadata"]

base_version = "0.0.1"


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
) -> str:
    try:
        assert field == "version"

        patched_version = base_version

        # Try to add git hash
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            patched_version += f".dev0+g{git_hash}"
        except Exception:
            pass

        # VERSION_SUFFIX='+cu128'
        if version_ext := os.environ.get("VERSION_SUFFIX"):
            patched_version = base_version + version_ext
        elif cuda_version := os.environ.get("CUDA_VERSION"):
            major, minor, *_ = cuda_version.split(".")
            backend = f"+cu{major}{minor}"
            patched_version = base_version + backend

        return patched_version
    except Exception:
        return base_version
