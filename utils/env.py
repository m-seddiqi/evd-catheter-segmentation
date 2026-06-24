from __future__ import annotations

import os


def require_conda_env(expected_env: str | None) -> None:
    if not expected_env:
        return
    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    if active != expected_env:
        raise EnvironmentError(
            f"Active conda env is '{active}' (expected '{expected_env}'). "
            f"Run with: conda run -n {expected_env} python ..."
        )
