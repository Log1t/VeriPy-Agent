from __future__ import annotations

from collections.abc import Callable
from typing import Any


def install_make_envs_patch(
    module: Any, make_veripy_envs: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Explicitly install the factory seam before verl imports its rollout entrypoint."""

    original = module.make_envs

    def patched(config: Any) -> Any:
        if str(config.env.env_name).lower() == "veripy":
            return make_veripy_envs(config)
        return original(config)

    module.make_envs = patched
    return original
