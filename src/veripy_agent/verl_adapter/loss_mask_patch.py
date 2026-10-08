from __future__ import annotations

from functools import wraps
from typing import Any


def install_agent_rollout_loss_mask_patch(trajectory_collector_class: Any) -> Any:
    """Bridge the legacy agent collector to veRL's multi-turn loss-mask contract.

    Before generation, the pinned collector pops the prompt tensors from its
    working batch and then unions the generator output back in.  Its final
    ``input_ids`` and ``attention_mask`` therefore cover prompt plus response,
    while ``responses`` contains only assistant policy tokens; tool observations
    are carried in the following prompt.  At this collector boundary, the full
    attention mask is therefore the correct loss mask, and veRL later slices its
    response-length tail.  Newer collectors that already provide ``loss_mask``
    are left untouched.
    """

    installed = getattr(trajectory_collector_class, "_veripy_loss_mask_patch", None)
    if installed is not None:
        return installed

    original = trajectory_collector_class.gather_rollout_data

    @wraps(original)
    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        rollout = original(self, *args, **kwargs)
        batch = rollout.batch
        if "loss_mask" in batch:
            return rollout

        if "input_ids" not in batch or "attention_mask" not in batch:
            raise KeyError(
                "agent rollout requires input_ids and attention_mask to derive loss_mask"
            )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "cannot derive loss_mask: input_ids and attention_mask shapes differ "
                f"({input_ids.shape} != {attention_mask.shape})"
            )

        batch["loss_mask"] = attention_mask.clone()
        return rollout

    trajectory_collector_class.gather_rollout_data = patched
    trajectory_collector_class._veripy_loss_mask_patch = original
    return original
