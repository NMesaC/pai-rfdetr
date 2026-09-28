# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""EMA bookkeeping for a PerforatedAI model, independent of the training framework.

Two problems come up whenever a trainer validates an EMA copy of a perforated model. First, PAI checkpoints the model
object it is handed, so the weights it restores at a switch must be the EMA weights that were scored: swap them into
the live model around ``add_validation_score`` and swap them back when PAI does not restructure. Second, PAI registers
buffers lazily and resizes some at every validation, so an EMA that pairs tensors by position (``AveragedModel``)
misaligns: update by name instead, and mirror PAI's own buffers rather than averaging them, since averaging an integer
index through float rounding once broadcast dendrite weights on the wrong axis. This file imports nothing from
``rfdetr`` so it can move into the PerforatedAI library as is.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

__all__ = [
    "ema_update_by_name",
    "ema_weight_keys",
    "pai_buffer_names",
    "replace_buffer",
    "restore_weights",
    "swap_in_weights",
    "sync_resized_buffers",
]


def ema_weight_keys(model: nn.Module) -> set[str]:
    """State-dict keys the EMA averages and validation scores: parameters plus norm running statistics.

    PAI's float buffers (``initialized``, ``current_d_init``, ``num_cycles``) are left out on purpose: copying an
    averaged copy of them into the live model corrupts its dendrite bookkeeping.

    Args:
        model: Live perforated model.

    Returns:
        Parameter names plus ``running_mean`` / ``running_var`` buffer names.
    """
    keys = {name for name, _ in model.named_parameters()}
    keys |= {name for name, _ in model.named_buffers() if name.endswith(("running_mean", "running_var"))}
    return keys


def pai_buffer_names(model: nn.Module) -> set[str]:
    """State-dict keys of the buffers under every ``PAINeuronModule``.

    Args:
        model: Live perforated model.

    Returns:
        Buffer names PAI keeps for bookkeeping.
    """
    prefixes = [name + "." for name, module in model.named_modules() if type(module).__name__ == "PAINeuronModule"]
    return {name for name, _ in model.named_buffers() if any(name.startswith(p) for p in prefixes)}


@torch.no_grad()
def swap_in_weights(live: nn.Module, source: nn.Module, keys: Iterable[str]) -> dict[str, torch.Tensor]:
    """Copy ``source``'s tensors into ``live`` in place and return the overwritten ones.

    Copying in place keeps the module objects PAI's tracker holds.

    Args:
        live: Model whose tensors are overwritten.
        source: Model the tensors are read from (the EMA copy).
        keys: State-dict keys to copy, typically :func:`ema_weight_keys` of ``live``.

    Returns:
        ``{key: clone of the previous live tensor}`` for :func:`restore_weights`.
    """
    live_state = live.state_dict()
    source_state = source.state_dict()
    saved: dict[str, torch.Tensor] = {}
    for key in keys:
        saved[key] = live_state[key].clone()
        live_state[key].copy_(source_state[key])
    return saved


@torch.no_grad()
def restore_weights(live: nn.Module, saved: dict[str, torch.Tensor]) -> None:
    """Put the tensors :func:`swap_in_weights` overwrote back in place.

    Args:
        live: Model to restore.
        saved: Return value of :func:`swap_in_weights`.
    """
    live_state = live.state_dict()
    for key, tensor in saved.items():
        live_state[key].copy_(tensor)


def replace_buffer(root: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Re-register a nested buffer with a copy of ``tensor``, for buffers whose shape changed.

    Args:
        root: Module the dotted name is relative to.
        name: Dotted state-dict name of the buffer.
        tensor: Tensor whose copy becomes the buffer.
    """
    owner_name, _, attr = name.rpartition(".")
    owner = root.get_submodule(owner_name) if owner_name else root
    owner.register_buffer(attr, tensor.detach().clone())


@torch.no_grad()
def sync_resized_buffers(ema: nn.Module, live: nn.Module) -> int:
    """Replace every non-float buffer of the EMA copy whose live shape changed.

    Args:
        ema: EMA copy.
        live: Live model with the same module tree.

    Returns:
        Number of buffers replaced.
    """
    ema_buffers = dict(ema.named_buffers())
    replaced = 0
    for name, live_t in live.named_buffers():
        ema_t = ema_buffers.get(name)
        if ema_t is None or ema_t.is_floating_point():
            continue
        if ema_t.shape != live_t.shape:
            replace_buffer(ema, name, live_t)
            replaced += 1
    return replaced


@torch.no_grad()
def ema_update_by_name(
    ema: nn.Module,
    live: nn.Module,
    decay: float,
    mirror: set[str],
    *,
    first_update: bool = False,
) -> None:
    """One EMA step that pairs tensors by state-dict name, mirroring instead of averaging where asked.

    Tensors named in ``mirror``, non-float tensors, and every tensor on the first update are copied. Floating tensors
    whose shape changed are skipped (they are PAI's lazily resized bookkeeping); non-float ones are re-registered at the
    new shape. Everything else is averaged as ``ema = decay * ema + (1 - decay) * live`` in one multi-tensor launch per
    device and dtype.

    Args:
        ema: EMA copy to update in place.
        live: Live model.
        decay: EMA decay for this step.
        mirror: State-dict names to copy rather than average, typically :func:`pai_buffer_names`.
        first_update: Copy everything, for the step that seeds a fresh copy.
    """
    ema_tensors: dict[str, torch.Tensor] = dict(ema.named_parameters())
    ema_tensors.update(ema.named_buffers())
    live_tensors: dict[str, torch.Tensor] = dict(live.named_parameters())
    live_tensors.update(live.named_buffers())
    groups: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for name, ema_t in ema_tensors.items():
        live_t = live_tensors.get(name)
        if live_t is None:
            continue
        if live_t.shape != ema_t.shape:
            if not ema_t.is_floating_point():
                replace_buffer(ema, name, live_t)
            continue
        if name in mirror or first_update or not ema_t.is_floating_point():
            ema_t.copy_(live_t)
            continue
        averaged, current = groups.setdefault((ema_t.device, ema_t.dtype), ([], []))
        averaged.append(ema_t)
        current.append(live_t.to(ema_t.dtype))
    for averaged, current in groups.values():
        torch._foreach_mul_(averaged, decay)
        torch._foreach_add_(averaged, current, alpha=1.0 - decay)
