# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regroup a model into perforable sub-blocks before PerforatedAI wraps it.

PerforatedAI perforates one ``nn.Module`` at a time and its conventions put every normalization layer inside the
module that receives dendrites. Transformer blocks keep their norms, branches, and residual adds as sibling calls in
one forward, so a *recipe* rebuilds such a block from its existing children into sub-block modules. This file holds
the generic parts: the walker that applies recipes, the sub-block classes recipes assemble, the state-dict key map
that keeps checkpoints loadable, and an equivalence check for tests. It imports nothing from ``rfdetr`` so it can move
into the PerforatedAI library as is.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = [
    "PostNormResidual",
    "PreNormBranch",
    "Recipe",
    "Restructured",
    "assert_equivalent",
    "invert_key_map",
    "layer_with_norm",
    "remap_state_dict",
    "restructure_model",
]

#: Builds the replacement for one module out of that module's own children.
Recipe = Callable[[nn.Module], nn.Module]


# --- Sub-block modules ------------------------------------------------------------------------------------------------


class PreNormBranch(nn.Module):
    """``scale(fn(norm(x)))``: a pre-norm branch whose residual add stays in the parent block.

    A dendrite copy of this module is the branch alone, which is what PerforatedAI needs: the skip connection never
    enters a dendrite because it never enters the wrapped module.

    Args:
        norm: Normalization applied to the branch input.
        fn: The branch body (attention, MLP, ...). Extra ``forward`` arguments are passed through to it.
        scale: Optional per-channel scale applied to the branch output (layer scale).
        select: Index into ``fn``'s output when it returns a tuple.
    """

    #: Axis of the output tensor that indexes neurons. ``-1`` is the channel axis of token tensors.
    neuron_axis: int = -1

    def __init__(
        self,
        norm: nn.Module,
        fn: nn.Module,
        scale: nn.Module | None = None,
        select: int | None = None,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.scale = scale if scale is not None else nn.Identity()
        self.select = select

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the branch.

        Args:
            x: Branch input, ``[..., C]``.
            *args: Passed to ``fn`` after the normalized input.
            **kwargs: Passed to ``fn``.

        Returns:
            The scaled branch output, same shape as ``x``.
        """
        out = self.fn(self.norm(x), *args, **kwargs)
        if self.select is not None:
            out = out[self.select]
        return self.scale(out)


class PostNormResidual(nn.Module):
    """``norm(x + dropout(fn(x)))``: a post-norm block with its residual add and norm inside.

    The neuron keeps the skip connection. A dendrite copy drops it and computes ``norm(dropout(fn(x)))``, which
    :func:`configure_as_dendrite` switches on when PerforatedAI initializes the copy.

    Args:
        fn: The block body. Extra ``forward`` arguments are passed through to it after ``x``.
        dropout: Dropout applied to the body output before the residual add.
        norm: Normalization applied after the residual add.
    """

    neuron_axis: int = -1

    def __init__(self, fn: nn.Module, dropout: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.dropout = dropout
        self.norm = norm
        # A plain attribute, not a buffer: the state dict stays that of the original block. Dendrite copies are made
        # by deep copy and then configured, so the flag travels with the copy.
        self.skip = True

    def configure_as_dendrite(self) -> None:
        """Drop the skip connection: this copy is a dendrite, not the neuron."""
        self.skip = False

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Block input, ``[..., C]``.
            *args: Passed to ``fn`` after ``x``.
            **kwargs: Passed to ``fn``.

        Returns:
            The normalized block output, same shape as ``x``.
        """
        out = self.dropout(self.fn(x, *args, **kwargs))
        return self.norm(x + out if self.skip else out)


def layer_with_norm(layer: nn.Module, norm: nn.Module) -> nn.Module:
    """Group a layer with the normalization that follows it, PerforatedAI's ``PAISequential`` convention.

    Args:
        layer: The weight-carrying layer.
        norm: The normalization applied to its output.

    Returns:
        A ``PAISequential`` running ``norm(layer(x))``.
    """
    # Optional-dependency boundary: the PAI container is only needed when a model is restructured for dendrites.
    from perforatedai import globals_perforatedai as gpa

    sequential: nn.Module = gpa.PAISequential([layer, norm])
    return sequential


# --- Walker -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Restructured:
    """Result of :func:`restructure_model`.

    Args:
        model: The restructured model (the same object that was passed in).
        key_map: State-dict key of the original model to the key of the same tensor in the restructured model.
        replaced: Ids (``named_modules`` names) of the modules a recipe rebuilt, in walk order.
    """

    model: nn.Module
    key_map: dict[str, str]
    replaced: tuple[str, ...]


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at a dotted ``named_modules`` name.

    Args:
        model: Root module.
        name: Dotted name of the submodule to replace.
        module: Replacement.
    """
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, module)


def map_state_dict_keys(
    old: Mapping[str, torch.Tensor],
    new: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """Pair every tensor of the original state dict with its key in the restructured one, by tensor identity.

    Args:
        old: ``state_dict(keep_vars=True)`` before restructuring.
        new: ``state_dict(keep_vars=True)`` after restructuring.

    Returns:
        ``{old_key: new_key}`` covering every key of ``old``.

    Raises:
        ValueError: If a recipe dropped a tensor, created one, or moved a tied tensor ambiguously.
    """
    old_by_id: dict[int, list[str]] = defaultdict(list)
    new_by_id: dict[int, list[str]] = defaultdict(list)
    for key, tensor in old.items():
        old_by_id[id(tensor)].append(key)
    for key, tensor in new.items():
        new_by_id[id(tensor)].append(key)

    key_map: dict[str, str] = {}
    for tensor_id, new_keys in new_by_id.items():
        old_keys = old_by_id.get(tensor_id)
        if old_keys is None:
            raise ValueError(f"Restructuring created tensors that the original model did not have: {new_keys}.")
        moved_old = [k for k in old_keys if k not in new_keys]
        moved_new = [k for k in new_keys if k not in old_keys]
        for key in old_keys:
            if key in new_keys:
                key_map[key] = key
        if len(moved_old) != len(moved_new):
            raise ValueError(f"Restructuring changed how a tied tensor is shared: {old_keys} became {new_keys}.")
        for old_key, new_key in zip(sorted(moved_old), sorted(moved_new)):
            key_map[old_key] = new_key
    missing = sorted(set(old) - set(key_map))
    if missing:
        raise ValueError(f"Restructuring dropped tensors of the original model: {missing}.")
    return key_map


def restructure_model(model: nn.Module, recipes: Mapping[type[nn.Module], Recipe]) -> Restructured:
    """Rebuild every module whose exact type has a recipe, in place, and map the state-dict keys.

    A recipe receives the original module and returns its replacement built from the original's children, so no
    weights are copied or re-initialized. A recipe may also modify the module in place and return it. The walk restarts
    after every replacement, so recipes may nest (a block recipe inside a container another recipe rebuilt).

    Args:
        model: Model with its weights already loaded.
        recipes: Exact module type to the recipe that rebuilds it.

    Returns:
        The model, the key map, and the ids that were rebuilt.
    """
    old_state = model.state_dict(keep_vars=True)
    handled: set[int] = set()
    replaced: list[str] = []
    while True:
        match = next(
            (
                (name, module)
                for name, module in model.named_modules()
                if name and id(module) not in handled and type(module) in recipes
            ),
            None,
        )
        if match is None:
            break
        name, module = match
        replacement = recipes[type(module)](module)
        handled.add(id(module))
        handled.add(id(replacement))
        if replacement is not module:
            set_submodule(model, name, replacement)
        replaced.append(name)
    key_map = map_state_dict_keys(old_state, model.state_dict(keep_vars=True))
    return Restructured(model=model, key_map=key_map, replaced=tuple(replaced))


def invert_key_map(key_map: Mapping[str, str]) -> dict[str, str]:
    """Return ``{new_key: old_key}`` for a map produced by :func:`restructure_model`."""
    # The forward map is one-to-one, so the inverse is a plain swap.
    return {new: old for old, new in key_map.items()}


def remap_state_dict(state: Mapping[str, torch.Tensor], key_map: Mapping[str, str]) -> dict[str, torch.Tensor]:
    """Rename the keys of a state dict; keys absent from the map are kept as they are.

    Args:
        state: State dict to rename.
        key_map: ``{from_key: to_key}``.

    Returns:
        A new state dict.
    """
    # Keys outside the map (PAI bookkeeping, unchanged modules) pass through untouched.
    return {key_map.get(key, key): tensor for key, tensor in state.items()}


# --- Equivalence ------------------------------------------------------------------------------------------------------


def flatten_outputs(output: Any) -> list[torch.Tensor]:
    """Collect every tensor in a nested output of tensors, sequences, and dicts, in a stable order.

    Args:
        output: Model output.

    Returns:
        The tensors, depth first, dict entries by sorted key.
    """
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, Mapping):
        return [t for key in sorted(output) for t in flatten_outputs(output[key])]
    if isinstance(output, Sequence) and not isinstance(output, str | bytes):
        return [t for item in output for t in flatten_outputs(item)]
    return []


def assert_equivalent(
    reference: nn.Module,
    model: nn.Module,
    inputs: Sequence[Any],
    *,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Fail if two models disagree on ``inputs`` in eval or train mode.

    Both forwards start from the same random seed, so dropout and stochastic depth draw the same masks as long as the
    restructured model calls them in the original order.

    Args:
        reference: A deep copy of the model taken before :func:`restructure_model`.
        model: The restructured model.
        inputs: Positional forward arguments.
        seed: Seed set before each forward.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Raises:
        AssertionError: If the outputs differ in count or value.
    """
    was_training = (reference.training, model.training)
    try:
        for training in (False, True):
            reference.train(training)
            model.train(training)
            with torch.no_grad():
                torch.manual_seed(seed)
                expected = flatten_outputs(reference(*inputs))
                torch.manual_seed(seed)
                actual = flatten_outputs(model(*inputs))
            assert len(expected) == len(actual), f"{len(actual)} output tensors, expected {len(expected)}"
            for index, (a, b) in enumerate(zip(expected, actual)):
                assert a.shape == b.shape, f"output {index}: shape {tuple(b.shape)}, expected {tuple(a.shape)}"
                torch.testing.assert_close(b, a, atol=atol, rtol=rtol, msg=lambda m, i=index: f"output {i}: {m}")
    finally:
        reference.train(was_training[0])
        model.train(was_training[1])
