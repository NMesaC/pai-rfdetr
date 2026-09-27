# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Perforated Integration to Roboflow"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, PerforationTargetName, TrainConfig
from rfdetr.training.callbacks import RFDETREMACallback
from rfdetr.training.callbacks.coco_eval import _get_ema_inner_module
from rfdetr.training.param_groups import get_param_dict
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.training.module_model import RFDETRModelModule

logger = get_logger()

__all__ = [
    "CLEAN_MODEL_NAME",
    "PERFORATION_TARGETS",
    "PerforatedAICallback",
    "PerforationTarget",
    "build_perforated_save_name",
    "clean_perforated_model",
    "extract_plain_state",
    "extract_start_weights",
    "get_perforation_target",
    "list_perforable_modules",
    "perforate_detection_model",
    "resolve_perforate_ids",
    "setup_perforated_optimizer",
]

# --- Targets ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PerforationTarget:
    """Detector Modules that receive dendrites

    The neuron axis of every wrapped module is measured by :func:`record_forward_stats`, so a target only names modules.

    Args:
        name: Short name, used in the PAI save folder name.
        perforate_ids: Module ids (``named_modules`` names with a leading dot) that receive dendrites.
        perforate_type_names: Module type names that receive dendrites wherever they occur.
        skip_ids: Module ids PAI wraps as tracked modules instead of perforating, even when a type name matches.
            :func:`perforate_detection_model` adds every candidate called more than once per forward, which PAI's
            perforated backpropagation cannot pair with a single error.
        track_ids: Module ids PAI wraps in ``TrackedNeuronModule``; ``pai_track_leaves`` fills it.
    """

    name: str
    perforate_ids: tuple[str, ...] = ()
    perforate_type_names: tuple[str, ...] = ()
    skip_ids: tuple[str, ...] = ()
    track_ids: tuple[str, ...] = ()


ALL = PerforationTarget(name="all", perforate_type_names=("Linear", "Conv2d"))
CLS_HEAD = PerforationTarget(name="cls_head", perforate_ids=(".class_embed",))
BBOX_HEAD = PerforationTarget(name="bbox_head", perforate_ids=(".bbox_embed",))
CLS_BBOX_HEAD = PerforationTarget(name="cls_bbox_head", perforate_ids=(".class_embed", ".bbox_embed"))

#: Targets selectable through ``TrainConfig.pai_target``.
PERFORATION_TARGETS: dict[str, PerforationTarget] = {t.name: t for t in (ALL, CLS_HEAD, BBOX_HEAD, CLS_BBOX_HEAD)}

#: Module types :func:`list_perforable_modules` reports and :func:`record_forward_stats` measures.
PERFORABLE_TYPE_NAMES = ("Linear", "Conv2d", "LayerNorm", "Embedding", "MLP")


def get_perforation_target(name: PerforationTargetName | str) -> PerforationTarget:
    """Look up a target by name.

    Args:
        name: One of the keys of :data:`PERFORATION_TARGETS`.

    Returns:
        The matching :class:`PerforationTarget`.

    Raises:
        ValueError: If ``name`` is not a known target.
    """
    try:
        return PERFORATION_TARGETS[name]
    except KeyError:
        known = ", ".join(sorted(PERFORATION_TARGETS))
        raise ValueError(f"Unknown pai_target {name!r}; choose from {known}.") from None


def list_perforable_modules(model: nn.Module) -> dict[str, str]:
    """Map every weight-carrying module id of the detector to its type name.

    Args:
        model: ``LWDETR`` before PAI converts it.

    Returns:
        ``{module_id: type_name}`` for every module whose type is in :data:`PERFORABLE_TYPE_NAMES`.
    """
    return {
        "." + name: type(module).__name__
        for name, module in model.named_modules()
        if name and type(module).__name__ in PERFORABLE_TYPE_NAMES
    }


def check_target_ids(model: nn.Module, target: PerforationTarget) -> None:
    """Reject target ids the detector does not have.

    Args:
        model: ``LWDETR`` before PAI converts it.
        target: Target whose ids are checked.

    Raises:
        ValueError: If any id in ``target.perforate_ids`` names a module the model lacks.
    """
    names = {"." + name for name, _ in model.named_modules()}
    missing = [module_id for module_id in target.perforate_ids if module_id not in names]
    if missing:
        raise ValueError(
            f"Target {target.name!r} names modules the detector does not have: {missing}. "
            "Run list_perforable_modules on the model for the ids it does have."
        )


def resolve_perforate_ids(model: nn.Module, target: PerforationTarget) -> tuple[str, ...]:
    """Return every module id the target wraps: its ids plus every module of its type names, minus ``skip_ids``.

    Args:
        model: ``LWDETR`` before PAI converts it.
        target: Modules that receive dendrites.

    Returns:
        Outermost module ids with the leading dot PAI matches against.
    """
    ids = list(target.perforate_ids)
    for name, module in model.named_modules():
        if name and type(module).__name__ in target.perforate_type_names and "." + name not in ids:
            ids.append("." + name)
    ids = [i for i in ids if i not in target.skip_ids]
    return tuple(i for i in ids if not any(i.startswith(o + ".") for o in ids))


def _inside_target(module_id: str, ids: tuple[str, ...], *, include_self: bool) -> bool:
    """Return whether ``module_id`` is a perforated module or lies below one."""
    return any(module_id.startswith(p + ".") or (include_self and module_id == p) for p in ids)


def build_tracked_parameter_ids(model: nn.Module, target: PerforationTarget) -> list[str]:
    """Return every parameter id outside the perforated modules, for ``parameter_ids_to_track``.

    Args:
        model: ``LWDETR`` before PAI converts it.
        target: Modules that receive dendrites.

    Returns:
        Parameter ids with the leading dot PAI matches against.
    """
    # Perforated and tracked modules set parameter_type on their own parameters.
    ids = resolve_perforate_ids(model, target) + target.skip_ids
    return [
        "." + name for name, _ in model.named_parameters() if not _inside_target("." + name, ids, include_self=False)
    ]


def build_tracked_leaf_ids(model: nn.Module, target: PerforationTarget) -> list[str]:
    """Return every non-perforated leaf module id (has parameters, no children), for ``pai_track_leaves``.

    Args:
        model: ``LWDETR`` before PAI converts it.
        target: Modules that receive dendrites.

    Returns:
        Module ids with the leading dot PAI matches against.
    """
    perforated = resolve_perforate_ids(model, target)
    ids: list[str] = []
    for name, module in model.named_modules():
        if not name or list(module.children()) or not list(module.parameters(recurse=False)):
            continue
        if not _inside_target("." + name, perforated, include_self=True):
            ids.append("." + name)
    return ids


def record_forward_stats(model: nn.Module, sample: torch.Tensor) -> tuple[dict[str, int], dict[str, int]]:
    """Run one forward of the unperforated model and measure every perforable module.

    The pass runs in train mode because RF-DETR calls the extra ``group_detr`` head copies only while training.
    Buffers are restored afterwards so BatchNorm statistics do not see the sample.

    Args:
        model: ``LWDETR`` before PAI converts it.
        sample: Image batch of at least two images, ``[B, 3, H, W]``, on the model's device.

    Returns:
        ``(ranks, calls)``: the output rank and the number of calls per forward of every module whose type is in
        :data:`PERFORABLE_TYPE_NAMES`, keyed by module id.
    """
    ranks: dict[str, int] = {}
    calls: dict[str, int] = {}

    def make_hook(module_id: str) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            calls[module_id] = calls.get(module_id, 0) + 1
            if isinstance(output, torch.Tensor):
                ranks[module_id] = output.ndim

        return hook

    handles = [
        module.register_forward_hook(make_hook("." + name))
        for name, module in model.named_modules()
        if name and type(module).__name__ in PERFORABLE_TYPE_NAMES
    ]
    buffers = {name: buffer.clone() for name, buffer in model.named_buffers()}
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            model(sample)
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
    return ranks, calls


def output_dimensions_for(type_name: str, rank: int) -> list[int]:
    """Return PAI's output dimension vector: ``0`` at the neuron axis, ``-1`` elsewhere.

    Args:
        type_name: Type name of the wrapped module. Convolutions keep channels at axis 1; everything else at the end.
        rank: Rank of the module output.

    Returns:
        A list of length ``rank`` with a single ``0``.
    """
    dims = [-1] * rank
    dims[1 if type_name.startswith("Conv") else rank - 1] = 0
    return dims


def apply_output_dimensions(model: nn.Module, ranks: dict[str, int]) -> None:
    """Set every ``PAINeuronModule``'s neuron axis from the ranks :func:`record_forward_stats` measured.

    Args:
        model: Model returned by ``perforate_model``.
        ranks: ``{module_id: output.ndim}`` of the unperforated model.

    Raises:
        ValueError: If a wrapped module was not called during the measuring forward.
    """
    wrapped = [module for module in model.modules() if type(module).__name__ == "PAINeuronModule"]
    missing = [str(module.name) for module in wrapped if str(module.name) not in ranks]
    if missing:
        raise ValueError(f"PerforatedAI wrapped modules the measuring forward never called: {missing}.")
    for module in wrapped:
        type_name = type(module.get_submodule("main_module")).__name__
        cast(Any, module).set_this_output_dimensions(output_dimensions_for(type_name, ranks[str(module.name)]))


# --- PAI configuration and wrapping -----------------------------------------------------------------------------------


def resolve_forward_function(
    spec: str | Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve ``TrainConfig.pai_forward_function`` to the callable PAI applies to dendrite outputs.

    Args:
        spec: A callable, ``"identity"``, or the name of a function on ``torch`` or ``torch.nn.functional``.

    Returns:
        The nonlinearity.

    Raises:
        ValueError: If a name matches nothing.
    """
    if callable(spec):
        return spec
    if spec == "identity":
        return nn.Identity()
    function = getattr(torch, spec, None) or getattr(torch.nn.functional, spec, None)
    if function is None or not callable(function):
        raise ValueError(
            f"Unknown pai_forward_function {spec!r}; pass 'identity', a torch or torch.nn.functional name, "
            "or a callable."
        )
    return cast(Callable[[torch.Tensor], torch.Tensor], function)


def import_perforatedai() -> tuple[Any, Any]:
    """Import the PerforatedAI globals and utils modules.

    Returns:
        ``(globals_perforatedai, utils_perforatedai)``.

    Raises:
        ImportError: If the optional ``rfdetr[pai]`` extra is not installed.
    """
    try:
        from perforatedai import globals_perforatedai as gpa
        from perforatedai import utils_perforatedai as upa
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("perforatedai"):
            raise
        raise ImportError(
            "PerforatedAI is not installed but perforate=True was requested. "
            'Install it with `pip install "rfdetr[pai]"` and try again.'
        ) from exc
    return gpa, upa


def build_perforated_save_name(model_config: ModelConfig, train_config: TrainConfig) -> str:
    """Return the PAI system folder name for a run.

    Args:
        model_config: Architecture configuration; ``model_name`` or the config class name labels the folder.
        train_config: Training configuration; ``pai_save_name`` is used verbatim when set.

    Returns:
        ``pai_save_name``, or ``"<model>_<target>_dendritic_<timestamp>"`` in lowercase.
    """
    if train_config.pai_save_name:
        return train_config.pai_save_name
    model_label = model_config.model_name or type(model_config).__name__.removesuffix("Config")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_label.lower()}_{train_config.pai_target}_dendritic_{timestamp}"


def configure_perforated_ai(model: nn.Module, target: PerforationTarget, train_config: TrainConfig) -> None:
    """Push every PAI setting for this run into the library's global config.

    Args:
        model: ``LWDETR`` before PAI converts it.
        target: Modules that receive dendrites.
        train_config: Training configuration with ``perforate=True``.

    Raises:
        ValueError: If ``pai_forward_function`` names nothing on ``torch``.
    """
    gpa, _ = import_perforatedai()
    tc = train_config
    check_target_ids(model, target)
    gpa.pc.set_testing_dendrite_capacity(tc.pai_testing_dendrite_capacity)
    gpa.pc.set_module_names_to_perforate(list(target.perforate_type_names))
    gpa.pc.set_module_ids_to_perforate(list(target.perforate_ids))
    gpa.pc.append_module_ids_to_track(list(target.track_ids) + list(target.skip_ids))
    gpa.pc.set_parameter_ids_to_track(build_tracked_parameter_ids(model, target))
    # A default only: apply_output_dimensions sets every wrapped module's axis from the measured ranks.
    gpa.pc.set_output_dimensions([-1, -1, 0])
    gpa.pc.set_n_epochs_to_switch(tc.pai_n_epochs_to_switch)
    gpa.pc.set_p_epochs_to_switch(tc.pai_p_epochs_to_switch)
    # 0 is resolved by the callback at train start, once the dataloader length is known.
    if tc.pai_initial_correlation_batches:
        gpa.pc.set_initial_correlation_batches(tc.pai_initial_correlation_batches)
    gpa.pc.set_max_dendrite_tries(tc.pai_max_dendrite_tries)
    gpa.pc.set_max_dendrites(tc.pai_max_dendrites)
    gpa.pc.set_perforated_backpropagation(True)
    # Most of RF-DETR is left unwrapped on purpose, the targets are fixed in code rather than picked at PAI's prompt,
    # and the model's own weights keep RF-DETR's weight decay (dendrite parameters get 0).
    gpa.pc.set_unwrapped_modules_confirmed(True)
    gpa.pc.set_weight_decay_accepted(True)
    gpa.pc.set_configuration_confirmed(True)

    if tc.pai_candidate_init_mult is not None:
        gpa.pc.set_candidate_weight_initialization_multiplier(tc.pai_candidate_init_mult)
    if tc.pai_candidate_init_by_main is not None:
        gpa.pc.set_candidate_weight_init_by_main(tc.pai_candidate_init_by_main)
    if tc.pai_global_candidates is not None:
        gpa.pc.set_global_candidates(tc.pai_global_candidates)
        if tc.pai_global_candidates > 1:
            # Extra candidates register their score buffers lazily, so the strict load at the first switch fails, and
            # their stacked backward needs PAI's workaround for models without recurrent layers.
            gpa.pc.set_strict_loading(False)
            gpa.pc.set_no_backward_workaround(True)
    if tc.pai_forward_function is not None:
        gpa.pc.set_pai_forward_function(resolve_forward_function(tc.pai_forward_function))

    if tc.pai_fixed_switch_every is not None:
        gpa.pc.set_switch_mode(gpa.pc.DOING_FIXED_SWITCH)
        gpa.pc.set_first_fixed_switch_num(tc.pai_fixed_switch_every)
        gpa.pc.set_fixed_switch_num(tc.pai_fixed_switch_every)


def verify_perforated_wrapping(model: nn.Module, expected_ids: tuple[str, ...]) -> None:
    """Fail if PAI wrapped a different set of modules than requested.

    Args:
        model: Model returned by ``perforate_model``.
        expected_ids: Ids from :func:`resolve_perforate_ids`.

    Raises:
        RuntimeError: If the wrapped ids differ from ``expected_ids``.
    """
    wrapped = sorted(str(module.name) for module in model.modules() if type(module).__name__ == "PAINeuronModule")
    requested = sorted(expected_ids)
    if wrapped != requested:
        raise RuntimeError(f"PerforatedAI wrapped {wrapped} but {requested} were requested.")
    logger.info("PerforatedAI wrapped %d modules, as requested", len(wrapped))


def perforate_detection_model(model: nn.Module, model_config: ModelConfig, train_config: TrainConfig) -> nn.Module:
    """Configure PAI, wrap the detector, verify the wrapping, and optionally load a saved PAI system.

    Args:
        model: ``LWDETR`` with pretrained weights already loaded.
        model_config: Architecture configuration, used to label the PAI save folder.
        train_config: Training configuration with ``perforate=True``.

    Returns:
        The perforated model PAI returned.
    """
    gpa, upa = import_perforatedai()
    target = get_perforation_target(train_config.pai_target)
    if train_config.pai_track_leaves:
        target = replace(target, track_ids=tuple(build_tracked_leaf_ids(model, target)))
        logger.info("PerforatedAI tracking %d leaf modules", len(target.track_ids))
    device = next(model.parameters()).device
    resolution = model_config.resolution
    ranks, calls = record_forward_stats(model, torch.zeros(2, 3, resolution, resolution, device=device))
    # PAI's perforated backpropagation pairs one output with one error per module per step, so a candidate RF-DETR
    # calls more than once per forward (the two-stage class-head copies) is tracked instead of perforated.
    multi_call = tuple(i for i in resolve_perforate_ids(model, target) if calls.get(i, 0) > 1)
    if multi_call:
        target = replace(target, skip_ids=target.skip_ids + multi_call)
        logger.info(
            "PerforatedAI tracking %d modules called more than once per forward: %s", len(multi_call), multi_call
        )
    expected_ids = resolve_perforate_ids(model, target)
    configure_perforated_ai(model, target, train_config)
    logger.info("PerforatedAI perforating %d modules for target %s", len(expected_ids), target.name)
    save_name = build_perforated_save_name(model_config, train_config)
    model = upa.perforate_model(model, save_name=save_name, maximizing_score=True)
    verify_perforated_wrapping(model, expected_ids)
    apply_output_dimensions(model, ranks)

    if train_config.pai_load_folder is not None:
        model = upa.load_system(model, str(train_config.pai_load_folder), train_config.pai_load_stage, switch_call=True)
        logger.info("PerforatedAI loaded system %s from %s", train_config.pai_load_stage, train_config.pai_load_folder)
    return model


def before_first_switch(train_config: TrainConfig) -> bool:
    """Return whether the run is in the neuron epoch that a forced first switch follows.

    Args:
        train_config: Training configuration with ``perforate=True``.

    Returns:
        ``True`` only when ``pai_force_first_switch`` is set and PAI has not switched yet.
    """
    gpa, _ = import_perforatedai()
    return bool(
        train_config.pai_force_first_switch
        and gpa.pai_tracker.member_vars["mode"] == "n"
        and len(gpa.pai_tracker.member_vars["switch_epochs"]) == 0
    )


# --- Optimizer --------------------------------------------------------------------------------------------------------
# PAI's learning rate search only runs when PAI built the scheduler, so the optimizer comes from
# pai_tracker.setup_optimizer and Lightning gets no scheduler. Patience 14 gives two factor-0.1 drops inside the 30
# validations PAI waits before adding a dendrite; the threshold matches PAI's 0.1 percent improvement bar so the two
# plateau counters agree; the floor is two drops below lr, which the search needs to see its second step.
_PLATEAU_PATIENCE = 14
_PLATEAU_FACTOR = 0.1
_PLATEAU_THRESHOLD = 0.001
_LR_FLOOR_RATIO = 0.01


def collect_dendrite_param_ids(model: nn.Module) -> set[int]:
    """Return the ``id()`` of every parameter PAI added beside a ``main_module``.

    Args:
        model: Perforated ``LWDETR``.

    Returns:
        Python ids of the dendrite parameters.
    """
    ids: set[int] = set()
    for module in model.modules():
        if type(module).__name__ != "PAINeuronModule":
            continue
        main = {id(p) for p in module.get_submodule("main_module").parameters()}
        ids |= {id(p) for p in module.parameters() if id(p) not in main}
    return ids


def build_perforated_param_groups(pl_module: RFDETRModelModule) -> list[dict[str, Any]]:
    """Build RF-DETR's param groups for the current PAI mode.

    Dendrite parameters are split into their own groups with ``weight_decay=0.0``. In a dendrite cycle only the
    parameters of the ``PAINeuronModule`` s are handed over, which keeps the rest of the model frozen.

    Args:
        pl_module: Lightning module holding the perforated ``LWDETR``.

    Returns:
        Param groups for the optimizer constructor.
    """
    gpa, upa = import_perforatedai()
    namespace = _namespace_from_configs(pl_module.model_config, pl_module.train_config)
    groups = get_param_dict(namespace, pl_module.model)

    dendrite_ids = collect_dendrite_param_ids(pl_module.model)
    split: list[dict[str, Any]] = []
    for group in groups:
        params = group["params"]
        model_params = [p for p in params if id(p) not in dendrite_ids]
        dendrite_params = [p for p in params if id(p) in dendrite_ids]
        if model_params:
            split.append({**group, "params": model_params})
        if dendrite_params:
            split.append({**group, "params": dendrite_params, "weight_decay": 0.0})

    if gpa.pai_tracker.member_vars["mode"] != "p":
        return split

    allowed = {id(p) for p in upa.get_pai_network_params(pl_module.model)}
    narrowed: list[dict[str, Any]] = []
    for group in split:
        params = [p for p in group["params"] if id(p) in allowed]
        if params:
            narrowed.append({**group, "params": params})
    dendrite_lr = pl_module.train_config.pai_dendrite_lr
    if dendrite_lr is not None:
        for group in narrowed:
            group["lr"] = dendrite_lr
    return narrowed


def stash_optimizer_state(model: nn.Module, optimizer: optim.Optimizer) -> dict[str, dict[str, Any]]:
    """Keep the AdamW state of every held parameter by parameter name (references, nothing cloned).

    Args:
        model: Perforated ``LWDETR`` the optimizer was built on.
        optimizer: Optimizer whose state is kept.

    Returns:
        ``{parameter_name: optimizer_state}``.
    """
    names = {id(p): name for name, p in model.named_parameters()}
    return {
        names[id(p)]: optimizer.state[p]
        for group in optimizer.param_groups
        for p in group["params"]
        if p in optimizer.state and id(p) in names
    }


def restore_optimizer_state(model: nn.Module, optimizer: optim.Optimizer, stash: dict[str, dict[str, Any]]) -> int:
    """Put stashed AdamW state back on parameters that still exist under the same name and shape.

    Args:
        model: Restructured ``LWDETR`` the new optimizer was built on.
        optimizer: Freshly built optimizer.
        stash: State returned by :func:`stash_optimizer_state`.

    Returns:
        Number of parameters whose state was restored.
    """
    params = dict(model.named_parameters())
    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    hit = 0
    for name, state in stash.items():
        param = params.get(name)
        if param is None or id(param) not in held:
            continue
        moment = state.get("exp_avg")
        if moment is not None and moment.shape != param.shape:
            continue
        optimizer.state[param] = state
        hit += 1
    return hit


def setup_perforated_optimizer(pl_module: RFDETRModelModule) -> optim.Optimizer:
    """Build AdamW and the plateau schedule through PAI.

    Every call starts with empty optimizer state and a fresh plateau history; the callback restores the stashed AdamW
    moments afterwards.

    Args:
        pl_module: Lightning module holding the perforated ``LWDETR``.

    Returns:
        The optimizer PAI built.
    """
    gpa, _ = import_perforatedai()
    tc = pl_module.train_config
    groups = build_perforated_param_groups(pl_module)
    on_cuda = next(pl_module.model.parameters()).is_cuda

    # lr 0 keeps the loaded weights untouched on the epoch before a forced switch.
    frozen_epoch = before_first_switch(tc)
    if frozen_epoch:
        for group in groups:
            group["lr"] = 0.0
        logger.info("PerforatedAI first neuron epoch runs at lr 0 ahead of the forced switch to dendrite training")

    gpa.pai_tracker.set_optimizer(optim.AdamW)
    gpa.pai_tracker.set_scheduler(optim.lr_scheduler.ReduceLROnPlateau)
    optimizer, _ = gpa.pai_tracker.setup_optimizer(
        pl_module.model,
        {"params": groups, "lr": 0.0 if frozen_epoch else tc.lr, "weight_decay": tc.weight_decay, "fused": on_cuda},
        {
            "mode": "max",
            "factor": _PLATEAU_FACTOR,
            "patience": _PLATEAU_PATIENCE,
            "threshold": _PLATEAU_THRESHOLD,
            "threshold_mode": "rel",
            "min_lr": tc.lr * _LR_FLOOR_RATIO,
            "cooldown": 0,
        },
    )
    # PAI's narrowed groups can drop initial_lr, which Lightning's LR logging reads.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    lrs = sorted({group["lr"] for group in optimizer.param_groups})
    logger.info(
        "PerforatedAI built AdamW in mode %s with %d param groups at lr %s",
        gpa.pai_tracker.member_vars["mode"],
        len(optimizer.param_groups),
        lrs,
    )
    assert isinstance(optimizer, optim.Optimizer)
    return optimizer


# --- EMA bookkeeping --------------------------------------------------------------------------------------------------
# AveragedModel pairs tensors by position, but PAI registers buffers lazily and resizes some at every validation, so
# the EMA update is replaced by one that pairs by name. PAI's own buffers (integer indices, tracker_string) are copied
# rather than averaged: averaging an index through float rounding once broadcast the dendrite weights on the wrong axis.


def find_ema_callback(trainer: Trainer) -> RFDETREMACallback | None:
    """Return the EMA callback of the trainer, or ``None`` when ``use_ema=False``."""
    for callback in trainer.callbacks:  # type: ignore[attr-defined]
        if isinstance(callback, RFDETREMACallback):
            return callback
    return None


def ema_weight_keys(model: nn.Module) -> set[str]:
    """State dict keys the EMA averages and validation scores: parameters plus norm running stats.

    Args:
        model: Live perforated ``LWDETR``.

    Returns:
        Parameter names plus ``running_mean`` / ``running_var`` buffer names.
    """
    keys = {name for name, _ in model.named_parameters()}
    keys |= {name for name, _ in model.named_buffers() if name.endswith(("running_mean", "running_var"))}
    return keys


def pai_buffer_names(model: nn.Module) -> set[str]:
    """State dict keys of the buffers under every ``PAINeuronModule``.

    Args:
        model: Live perforated ``LWDETR``.

    Returns:
        Buffer names PAI keeps for bookkeeping.
    """
    prefixes = [name + "." for name, module in model.named_modules() if type(module).__name__ == "PAINeuronModule"]
    return {name for name, _ in model.named_buffers() if any(name.startswith(p) for p in prefixes)}


def replace_buffer(root: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Re-register a nested buffer with a copy of ``tensor``, for buffers whose shape changed.

    Args:
        root: Module the dotted name is relative to.
        name: Dotted state dict name of the buffer.
        tensor: Tensor whose copy becomes the buffer.
    """
    owner = root
    parts = name.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    owner.register_buffer(parts[-1], tensor.detach().clone())


@torch.no_grad()
def sync_resized_buffers(pl_module: LightningModule, ema_cb: RFDETREMACallback) -> int:
    """Replace every integer buffer of the EMA copy whose live shape changed since the last step.

    Args:
        pl_module: Module holding the live perforated ``LWDETR`` at ``.model``.
        ema_cb: EMA callback holding the averaged copy.

    Returns:
        Number of buffers replaced.
    """
    average_model = getattr(ema_cb, "_average_model", None)
    if average_model is None:
        return 0
    ema_buffers = dict(average_model.module.named_buffers())
    replaced = 0
    for name, live_t in pl_module.named_buffers():
        ema_t = ema_buffers.get(name)
        if ema_t is None or ema_t.is_floating_point():
            continue
        if ema_t.shape != live_t.shape:
            replace_buffer(average_model.module, name, live_t)
            replaced += 1
    return replaced


def install_named_ema_update(ema_cb: RFDETREMACallback, pl_module: RFDETRModelModule) -> None:
    """Replace ``AveragedModel.update_parameters`` with a by-name update that mirrors PAI's buffers.

    Args:
        ema_cb: EMA callback holding the averaged copy.
        pl_module: Module holding the live perforated ``LWDETR`` at ``.model``.
    """
    average_model = getattr(ema_cb, "_average_model", None)
    if average_model is None:
        return
    skip = {"model." + name for name in pai_buffer_names(pl_module.model)}

    @torch.no_grad()
    def update_parameters(model: LightningModule) -> None:
        n_averaged = int(average_model.n_averaged.item())
        decay = ema_cb._effective_decay(n_averaged)
        ema_tensors: dict[str, torch.Tensor] = dict(average_model.module.named_parameters())
        ema_tensors.update(average_model.module.named_buffers())
        live_tensors: dict[str, torch.Tensor] = dict(model.named_parameters())
        live_tensors.update(model.named_buffers())
        groups: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for name, ema_t in ema_tensors.items():
            live_t = live_tensors.get(name)
            if live_t is None:
                continue
            if live_t.shape != ema_t.shape:
                if not ema_t.is_floating_point():
                    replace_buffer(average_model.module, name, live_t)
                continue
            mirror = name in skip or n_averaged == 0 or not ema_t.is_floating_point()
            if mirror:
                ema_t.copy_(live_t)
                continue
            key = (ema_t.device, ema_t.dtype)
            averaged, current = groups.setdefault(key, ([], []))
            averaged.append(ema_t)
            current.append(live_t.to(ema_t.dtype))
        for averaged, current in groups.values():
            torch._foreach_mul_(averaged, decay)
            torch._foreach_add_(averaged, current, alpha=1.0 - decay)
        average_model.n_averaged += 1

    average_model.update_parameters = update_parameters


def rebuild_ema(ema_cb: RFDETREMACallback, pl_module: RFDETRModelModule) -> None:
    """Rebuild the EMA copy from the restructured module, keeping ``n_averaged``.

    Args:
        ema_cb: EMA callback to rebuild.
        pl_module: Module holding the restructured ``LWDETR`` at ``.model``.
    """
    gpa, _ = import_perforatedai()
    old = getattr(ema_cb, "_average_model", None)
    n_averaged = int(old.n_averaged.item()) if old is not None else 0
    # PAI processor state does not survive the deepcopy inside AveragedModel.
    gpa.pai_tracker.clear_all_processors()
    new = AveragedModel(model=pl_module, device=pl_module.device, use_buffers=True, multi_avg_fn=ema_cb._multi_avg_fn)
    new.n_averaged.fill_(n_averaged)
    new.eval()
    ema_cb._average_model = new
    install_named_ema_update(ema_cb, pl_module)


def check_ema_buffers(pl_module: RFDETRModelModule, ema_cb: RFDETREMACallback) -> None:
    """Rebuild the EMA copy once PAI has registered its lazy buffers, so both hold the same module set.

    Args:
        pl_module: Module holding the live perforated ``LWDETR`` at ``.model``.
        ema_cb: EMA callback holding the averaged copy.
    """
    ema_inner = _get_ema_inner_module(ema_cb)
    if ema_inner is None:
        return
    live = {name for name, _ in pl_module.model.named_buffers()}
    ema = {name for name, _ in ema_inner.model.named_buffers()}
    if live == ema:
        return
    rebuild_ema(ema_cb, pl_module)
    logger.info("PerforatedAI rebuilt the EMA copy: live model has %d buffers, copy had %d", len(live), len(ema))


@torch.no_grad()
def swap_in_ema_weights(pl_module: RFDETRModelModule, ema_cb: RFDETREMACallback) -> dict[str, torch.Tensor]:
    """Copy the EMA weights into the live detector in place, so PAI checkpoints the weights that were scored.

    Args:
        pl_module: Module holding the live perforated ``LWDETR`` at ``.model``.
        ema_cb: EMA callback holding the averaged copy.

    Returns:
        The live weights that were overwritten, to copy back when PAI does not restructure.
    """
    ema_inner = _get_ema_inner_module(ema_cb)
    if ema_inner is None:
        return {}
    live = pl_module.model.state_dict()
    ema = ema_inner.model.state_dict()
    saved: dict[str, torch.Tensor] = {}
    for key in ema_weight_keys(pl_module.model):
        saved[key] = live[key].clone()
        live[key].copy_(ema[key])
    return saved


# --- Dendrite scores --------------------------------------------------------------------------------------------------


def current_dendrite_score(fallback: float) -> float:
    """Mean best dendrite correlation, the score PAI gets in dendrite mode where mAP is flat.

    Args:
        fallback: Score to use when ``perforatedbp`` exposes no correlations.

    Returns:
        The score to hand PAI in dendrite mode.
    """
    gpa, _ = import_perforatedai()
    getter = getattr(gpa.pai_tracker, "get_current_pb_scores", None)
    if getter is None:
        return fallback
    scores = list(getter().values())
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def dendrite_correlations() -> dict[str, float]:
    """Best dendrite correlation so far of every perforated layer, by name (empty outside dendrite training).

    Returns:
        ``{layer_name: best_correlation}``.
    """
    gpa, _ = import_perforatedai()
    names = [module.name for module in gpa.pai_tracker.neuron_module_vector]
    scores = gpa.pai_tracker.member_vars.get("best_scores", [])
    return {name: float(scores[i][-1]) for i, name in enumerate(names) if i < len(scores) and len(scores[i]) > 0}


# --- Callback ---------------------------------------------------------------------------------------------------------

#: Detector with the PAI tracker removed, written to ``output_dir`` at the end of training.
CLEAN_MODEL_NAME = "final_clean_model.pth"

#: Record handed to an observer after every scored validation.
ObserverRecord = dict[str, Any]


class PerforatedAICallback(Callback):
    """Feed the validation metric to PerforatedAI and act on what it decides.

    Runs after ``COCOEvalCallback`` and ``BestModelCallback`` on ``on_validation_end``. After every validation the EMA
    weights are swapped in, PAI is handed the score, and then either training stops, the live weights go back, or the
    optimizer and EMA copy are rebuilt around the restructured model with the AdamW moments restored by name.

    Args:
        monitor: ``callback_metrics`` key PAI scores neuron phases on.
        observer: Called after every scored validation with the epoch, mode, dendrite count, ``val/`` metrics,
            dendrite correlations, and PAI's decision. The sweep driver logs its own stream from it.
    """

    def __init__(
        self,
        monitor: str = "val/mAP_50_95",
        observer: Callable[[ObserverRecord], None] | None = None,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.observer = observer
        self._moment_stash: dict[str, dict[str, Any]] = {}

    def set_observer(self, observer: Callable[[ObserverRecord], None] | None) -> None:
        """Replace the observer after construction."""
        self.observer = observer

    @staticmethod
    def _module_of(pl_module: LightningModule) -> RFDETRModelModule:
        """Return the module as an ``RFDETRModelModule``, failing if it was not perforated."""
        if not getattr(getattr(pl_module, "train_config", None), "perforate", False):
            raise RuntimeError("PerforatedAICallback needs a module built with perforate=True.")
        return cast("RFDETRModelModule", pl_module)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Install the by-name EMA update on the copy ``RFDETREMACallback`` just built."""
        module = self._module_of(pl_module)
        ema_cb = find_ema_callback(trainer)
        if ema_cb is not None:
            install_named_ema_update(ema_cb, module)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Resolve the auto correlation warmup and check it fits in one epoch (PAI halts otherwise)."""
        gpa, _ = import_perforatedai()
        module = self._module_of(pl_module)
        tc = module.train_config
        iters_per_epoch = trainer.num_training_batches
        if tc.pai_initial_correlation_batches == 0:
            if math.isinf(iters_per_epoch):
                raise ValueError(
                    "pai_initial_correlation_batches=0 (auto) needs a sized train dataloader. Set it explicitly."
                )
            warmup = max(1, int(tc.pai_initial_correlation_fraction * iters_per_epoch))
            gpa.pc.set_initial_correlation_batches(warmup)
            logger.info("PerforatedAI correlation warmup set to %d of %d iterations per epoch", warmup, iters_per_epoch)
        warmup = gpa.pc.get_initial_correlation_batches()
        if math.isinf(iters_per_epoch) or warmup < iters_per_epoch:
            return
        raise ValueError(
            f"pai_initial_correlation_batches ({warmup}) must be smaller than the {iters_per_epoch} iterations in one "
            "training epoch, or PAI halts once dendrites start their correlation warmup. Lower it, or lower "
            "batch_size to lengthen the epoch."
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Rebuild the EMA copy once PAI's lazy buffers exist."""
        ema_cb = find_ema_callback(trainer)
        if ema_cb is not None:
            check_ema_buffers(cast("RFDETRModelModule", pl_module), ema_cb)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Report the epoch mean training loss to PAI."""
        gpa, _ = import_perforatedai()
        loss = trainer.callback_metrics.get("train/loss")
        if loss is not None:
            gpa.pai_tracker.add_extra_score_without_graphing(float(loss), "train_loss")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Hand PAI the validation score and rebuild the optimizer and EMA after a restructure."""
        if trainer.sanity_checking:
            return
        gpa, _ = import_perforatedai()
        module = self._module_of(pl_module)
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            raise RuntimeError(
                f"{self.monitor} was not found in callback_metrics, so PAI has no score to act on. Keep "
                "eval_interval at 1 and the COCO eval callback enabled."
            )
        val_metrics = {
            key: float(value)
            for key, value in trainer.callback_metrics.items()
            if key.startswith("val/") and value.numel() == 1
        }
        for key, value in val_metrics.items():
            gpa.pai_tracker.add_extra_score_without_graphing(value, key)

        score = float(metric)
        mode = gpa.pai_tracker.member_vars["mode"]
        added = gpa.pai_tracker.member_vars["num_dendrites_added"]
        if mode == "p":
            score = current_dendrite_score(score)
        if mode == "n":
            self._moment_stash = stash_optimizer_state(module.model, trainer.optimizers[0])

        force = before_first_switch(module.train_config)
        if force:
            logger.info(
                "PerforatedAI forcing the switch to dendrite training at epoch %d with score %.4f",
                trainer.current_epoch,
                score,
            )

        ema_cb = find_ema_callback(trainer)
        saved = swap_in_ema_weights(module, ema_cb) if ema_cb is not None else {}
        device = module.device
        model, restructured, training_complete = gpa.pai_tracker.add_validation_score(
            score, module.model, force_switch=force
        )

        if self.observer is not None:
            self.observer(
                {
                    "epoch": trainer.current_epoch,
                    "mode": mode,
                    "num_dendrites": added,
                    "metrics": val_metrics,
                    "correlations": dendrite_correlations(),
                    "restructured": bool(restructured),
                    "training_complete": bool(training_complete),
                }
            )

        if training_complete:
            logger.info("PerforatedAI reported training complete, stopping.")
            module.model = model.to(device)
            if ema_cb is not None:
                rebuild_ema(ema_cb, module)
            trainer.should_stop = True
            return
        if not restructured:
            live = module.model.state_dict()
            for key, tensor in saved.items():
                live[key].copy_(tensor)
            if ema_cb is not None:
                sync_resized_buffers(module, ema_cb)
            return

        module.model = model.to(device)
        optimizer = setup_perforated_optimizer(module)
        # The strategy rebuilds its LightningOptimizer wrappers from this list; PAI owns the only schedule.
        trainer.optimizers = [optimizer]
        trainer.strategy.lr_scheduler_configs = []
        restored = 0
        if gpa.pai_tracker.member_vars["mode"] == "n" and self._moment_stash:
            restored = restore_optimizer_state(module.model, optimizer, self._moment_stash)
            self._moment_stash = {}
        if ema_cb is not None:
            rebuild_ema(ema_cb, module)
        params = sum(p.numel() for p in module.model.parameters())
        logger.info(
            "PerforatedAI restructured at epoch %d into mode %s with %d parameters. Rebuilt the optimizer with AdamW "
            "state restored on %d tensors, a fresh plateau schedule, and the EMA.",
            trainer.current_epoch,
            gpa.pai_tracker.member_vars["mode"],
            params,
            restored,
        )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write the EMA detector without PAI scaffolding next to the checkpoints.

        The file carries the cleaned ``LWDETR`` under ``module`` because no constructor rebuilds its dendrite heads
        from a state dict: ``torch.load(path, weights_only=False)["module"]`` is the reload.
        """
        if not trainer.is_global_zero:
            return
        module = cast("RFDETRModelModule", pl_module)
        clean = clean_perforated_model(module.model)
        path = Path(module.train_config.output_dir) / CLEAN_MODEL_NAME
        torch.save(
            {
                "module": clean,
                "model": clean.state_dict(),
                "model_config": module.model_config.model_dump(mode="json"),
                "epoch": trainer.current_epoch,
            },
            path,
        )
        logger.info("PerforatedAI wrote the clean detector to %s", path)


# --- Checkpoints ------------------------------------------------------------------------------------------------------

#: State dict keys PAI writes beside the model, dropped by :func:`extract_plain_state`.
_PAI_TOP_LEVEL_KEYS = ("tracker_string",)
#: Segment PAI nests a wrapped module's own tensors under.
_MAIN_MODULE_SEGMENT = ".main_module."


def clean_perforated_model(model: nn.Module) -> nn.Module:
    """Return a deep copy of the detector with the PAI tracker removed and the dendrites kept.

    The copy runs in any process with ``perforatedai`` installed, which is what ``predict`` and ``export`` need. It is
    not a plain ``LWDETR``: the perforated heads keep PAI's clean module class.

    Args:
        model: Live perforated ``LWDETR``.

    Returns:
        The cleaned ``LWDETR`` copy.
    """
    _, upa = import_perforatedai()
    cleaned: nn.Module = upa.prepare_final_model(model)
    return cleaned


def extract_plain_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip PAI's wrapping, bookkeeping, and dendrites from a saved state dict.

    Args:
        state: State dict read from a PAI stage file (``<folder>/<stage>.pt``, safetensors).

    Returns:
        A state dict that loads into a plain ``LWDETR``.
    """
    prefixes = sorted({key.split(_MAIN_MODULE_SEGMENT)[0] for key in state if _MAIN_MODULE_SEGMENT in key})
    plain: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key in _PAI_TOP_LEVEL_KEYS:
            continue
        owner = next((p for p in prefixes if key == p or key.startswith(p + ".")), None)
        if owner is None:
            plain[key] = tensor
            continue
        marker = owner + _MAIN_MODULE_SEGMENT
        if key.startswith(marker):
            plain[owner + "." + key[len(marker) :]] = tensor
    return plain


def extract_start_weights(
    stage_path: str | Path,
    out_path: str | Path | None = None,
    model_args: dict[str, Any] | None = None,
) -> Path:
    """Write a plain RF-DETR checkpoint from a saved PAI stage, loadable through ``pretrain_weights``.

    Args:
        stage_path: PAI stage file, ``<system folder>/<stage>.pt``.
        out_path: Checkpoint to write. ``None`` writes ``<stage>_plain.pth`` beside the stage file.
        model_args: Stored under the checkpoint's ``args`` key, typically a ``training_config.json`` ``model_config``.

    Returns:
        Path of the written checkpoint.

    Raises:
        FileNotFoundError: If ``stage_path`` does not exist.
    """
    # Optional-dependency boundary: safetensors arrives through the transformers stack.
    from safetensors.torch import load_file

    stage = Path(stage_path)
    if not stage.is_file():
        raise FileNotFoundError(f"No PAI stage file at {stage}")
    out = Path(out_path) if out_path is not None else stage.with_name(f"{stage.stem}_plain.pth")
    state = load_file(str(stage))
    plain = extract_plain_state(state)
    logger.info("Extracted %d of %d tensors from %s into %s", len(plain), len(state), stage, out)
    torch.save({"model": plain, "args": dict(model_args or {})}, out)
    return out
