# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Perforated Integration to Roboflow"""

from __future__ import annotations

import json
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
from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersLayer
from rfdetr.models.heads.segmentation import DepthwiseConvBlock, MLPBlock
from rfdetr.models.transformer import Transformer, TransformerDecoderLayer
from rfdetr.training.callbacks import RFDETREMACallback
from rfdetr.training.callbacks.coco_eval import _get_ema_inner_module
from rfdetr.training.param_groups import get_param_dict
from rfdetr.training.perforated_ema import (
    ema_update_by_name,
    ema_weight_keys,
    pai_buffer_names,
    restore_weights,
    swap_in_weights,
    sync_resized_buffers,
)
from rfdetr.training.perforated_restructure import (
    PostNormResidual,
    PreNormBranch,
    Recipe,
    Restructured,
    invert_key_map,
    layer_with_norm,
    remap_state_dict,
    restructure_model,
)
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.training.module_model import RFDETRModelModule

logger = get_logger()

__all__ = [
    "CLEAN_MODEL_NAME",
    "KEY_MAP_NAME",
    "PERFORATION_TARGETS",
    "RECIPES",
    "PerforatedAICallback",
    "PerforationTarget",
    "build_perforated_save_name",
    "clean_perforated_model",
    "configure_dendrite_copy",
    "extract_plain_state",
    "extract_start_weights",
    "get_perforation_target",
    "install_dendrite_init_hook",
    "list_perforable_modules",
    "perforate_detection_model",
    "resolve_perforate_ids",
    "restructure_detection_model",
    "setup_perforated_optimizer",
]

# --- Sub-blocks -------------------------------------------------------------------------------------------------------
# PerforatedAI perforates one module at a time and wants every normalization layer inside the module that receives
# dendrites. RF-DETR's blocks call their norms, branches, and residual adds as siblings in one forward, so these recipes
# rebuild each block from its existing children into sub-block modules. Pre-norm branches (DINOv2, segmentation) leave
# the residual add in the parent block, so a dendrite is the branch alone. Post-norm blocks (decoder) keep the residual
# and the norm inside and drop the skip in their dendrite copies (see configure_dendrite_copy).


class PerforableDinoLayer(nn.Module):
    """A DINOv2 block rebuilt as two pre-norm branches with the residual adds and window reshapes outside.

    Args:
        layer: The ``WindowedDinov2WithRegistersLayer`` whose children are reused.
    """

    def __init__(self, layer: WindowedDinov2WithRegistersLayer) -> None:
        super().__init__()
        self.num_windows = layer.num_windows
        self.attention = PreNormBranch(layer.norm1, layer.attention, layer.layer_scale1, select=0)
        self.drop_path = layer.drop_path
        self.mlp = PreNormBranch(layer.norm2, layer.mlp, layer.layer_scale2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ) -> tuple[torch.Tensor]:
        """Run the block; same contract as the original layer.

        Args:
            hidden_states: Windowed tokens, ``[B * num_windows**2, T, C]``.
            output_attentions: Unsupported, as in the original layer.
            run_full_attention: Merge the windows for this layer's attention.

        Returns:
            A one-tuple with the block output.
        """
        assert not output_attentions, "output_attentions is not supported for windowed attention"
        shortcut = hidden_states
        if run_full_attention:
            batch_windows, tokens_per_window, channels = hidden_states.shape
            num_windows_squared = self.num_windows**2
            hidden_states = hidden_states.view(
                batch_windows // num_windows_squared, num_windows_squared * tokens_per_window, channels
            )
        attention_output = self.attention(hidden_states, output_attentions=output_attentions)
        if run_full_attention:
            # Layer scale is per channel, so applying it before this view matches the original order.
            attention_output = attention_output.view_as(shortcut)
        hidden_states = self.drop_path(attention_output) + shortcut
        layer_output = self.drop_path(self.mlp(hidden_states)) + hidden_states
        return (layer_output,)


class DecoderSelfAttention(nn.Module):
    """The decoder layer's self-attention body, with the group-DETR query split of training mode.

    Args:
        self_attn: The layer's ``nn.MultiheadAttention``.
        group_detr: Number of query groups attended separately while training.
    """

    def __init__(self, self_attn: nn.MultiheadAttention, group_detr: int) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.group_detr = group_detr

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend the queries to each other.

        Args:
            tgt: Queries, ``[B, Q, C]``.
            query_pos: Positional embedding added to queries and keys.
            tgt_mask: Attention mask.
            tgt_key_padding_mask: Key padding mask.

        Returns:
            Attention output, ``[B, Q, C]``.
        """
        bs, num_queries, _ = tgt.shape
        q = k = tgt if query_pos is None else tgt + query_pos
        v = tgt
        if self.training:
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)  # type: ignore[no-untyped-call]
            k = q
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)  # type: ignore[no-untyped-call]
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        return cast(torch.Tensor, tgt2)


class DecoderCrossAttention(nn.Module):
    """The decoder layer's deformable cross-attention body.

    Args:
        cross_attn: The layer's ``MSDeformAttn``.
    """

    def __init__(self, cross_attn: nn.Module) -> None:
        super().__init__()
        self.cross_attn = cross_attn

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor | None,
        reference_points: torch.Tensor | None,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor | None,
        level_start_index: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
        spatial_shapes_hw: list[tuple[int, int]] | None,
    ) -> torch.Tensor:
        """Attend the queries to the encoder memory.

        Args:
            tgt: Queries, ``[B, Q, C]``.
            query_pos: Positional embedding added to the queries.
            reference_points: Sampling reference points.
            memory: Encoder memory.
            spatial_shapes: Feature level shapes.
            level_start_index: Start index of every level in ``memory``.
            memory_key_padding_mask: Memory padding mask.
            spatial_shapes_hw: Feature level shapes as Python ints.

        Returns:
            Attention output, ``[B, Q, C]``.
        """
        query = tgt if query_pos is None else tgt + query_pos
        out = self.cross_attn(
            query,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_spatial_shapes_hw=spatial_shapes_hw,
        )
        return cast(torch.Tensor, out)


class DecoderFeedForward(nn.Module):
    """The decoder layer's FFN body ``linear2(dropout(activation(linear1(x))))``.

    Args:
        linear1: First projection.
        activation: Activation function between the projections.
        dropout: Dropout after the activation.
        linear2: Second projection.
    """

    def __init__(
        self,
        linear1: nn.Linear,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout: nn.Module,
        linear2: nn.Linear,
    ) -> None:
        super().__init__()
        self.linear1 = linear1
        self.activation = activation
        self.dropout = dropout
        self.linear2 = linear2

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """Apply the FFN to ``[B, Q, C]`` queries."""
        # The original layer computes exactly this before its third residual add.
        return self.linear2(self.dropout(self.activation(self.linear1(tgt))))


class PerforableDecoderLayer(nn.Module):
    """A decoder layer rebuilt as three post-norm residual blocks.

    Args:
        layer: The ``TransformerDecoderLayer`` whose children are reused.

    Raises:
        ValueError: If the layer runs the keypoint subnetwork, which ``perforate=True`` does not support.
    """

    def __init__(self, layer: TransformerDecoderLayer) -> None:
        super().__init__()
        if layer.enable_keypoint_processing:
            raise ValueError("Block perforation does not support the keypoint decoder layer.")
        self.self_attn = PostNormResidual(
            DecoderSelfAttention(layer.self_attn, layer.group_detr), layer.dropout1, layer.norm1
        )
        self.cross_attn = PostNormResidual(DecoderCrossAttention(layer.cross_attn), layer.dropout2, layer.norm2)
        self.ffn = PostNormResidual(
            DecoderFeedForward(layer.linear1, layer.activation, layer.dropout, layer.linear2),
            layer.dropout3,
            layer.norm3,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
        keypoint_tgt: torch.Tensor | None = None,
        keypoint_pos: torch.Tensor | None = None,
        keypoint_class_mask: torch.Tensor | None = None,
        kp_cross_attn_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the layer; same contract as ``TransformerDecoderLayer.forward`` without keypoints.

        Args:
            tgt: Queries, ``[B, Q, C]``.
            memory: Encoder memory.
            tgt_mask: Self-attention mask.
            memory_mask: Unused, kept for the call signature.
            tgt_key_padding_mask: Self-attention key padding mask.
            memory_key_padding_mask: Memory padding mask.
            query_pos: Query positional embedding.
            reference_points: Deformable attention reference points.
            spatial_shapes: Feature level shapes.
            spatial_shapes_hw: Feature level shapes as Python ints.
            level_start_index: Start index of every level in ``memory``.
            keypoint_tgt: Must be ``None``.
            keypoint_pos: Must be ``None``.
            keypoint_class_mask: Must be ``None``.
            kp_cross_attn_memory: Must be ``None``.

        Returns:
            Updated queries, ``[B, Q, C]``.
        """
        if any(v is not None for v in (keypoint_tgt, keypoint_pos, keypoint_class_mask, kp_cross_attn_memory)):
            raise ValueError("Block perforation does not support the keypoint decoder layer.")
        tgt = self.self_attn(tgt, query_pos, tgt_mask, tgt_key_padding_mask)
        tgt = self.cross_attn(
            tgt,
            query_pos,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            spatial_shapes_hw,
        )
        return cast(torch.Tensor, self.ffn(tgt))


def restructure_two_stage(transformer: Transformer) -> Transformer:
    """Group every two-stage ``enc_output[i]`` Linear with its ``enc_output_norm[i]`` in place.

    ``enc_output_norm[i]`` becomes an identity so the transformer's forward is unchanged. The stacked fast path checks
    for plain ``nn.Linear`` / ``nn.LayerNorm`` and falls back to the per-group loop on its own.

    Args:
        transformer: The detector's ``Transformer``.

    Returns:
        The same transformer.
    """
    enc_output = getattr(transformer, "enc_output", None)
    enc_output_norm = getattr(transformer, "enc_output_norm", None)
    if enc_output is None or enc_output_norm is None:
        return transformer
    for index in range(len(enc_output)):
        enc_output[index] = layer_with_norm(enc_output[index], enc_output_norm[index])
        enc_output_norm[index] = nn.Identity()
    return transformer


class ParameterScale(nn.Module):
    """Multiply by a per-channel parameter, so a bare ``gamma`` parameter can sit in a branch's ``scale`` slot.

    Args:
        weight: The existing parameter, reused as is.
    """

    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` along its last axis."""
        # Same broadcast as the ``gamma * x`` the segmentation blocks compute.
        return x * self.weight


class DepthwiseBranch(DepthwiseConvBlock):
    """The segmentation ``DepthwiseConvBlock`` without its residual add.

    Subclassing keeps the block's cuDNN-free depthwise convolution.

    Args:
        block: The block whose children are reused.
    """

    neuron_axis: int = 1

    def __init__(self, block: DepthwiseConvBlock) -> None:
        nn.Module.__init__(self)
        self.dwconv = block.dwconv
        self.norm = block.norm
        self.pwconv1 = block.pwconv1
        self.act = block.act
        self.gamma = block.gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the branch to ``[B, C, H, W]`` features and return ``[B, C, H, W]``."""
        x = self._depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.pwconv1(self.norm(x)))
        if self.gamma is not None:
            x = self.gamma * x
        return x.permute(0, 3, 1, 2)


class PerforableDepthwiseConvBlock(nn.Module):
    """A segmentation ``DepthwiseConvBlock`` rebuilt as a branch plus an outside residual add.

    Args:
        block: The block whose children are reused.
    """

    def __init__(self, block: DepthwiseConvBlock) -> None:
        super().__init__()
        self.branch = DepthwiseBranch(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block on ``[B, C, H, W]`` features."""
        # The residual add stays out here so a dendrite of the branch carries no skip connection.
        return x + self.branch(x)


class SegmentationMLPBranch(PreNormBranch):
    """The segmentation ``MLPBlock`` branch; a distinct type so targets can name it apart from the backbone's."""


class PerforableMLPBlock(nn.Module):
    """A segmentation ``MLPBlock`` rebuilt as a pre-norm branch plus an outside residual add.

    Args:
        block: The block whose children are reused.
    """

    def __init__(self, block: MLPBlock) -> None:
        super().__init__()
        scale = ParameterScale(block.gamma) if block.gamma is not None else None
        self.branch = SegmentationMLPBranch(block.norm_in, nn.Sequential(*block.layers), scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block on ``[B, N, C]`` query features."""
        # The residual add stays out here so a dendrite of the branch carries no skip connection.
        return x + self.branch(x)


#: Exact module type to the recipe that rebuilds it, applied by :func:`restructure_detection_model`.
RECIPES: dict[type[nn.Module], Recipe] = {
    WindowedDinov2WithRegistersLayer: PerforableDinoLayer,
    TransformerDecoderLayer: PerforableDecoderLayer,
    Transformer: restructure_two_stage,
    DepthwiseConvBlock: PerforableDepthwiseConvBlock,
    MLPBlock: PerforableMLPBlock,
}

#: File in ``output_dir`` mapping original state-dict keys to the restructured ones.
KEY_MAP_NAME = "pai_key_map.json"


def restructure_detection_model(model: nn.Module) -> Restructured:
    """Apply :data:`RECIPES` to a detector.

    Args:
        model: ``LWDETR`` with pretrained weights already loaded.

    Returns:
        The restructured model with its key map.
    """
    # A thin name for the detector so callers do not pass the recipe table around.
    return restructure_model(model, RECIPES)


# --- Targets ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PerforationTarget:
    """Detector modules that receive dendrites.

    The neuron axis of every wrapped module is measured by :func:`record_forward_stats`, so a target only names modules.

    Args:
        name: Short name, used in the PAI save folder name.
        perforate_ids: Module ids (``named_modules`` names with a leading dot) that receive dendrites.
        perforate_type_names: Module type names that receive dendrites wherever they occur.
        skip_ids: Module ids PAI wraps as tracked modules instead of perforating, even when a type name matches.
            :func:`perforate_detection_model` adds every candidate called more than once per forward, which PAI's
            perforated backpropagation cannot pair with a single error.
        track_ids: Module ids PAI wraps in ``TrackedNeuronModule``; ``pai_track_leaves`` fills it.
        restructure: Rebuild the detector's blocks into sub-blocks (:data:`RECIPES`) before wrapping. Block targets
            need it, since the sub-block modules only exist after it.
    """

    name: str
    perforate_ids: tuple[str, ...] = ()
    perforate_type_names: tuple[str, ...] = ()
    skip_ids: tuple[str, ...] = ()
    track_ids: tuple[str, ...] = ()
    restructure: bool = False


ALL = PerforationTarget(name="all", perforate_type_names=("Linear", "Conv2d"))
CLS_HEAD = PerforationTarget(name="cls_head", perforate_ids=(".class_embed",))
BBOX_HEAD = PerforationTarget(name="bbox_head", perforate_ids=(".bbox_embed",))
CLS_BBOX_HEAD = PerforationTarget(name="cls_bbox_head", perforate_ids=(".class_embed", ".bbox_embed"))
#: DINOv2 attention and FFN branches, norms inside, residual adds outside.
BACKBONE_BLOCKS = PerforationTarget(name="backbone_blocks", perforate_type_names=("PreNormBranch",), restructure=True)
#: Decoder self-attention, cross-attention, and FFN blocks with their post-norms.
DECODER_BLOCKS = PerforationTarget(name="decoder_blocks", perforate_type_names=("PostNormResidual",), restructure=True)
#: Two-stage ``enc_output`` Linear + LayerNorm pairs.
ENC_OUTPUT = PerforationTarget(name="enc_output", perforate_type_names=("PAISequential",), restructure=True)
#: Projector conv + norm units; the bottleneck residual adds stay outside them.
PROJECTOR_BLOCKS = PerforationTarget(name="projector_blocks", perforate_type_names=("ConvX",))
#: Segmentation head branches.
SEGMENTATION_BLOCKS = PerforationTarget(
    name="segmentation_blocks", perforate_type_names=("DepthwiseBranch", "SegmentationMLPBranch"), restructure=True
)
#: Every block above plus the detection heads.
ALL_BLOCKS = PerforationTarget(
    name="all_blocks",
    perforate_ids=(".class_embed",),
    perforate_type_names=(
        BACKBONE_BLOCKS.perforate_type_names
        + DECODER_BLOCKS.perforate_type_names
        + ENC_OUTPUT.perforate_type_names
        + PROJECTOR_BLOCKS.perforate_type_names
        + SEGMENTATION_BLOCKS.perforate_type_names
        + ("MLP",)
    ),
    restructure=True,
)

#: Targets selectable through ``TrainConfig.pai_target``.
PERFORATION_TARGETS: dict[str, PerforationTarget] = {
    t.name: t
    for t in (
        ALL,
        CLS_HEAD,
        BBOX_HEAD,
        CLS_BBOX_HEAD,
        BACKBONE_BLOCKS,
        DECODER_BLOCKS,
        ENC_OUTPUT,
        PROJECTOR_BLOCKS,
        SEGMENTATION_BLOCKS,
        ALL_BLOCKS,
    )
}

#: Module types :func:`list_perforable_modules` reports and :func:`record_forward_stats` measures.
PERFORABLE_TYPE_NAMES = (
    "Linear",
    "Conv2d",
    "LayerNorm",
    "Embedding",
    "MLP",
    "ConvX",
    "PreNormBranch",
    "PostNormResidual",
    "PAISequential",
    "DepthwiseBranch",
    "SegmentationMLPBranch",
)

#: Module types whose output keeps channels at axis 1 (``[B, C, H, W]``); everything else has channels last.
CHANNELS_FIRST_TYPE_NAMES = ("Conv2d", "ConvTranspose2d", "ConvX", "BatchNorm2d")


def get_perforation_target(target: PerforationTargetName | str | list[str] | tuple[str, ...]) -> PerforationTarget:
    """Look up a target by name, or build one from explicit module ids.

    Args:
        target: One of the keys of :data:`PERFORATION_TARGETS`, or a list of module ids with a leading dot. Ids may
            name sub-blocks, so an id list restructures the detector first.

    Returns:
        The matching :class:`PerforationTarget`.

    Raises:
        ValueError: If a name is unknown or an id lacks its leading dot.
    """
    if not isinstance(target, str):
        ids = tuple(target)
        bad = [i for i in ids if not i.startswith(".")]
        if not ids or bad:
            raise ValueError(f"pai_target ids must start with '.', like '.class_embed'; got {bad or ids}.")
        return PerforationTarget(name="custom", perforate_ids=ids, restructure=True)
    try:
        return PERFORATION_TARGETS[target]
    except KeyError:
        known = ", ".join(sorted(PERFORATION_TARGETS))
        raise ValueError(f"Unknown pai_target {target!r}; choose from {known} or pass a list of module ids.") from None


def list_perforable_modules(model: nn.Module) -> dict[str, str]:
    """Map every weight-carrying module id of the detector to its type name.

    Args:
        model: ``LWDETR`` before PAI converts it, restructured if block ids are wanted.

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
            "Run list_perforable_modules on the (restructured) model for the ids it does have."
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


def record_forward_stats(
    model: nn.Module,
    sample: torch.Tensor,
    module_ids: tuple[str, ...] = (),
) -> tuple[dict[str, int], dict[str, int]]:
    """Run one forward of the unperforated model and measure every perforable module.

    The pass runs in train mode because RF-DETR calls the extra ``group_detr`` head copies only while training.
    Buffers are restored afterwards so BatchNorm statistics do not see the sample.

    Args:
        model: ``LWDETR`` before PAI converts it.
        sample: Image batch of at least two images, ``[B, 3, H, W]``, on the model's device.
        module_ids: Ids measured in addition to every module whose type is in :data:`PERFORABLE_TYPE_NAMES`.

    Returns:
        ``(ranks, calls)``: the output rank and the number of calls per forward of every measured module, keyed by
        module id. A module whose output is not a tensor gets a call count but no rank.
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
        if name and (type(module).__name__ in PERFORABLE_TYPE_NAMES or "." + name in module_ids)
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


def output_dimensions_for(module: nn.Module, rank: int) -> list[int]:
    """Return PAI's output dimension vector: ``0`` at the neuron axis, ``-1`` elsewhere.

    A module declaring ``neuron_axis`` (the sub-blocks) decides for itself. Otherwise convolution-like types keep
    channels at axis 1 and everything else at the end, which also covers a Linear applied to 4D stacked outputs.

    Args:
        module: The module PAI wrapped (the ``main_module``).
        rank: Rank of the module output.

    Returns:
        A list of length ``rank`` with a single ``0``.
    """
    axis = getattr(module, "neuron_axis", None)
    if axis is None:
        axis = 1 if type(module).__name__ in CHANNELS_FIRST_TYPE_NAMES else rank - 1
    if axis < 0:
        axis += rank
    dims = [-1] * rank
    dims[axis] = 0
    return dims


def apply_output_dimensions(model: nn.Module, ranks: dict[str, int]) -> None:
    """Set every ``PAINeuronModule``'s neuron axis from the ranks :func:`record_forward_stats` measured.

    Args:
        model: Model returned by ``perforate_model``.
        ranks: ``{module_id: output.ndim}`` of the unperforated model.

    Raises:
        ValueError: If a wrapped module was not called during the measuring forward or returned no tensor.
    """
    wrapped = [module for module in model.modules() if type(module).__name__ == "PAINeuronModule"]
    missing = [str(module.name) for module in wrapped if str(module.name) not in ranks]
    if missing:
        raise ValueError(
            f"PerforatedAI wrapped modules the measuring forward never called or that return no tensor: {missing}."
        )
    for module in wrapped:
        main = module.get_submodule("main_module")
        cast(Any, module).set_this_output_dimensions(output_dimensions_for(main, ranks[str(module.name)]))


# --- Dendrite copies --------------------------------------------------------------------------------------------------
# PerforatedAI builds every dendrite candidate as a deep copy of the wrapped module and re-randomizes all of its
# parameters (modules_perforatedai.init_params). For a sub-block that is wrong twice over: a post-norm block would keep
# its skip connection, and norm affine and layer-scale weights would start at random values instead of the neuron's.
# Until the library grows the hook, init_params is wrapped here to fix both on every copy.

#: Normalization module type names whose parameters a dendrite copy inherits from the neuron.
NORM_TYPE_NAMES = ("LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm", "RMSNorm")

_dendrite_copies_configured = 0


def configure_dendrite_copy(module: nn.Module, main: nn.Module) -> None:
    """Finish a fresh dendrite copy after PAI re-randomized its parameters.

    Calls ``configure_as_dendrite`` when the module defines it (post-norm blocks drop their skip), then copies the
    parameters of every normalization module and of every scale-only module (all own parameters 1D, like layer scale
    or ``gamma``) back from the neuron's ``main_module``.

    Args:
        module: The dendrite copy PAI just initialized.
        main: The neuron's ``main_module`` the copy was made from.
    """
    global _dendrite_copies_configured
    configure = getattr(module, "configure_as_dendrite", None)
    if callable(configure):
        configure()
    main_params = dict(main.named_parameters())
    with torch.no_grad():
        for name, sub in module.named_modules():
            own = list(sub.named_parameters(recurse=False))
            if not own:
                continue
            scale_only = all(p.ndim <= 1 for _, p in own)
            if type(sub).__name__ not in NORM_TYPE_NAMES and not scale_only:
                continue
            for param_name, param in own:
                full = f"{name}.{param_name}" if name else param_name
                source = main_params.get(full)
                if source is not None and source.shape == param.shape:
                    param.copy_(source)
    _dendrite_copies_configured += 1


def install_dendrite_init_hook() -> None:
    """Wrap ``modules_perforatedai.init_params`` so :func:`configure_dendrite_copy` runs on every dendrite copy.

    Idempotent. ``create_new_dendrite_module`` looks ``init_params`` up as a module global at call time, which is
    what makes the wrap take effect.
    """
    from perforatedai import modules_perforatedai as mpa

    original = mpa.init_params
    if getattr(original, "_rfdetr_wrapped", False):
        return

    def init_params(module: nn.Module, neuron_main_module: nn.Module) -> None:
        original(module, neuron_main_module)
        configure_dendrite_copy(module, neuron_main_module)

    init_params._rfdetr_wrapped = True  # type: ignore[attr-defined]
    mpa.init_params = init_params


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
    target_name = get_perforation_target(train_config.pai_target).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_label.lower()}_{target_name}_dendritic_{timestamp}"


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


def write_key_map(key_map: dict[str, str], output_dir: str | Path) -> Path:
    """Write the restructure key map next to the run's checkpoints.

    Args:
        key_map: ``{original_key: restructured_key}`` from :func:`restructure_detection_model`.
        output_dir: The run's ``output_dir``.

    Returns:
        Path of the written file.
    """
    path = Path(output_dir) / KEY_MAP_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(key_map, handle, indent=1, sort_keys=True)
    return path


def perforate_detection_model(model: nn.Module, model_config: ModelConfig, train_config: TrainConfig) -> nn.Module:
    """Restructure if the target needs it, configure PAI, wrap the detector, verify, and optionally load a saved system.

    Args:
        model: ``LWDETR`` with pretrained weights already loaded.
        model_config: Architecture configuration, used to label the PAI save folder.
        train_config: Training configuration with ``perforate=True``.

    Returns:
        The perforated model PAI returned.
    """
    gpa, upa = import_perforatedai()
    target = get_perforation_target(train_config.pai_target)
    if target.restructure:
        restructured = restructure_detection_model(model)
        model = restructured.model
        moved = sum(1 for old, new in restructured.key_map.items() if old != new)
        key_map_path = write_key_map(restructured.key_map, train_config.output_dir)
        logger.info(
            "PerforatedAI restructured %d blocks into sub-blocks (%d state dict keys renamed, map at %s)",
            len(restructured.replaced),
            moved,
            key_map_path,
        )
    install_dendrite_init_hook()
    if train_config.pai_track_leaves:
        target = replace(target, track_ids=tuple(build_tracked_leaf_ids(model, target)))
        logger.info("PerforatedAI tracking %d leaf modules", len(target.track_ids))
    device = next(model.parameters()).device
    resolution = model_config.resolution
    sample = torch.zeros(2, 3, resolution, resolution, device=device)
    ranks, calls = record_forward_stats(model, sample, module_ids=resolve_perforate_ids(model, target))
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


# --- EMA glue ---------------------------------------------------------------------------------------------------------
# The framework-free half lives in rfdetr.training.perforated_ema. These functions bind it to RFDETREMACallback's
# AveragedModel, whose positional update_parameters is replaced by the by-name update.


def find_ema_callback(trainer: Trainer) -> RFDETREMACallback | None:
    """Return the EMA callback of the trainer, or ``None`` when ``use_ema=False``."""
    for callback in trainer.callbacks:  # type: ignore[attr-defined]
        if isinstance(callback, RFDETREMACallback):
            return callback
    return None


def install_named_ema_update(ema_cb: RFDETREMACallback, pl_module: RFDETRModelModule) -> None:
    """Replace ``AveragedModel.update_parameters`` with the by-name update that mirrors PAI's buffers.

    Args:
        ema_cb: EMA callback holding the averaged copy.
        pl_module: Module holding the live perforated ``LWDETR`` at ``.model``.
    """
    average_model = getattr(ema_cb, "_average_model", None)
    if average_model is None:
        return
    # The averaged copy wraps the LightningModule, so its names carry the ``model.`` prefix.
    mirror = {"model." + name for name in pai_buffer_names(pl_module.model)}

    def update_parameters(model: LightningModule) -> None:
        n_averaged = int(average_model.n_averaged.item())
        ema_update_by_name(
            average_model.module, model, ema_cb._effective_decay(n_averaged), mirror, first_update=n_averaged == 0
        )
        average_model.n_averaged += 1

    average_model.update_parameters = update_parameters


def sync_ema_buffers(pl_module: RFDETRModelModule, ema_cb: RFDETREMACallback) -> int:
    """Re-register every integer buffer of the EMA copy whose live shape changed since the last step.

    Args:
        pl_module: Module holding the live perforated ``LWDETR`` at ``.model``.
        ema_cb: EMA callback holding the averaged copy.

    Returns:
        Number of buffers replaced.
    """
    average_model = getattr(ema_cb, "_average_model", None)
    if average_model is None:
        return 0
    return sync_resized_buffers(average_model.module, pl_module)


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
    return swap_in_weights(pl_module.model, ema_inner.model, ema_weight_keys(pl_module.model))


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
        copies_before = _dendrite_copies_configured
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
            restore_weights(module.model, saved)
            if ema_cb is not None:
                sync_ema_buffers(module, ema_cb)
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
            "state restored on %d tensors, a fresh plateau schedule, and the EMA. Dendrite init hook configured %d "
            "copies.",
            trainer.current_epoch,
            gpa.pai_tracker.member_vars["mode"],
            params,
            restored,
            _dendrite_copies_configured - copies_before,
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
        A state dict with the wrapped modules' own tensors back at their pre-wrapping keys. If the run restructured the
        detector these are the sub-block keys; :func:`extract_start_weights` maps them back with the run's key map.
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
    key_map_path: str | Path | None = None,
) -> Path:
    """Write a plain RF-DETR checkpoint from a saved PAI stage, loadable through ``pretrain_weights``.

    Args:
        stage_path: PAI stage file, ``<system folder>/<stage>.pt``.
        out_path: Checkpoint to write. ``None`` writes ``<stage>_plain.pth`` beside the stage file.
        model_args: Stored under the checkpoint's ``args`` key, typically a ``training_config.json`` ``model_config``.
        key_map_path: The run's :data:`KEY_MAP_NAME` file when the run restructured the detector; its inverse puts the
            sub-block keys back on the plain ``LWDETR`` names.

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
    if key_map_path is not None:
        with open(key_map_path) as handle:
            key_map = json.load(handle)
        plain = remap_state_dict(plain, invert_key_map(key_map))
    logger.info("Extracted %d of %d tensors from %s into %s", len(plain), len(state), stage, out)
    torch.save({"model": plain, "args": dict(model_args or {})}, out)
    return out
