################################################################################
# PerforatedAI dendrite training for RF-DETR.                                  #
################################################################################
from __future__ import annotations

#
"""
Imports
"""

import copy
import torch
import shutil
import fnmatch
import inspect
import importlib

from pathlib               import Path
from datetime              import datetime
from torch                 import nn, optim
from torch.optim.swa_utils import AveragedModel
from collections.abc       import Callable
from typing                import TYPE_CHECKING, Any, cast
from pytorch_lightning     import Callback, LightningModule, Trainer

from rfdetr._namespace                                 import (
    _namespace_from_configs,
)
from rfdetr.config                                     import (
    ModelConfig,
    TrainConfig,
    _resolve_native_optimizer,
)
from rfdetr.training.callbacks                         import RFDETREMACallback
from rfdetr.training.callbacks.coco_eval               import (
    _get_ema_inner_module,
)
from rfdetr.training.param_groups                      import get_param_dict
from rfdetr.models.backbone.dinov2_with_windowed_attn  import (
    WindowedDinov2WithRegistersLayer,
)
from rfdetr.models.heads.segmentation                  import (
    DepthwiseConvBlock,
    MLPBlock,
)
from rfdetr.models.transformer                         import (
    Transformer,
    TransformerDecoderLayer,
)

# Imported only when perforate=True, so perforatedai is a hard import here
from perforatedai import globals_perforatedai as gpa
from perforatedai import modules_perforatedai as mpa
from perforatedai import utils_perforatedai   as upa

if TYPE_CHECKING:
    from rfdetr.training.module_model import RFDETRModelModule

#
"""
Config
"""

# Model with the PAI tracker removed, written to output_dir at train end
clean_model_name = 'final_clean_model.pth'
# Copy of pai_config written to output_dir
pai_config_name  = 'pai_config.json'
# Types whose output keeps channels at axis 1
channels_first_type_names = (
    'Conv2d',
    'ConvTranspose2d',
    'ConvX',
    'BatchNorm2d',
)
# Norm types whose parameters a dendrite copy inherits from the neuron
norm_type_names = (
    'LayerNorm',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'GroupNorm',
    'RMSNorm',
)

# Switch to dendrites at the first validation, w/ neuron at lr 0.
# NOTE: WILL SOON BE PART OF PERFORATEDBP
force_first_switch = False

# Record handed to an observer after every scored validation
ObserverRecord = dict[str, Any]

__all__ = [
    'PerforatedAICallback',
    'clean_model_name',
    'clean_perforated_model',
    'list_perforable_modules',
    'pai_config_name',
    'pai_replacements',
    'perforate_detection_model',
    'replace_blocks',
    'setup_perforated_optimizer',
]

#
"""
Blocks
"""

# NOTE:
# Perforation works best when a layer that gets Perforation is included in
# a module that contains its norm and has residuals outside the module.
# The below classes regroups one RF-DETR block that way, reusing the
# children of the block.

class DinoAttentionPAI(nn.Module):
    '''
    layer_scale1(attention(norm1(x))) of a DINOv2 layer
    '''

    def __init__(self, other: WindowedDinov2WithRegistersLayer) -> None:
        super().__init__()
        self.norm1        = other.norm1
        self.attention    = other.attention
        self.layer_scale1 = other.layer_scale1

    def forward(
        self,
        hidden_states    : torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        out = self.attention(
            self.norm1(hidden_states),
            output_attentions = output_attentions,
        )[0]
        return cast(torch.Tensor, self.layer_scale1(out))

class DinoMLPPAI(nn.Module):
    '''
    layer_scale2(mlp(norm2(x))) of a DINOv2 layer
    '''

    def __init__(self, other: WindowedDinov2WithRegistersLayer) -> None:
        super().__init__()
        self.norm2        = other.norm2
        self.mlp          = other.mlp
        self.layer_scale2 = other.layer_scale2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.mlp(self.norm2(hidden_states))
        return cast(torch.Tensor, self.layer_scale2(out))

class DinoLayerPAI(nn.Module):
    '''
    WindowedDinov2WithRegistersLayer with attention and mlp as branches
    '''
    def __init__(self, other: WindowedDinov2WithRegistersLayer) -> None:
        super().__init__()
        self.num_windows = other.num_windows
        self.drop_path   = other.drop_path
        self.attention   = DinoAttentionPAI(other)
        self.mlp         = DinoMLPPAI(other)

    def forward(
        self,
        hidden_states     : torch.Tensor,
        output_attentions : bool = False,
        run_full_attention: bool = False,
    ) -> tuple[torch.Tensor]:
        shortcut = hidden_states
        if run_full_attention:
            batch_windows, tokens_per_window, channels = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            hidden_states = hidden_states.view(
                batch_windows // num_windows_squared,
                num_windows_squared * tokens_per_window,
                channels,
            )
        attention_output = self.attention(
            hidden_states,
            output_attentions = output_attentions,
        )
        if run_full_attention:
            attention_output = attention_output.view_as(shortcut)
        hidden_states = self.drop_path(attention_output) + shortcut
        layer_output  = self.drop_path(self.mlp(hidden_states)) + hidden_states
        return (layer_output,)

class PostNormPAI(nn.Module):
    '''
    norm(x + dropout(body(x)))
    (Dendrite copy omits the residual)
    '''

    # Plain attribute so the state dict stays that of the original layer
    residual: bool = True

    def configure_as_dendrite(self) -> None:
        self.residual = False

    def add_and_norm(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out = self.dropout(out)
        return cast(torch.Tensor, self.norm(x + out if self.residual else out))

class DecoderSelfAttentionPAI(PostNormPAI):
    '''
    self_attn + dropout1 + norm1 of a decoder layer
    '''

    def __init__(self, other: TransformerDecoderLayer) -> None:
        super().__init__()
        self.self_attn  = other.self_attn
        self.dropout    = other.dropout1
        self.norm       = other.norm1
        self.group_detr = other.group_detr

    def forward(
        self,
        tgt                 : torch.Tensor,
        query_pos           : torch.Tensor | None,
        tgt_mask            : torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bs, num_queries, _ = tgt.shape
        q = k = tgt if query_pos is None else tgt + query_pos
        v = tgt
        if self.training:
            # Groups attend separately, as in the original layer
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            k = q
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)
        tgt2 = self.self_attn(
            q,
            k,
            v,
            attn_mask        = tgt_mask,
            key_padding_mask = tgt_key_padding_mask,
            need_weights     = False,
        )[0]
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        return self.add_and_norm(tgt, tgt2)

class DecoderCrossAttentionPAI(PostNormPAI):
    '''
    cross_attn + dropout2 + norm2 of a decoder layer
    '''

    def __init__(self, other: TransformerDecoderLayer) -> None:
        super().__init__()
        self.cross_attn = other.cross_attn
        self.dropout    = other.dropout2
        self.norm       = other.norm2

    def forward(
        self,
        tgt                    : torch.Tensor,
        query_pos              : torch.Tensor | None,
        reference_points       : torch.Tensor | None,
        memory                 : torch.Tensor,
        spatial_shapes         : torch.Tensor | None,
        level_start_index      : torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
        spatial_shapes_hw      : list[tuple[int, int]] | None,
    ) -> torch.Tensor:
        query = tgt if query_pos is None else tgt + query_pos
        tgt2  = self.cross_attn(
            query,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_spatial_shapes_hw = spatial_shapes_hw,
        )
        return self.add_and_norm(tgt, tgt2)

class DecoderFFNPAI(PostNormPAI):
    '''
    linear1 + activation + linear2 + dropout3 + norm3 of a decoder layer
    '''

    def __init__(self, other: TransformerDecoderLayer) -> None:
        super().__init__()
        self.linear1    = other.linear1
        self.activation = other.activation
        self.dropout_ff = other.dropout
        self.linear2    = other.linear2
        self.dropout    = other.dropout3
        self.norm       = other.norm3

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.dropout_ff(self.activation(self.linear1(tgt))))
        return self.add_and_norm(tgt, tgt2)

class DecoderLayerPAI(nn.Module):
    '''
    TransformerDecoderLayer as self_attn, cross_attn and ffn sub-blocks
    '''
    def __init__(self, other: TransformerDecoderLayer) -> None:
        super().__init__()
        self.self_attn  = DecoderSelfAttentionPAI(other)
        self.cross_attn = DecoderCrossAttentionPAI(other)
        self.ffn        = DecoderFFNPAI(other)

    def forward(
        self,
        tgt                    : torch.Tensor,
        memory                 : torch.Tensor,
        tgt_mask               : torch.Tensor | None = None,
        memory_mask            : torch.Tensor | None = None,
        tgt_key_padding_mask   : torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        query_pos              : torch.Tensor | None = None,
        reference_points       : torch.Tensor | None = None,
        spatial_shapes         : torch.Tensor | None = None,
        spatial_shapes_hw      : list[tuple[int, int]] | None = None,
        level_start_index      : torch.Tensor | None = None,
        keypoint_tgt           : torch.Tensor | None = None,
        keypoint_pos           : torch.Tensor | None = None,
        keypoint_class_mask    : torch.Tensor | None = None,
        kp_cross_attn_memory   : torch.Tensor | None = None,
    ) -> torch.Tensor:
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

def transformer_pai(other: Transformer) -> Transformer:
    '''
    Group each two-stage enc_output[g] with its norm in a PAISequential

    Notes:
        - Edits and returns the same Transformer rather than wrapping it
        - forward computes enc_output_norm[g](enc_output[g](x)). 
        - enc_output[g] = PAISequential([linear, norm]) 
          and
          enc_output_norm[g] = Identity
        - This is identical to norm(linear(x)), so forward needs no change
    '''
    enc_output      = getattr(other, 'enc_output', None)
    enc_output_norm = getattr(other, 'enc_output_norm', None)
    if enc_output is None or enc_output_norm is None:
        return other
    for g in range(len(enc_output)):
        pair               = [enc_output[g], enc_output_norm[g]]
        enc_output[g]      = gpa.PAISequential(pair)
        enc_output_norm[g] = nn.Identity()
    return other

class DepthwiseBranchPAI(DepthwiseConvBlock):
    '''
    Segmentation DepthwiseConvBlock without its residual add
    '''

    # Output keeps channels at axis 1
    neuron_axis: int = 1

    def __init__(self, other: DepthwiseConvBlock) -> None:
        nn.Module.__init__(self)
        self.dwconv  = other.dwconv
        self.norm    = other.norm
        self.pwconv1 = other.pwconv1
        self.act     = other.act
        self.gamma   = other.gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.pwconv1(self.norm(x)))
        if self.gamma is not None:
            x = self.gamma * x
        return x.permute(0, 3, 1, 2)

class DepthwiseConvBlockPAI(nn.Module):
    '''
    Segmentation DepthwiseConvBlock as x + branch(x)
    '''

    def __init__(self, other: DepthwiseConvBlock) -> None:
        super().__init__()
        self.branch = DepthwiseBranchPAI(other)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)

class MLPBranchPAI(nn.Module):
    '''
    Segmentation MLPBlock without its residual add
    '''

    def __init__(self, other: MLPBlock) -> None:
        super().__init__()
        self.norm_in = other.norm_in
        self.layers  = other.layers
        self.gamma   = other.gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x

class MLPBlockPAI(nn.Module):
    '''
    Segmentation MLPBlock as x + branch(x)
    '''

    def __init__(self, other: MLPBlock) -> None:
        super().__init__()
        self.branch = MLPBranchPAI(other)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)

# Original block type to its PAI version
pai_replacements: dict[type[nn.Module], Callable[[nn.Module], nn.Module]] = {
    WindowedDinov2WithRegistersLayer: DinoLayerPAI,
    TransformerDecoderLayer         : DecoderLayerPAI,
    Transformer                     : transformer_pai,
    DepthwiseConvBlock              : DepthwiseConvBlockPAI,
    MLPBlock                        : MLPBlockPAI,
}

def replace_blocks(model: nn.Module) -> None:
    '''
    Swap every block for its PAI version in place

    Notes:
        - Done here rather than by perforate_model because the target ids
          and the measuring forward need the sub-blocks first
    '''
    for name, module in list(model.named_modules()):
        build = pai_replacements.get(type(module))
        if not name or build is None:
            continue
        parent_name, _, attr = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, build(cast(Any, module)))

def list_perforable_modules(model: nn.Module) -> dict[str, str]:
    '''
    Every id pai_target can name, with its type, after block replacement

    Notes:
        - Works on a copy, so model is left as it is
    '''
    model = copy.deepcopy(model)
    replace_blocks(model)
    return {
        '.' + name: type(module).__name__
        for name, module in model.named_modules()
        if name and any(True for _ in module.parameters())
    }

def inside_target(
    module_id   : str,
    ids         : list[str],
    *,
    include_self: bool,
) -> bool:
    '''
    Whether module_id is one of ids or lies below one
    '''
    return any(
        module_id.startswith(p + '.') or (include_self and module_id == p)
        for p in ids
    )

#
"""
Wrapping
"""
def perforate_detection_model(
    model       : nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> nn.Module:
    '''
    Replace blocks, configure PAI, wrap the detector

    Notes:
        - pai_target patterns are expanded in order. 
        - Ids the model lacks are dropped and nested ids keep only the outermost
        - Modules called more than once per forward 
          (the two-stage head copies) 
          are tracked instead, since perforated backpropagation pairs
          one output with one error
        - Loads a saved PAI system when pai_load_folder is set
    '''
    tc = train_config
    replace_blocks(model)

    names = ['.' + name for name, _ in model.named_modules() if name]
    ids: list[str] = []
    for pattern in tc.pai_target:
        ids += [
            n for n in names if fnmatch.fnmatchcase(n, pattern) and n not in ids
        ]
    ids = [i for i in ids if not any(i.startswith(o + '.') for o in ids)]

    # One train-mode forward records each id's output rank and call count.
    # Train mode runs the group_detr head copies. Buffers are restored after
    ranks: dict[str, int] = {}
    calls: dict[str, int] = {}

    def record(module_id: str) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            calls[module_id] = calls.get(module_id, 0) + 1
            if isinstance(output, torch.Tensor):
                ranks[module_id] = output.ndim

        return hook

    handles = [
        model.get_submodule(i[1:]).register_forward_hook(record(i))
        for i in ids
    ]
    buffers      = {name: b.clone() for name, b in model.named_buffers()}
    was_training = model.training
    resolution   = model_config.resolution
    device       = next(model.parameters()).device
    model.train()
    try:
        with torch.no_grad():
            model(torch.zeros(2, 3, resolution, resolution, device=device))
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])

    perforate = [i for i in ids if calls.get(i, 0) <= 1]
    track     = [i for i in ids if calls.get(i, 0) > 1]
    if tc.pai_track_leaves:
        track += [
            '.' + name
            for name, module in model.named_modules()
            if name
            and not any(True for _ in module.children())
            and any(True for _ in module.parameters(recurse=False))
            and not inside_target('.' + name, ids, include_self=True)
        ]
    tracked_params = [
        '.' + name
        for name, _ in model.named_parameters()
        if not inside_target('.' + name, ids, include_self=False)
    ]

    gpa.pc.set_testing_dendrite_capacity(False)

    # The pai_config JSON loads first
    if tc.pai_config is not None:
        output_dir = Path(tc.output_dir)
        gpa.pc.load_config(str(tc.pai_config))
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tc.pai_config, output_dir / pai_config_name)
    # The following settings have priority over the config
    gpa.pc.set_module_names_to_perforate([])
    gpa.pc.set_module_ids_to_perforate(perforate)
    gpa.pc.append_module_ids_to_track(track)
    gpa.pc.set_parameter_ids_to_track(tracked_params)
    # Default only, each wrapped module gets its measured axis below
    gpa.pc.set_output_dimensions([-1, -1, 0])
    gpa.pc.set_perforated_backpropagation(True)
    # Skip PAI's prompts
    # Unwrapped modules and weight decay are intended
    gpa.pc.set_unwrapped_modules_confirmed(True)
    gpa.pc.set_weight_decay_accepted(True)
    gpa.pc.set_configuration_confirmed(True)
    if gpa.pc.get_global_candidates() > 1:
        # Extra candidates register buffers lazily and need the stacked
        # backward workaround
        gpa.pc.set_strict_loading(False)
        gpa.pc.set_no_backward_workaround(True)
    spec = tc.pai_forward_function
    if spec is not None:
        if spec == 'identity':
            spec = nn.Identity()
        elif isinstance(spec, str):
            spec = getattr(torch, spec, None) or getattr(nn.functional, spec)
        gpa.pc.set_pai_forward_function(spec)

    # PAI's modules_perforatedai.init_params re-randomizes every dendrite
    # copy. Wrap it, once, to turn off the residual add in PostNormPAI
    # copies and copy norm and scale-only parameters back from the neuron
    original = mpa.init_params
    if not getattr(original, 'rfdetr_wrapped', False):

        def init_params(module: nn.Module, main: nn.Module) -> None:
            original(module, main)
            configure = getattr(module, 'configure_as_dendrite', None)
            if callable(configure):
                configure()
            main_params = dict(main.named_parameters())
            with torch.no_grad():
                for name, sub in module.named_modules():
                    own   = list(sub.named_parameters(recurse=False))
                    norm  = type(sub).__name__ in norm_type_names
                    scale = all(p.ndim <= 1 for _, p in own)
                    if not own or not (norm or scale):
                        continue
                    for param_name, param in own:
                        key    = f'{name}.{param_name}' if name else param_name
                        source = main_params.get(key)
                        if source is not None and source.shape == param.shape:
                            param.copy_(source)

        init_params.rfdetr_wrapped = True
        mpa.init_params            = init_params

    if tc.pai_save_name:
        save_name = tc.pai_save_name
    else:
        label     = type(model_config).__name__.removesuffix('Config')
        label     = (model_config.model_name or label).lower()
        save_name = f'{label}_dendritic_{datetime.now():%Y%m%d_%H%M%S}'
    model = upa.perforate_model(
        model,
        save_name        = save_name,
        maximizing_score = True,
    )

    # Neuron axis of each wrapped module
    for module in model.modules():
        if type(module).__name__ != 'PAINeuronModule':
            continue
        main = module.get_submodule('main_module')
        axis = getattr(main, 'neuron_axis', None)
        if axis is None:
            axis = 1 if type(main).__name__ in channels_first_type_names else -1
        dims       = [-1] * ranks[str(module.name)]
        dims[axis] = 0
        cast(Any, module).set_this_output_dimensions(dims)

    if tc.pai_load_folder is not None:
        model = upa.load_system(
            model,
            str(tc.pai_load_folder),
            tc.pai_load_stage,
            switch_call = True,
        )
    return model

def before_first_switch() -> bool:
    '''
    True in the neuron epoch before a forced first switch
    '''
    member_vars = gpa.pai_tracker.member_vars
    return bool(
        force_first_switch
        and member_vars['mode'] == 'n'
        and len(member_vars['switch_epochs']) == 0
    )

#
"""
Optimizer
"""
def resolve_class(spec: str | type, native: bool = False) -> type:
    '''
    A class, a torch.optim name when native, or an imported dotted path
    '''
    if isinstance(spec, type):
        return spec
    if native and '.' not in spec:
        return _resolve_native_optimizer(spec)
    module_path, _, attribute = spec.rpartition('.')
    return getattr(importlib.import_module(module_path), attribute)

def setup_perforated_optimizer(pl_module: RFDETRModelModule) -> optim.Optimizer:
    '''
    Build the user's optimizer and scheduler through PAI, which controls both

    Notes:
        - Classes and kwargs come from TrainConfig.optimizer, optimizer_kwargs,
          lr_scheduler and lr_scheduler_kwargs. Lightning gets no scheduler.
          PAI steps it inside add_validation_score, which its lr search needs
        - Dendrite parameters get their own groups at weight_decay 0. Dendrite
          mode keeps only PAINeuronModule parameters, which freezes the rest,
          at pai_dendrite_lr when set
        - Starts with empty state. The callback restores the stashed moments
    '''
    tc    = pl_module.train_config
    model = pl_module.model

    # Parameters PAI added beside each main_module
    dendrites: set[int] = set()
    for module in model.modules():
        if type(module).__name__ != 'PAINeuronModule':
            continue
        main       = module.get_submodule('main_module').parameters()
        main_ids   = {id(p) for p in main}
        dendrites |= {
            id(p) for p in module.parameters() if id(p) not in main_ids
        }

    dendrite_mode = gpa.pai_tracker.member_vars['mode'] == 'p'
    allowed       = {id(p) for p in upa.get_pai_network_params(model)}
    namespace     = _namespace_from_configs(pl_module.model_config, tc)
    groups: list[dict[str, Any]] = []
    for group in get_param_dict(namespace, model):
        params = group['params']
        if dendrite_mode:
            params = [p for p in params if id(p) in allowed]
        model_params    = [p for p in params if id(p) not in dendrites]
        dendrite_params = [p for p in params if id(p) in dendrites]
        if model_params:
            groups.append({**group, 'params': model_params})
        if dendrite_params:
            groups.append(
                {**group, 'params': dendrite_params, 'weight_decay': 0.0}
            )
    if dendrite_mode and tc.pai_dendrite_lr is not None:
        for group in groups:
            group['lr'] = tc.pai_dendrite_lr
    frozen = before_first_switch()
    if frozen:
        for group in groups:
            group['lr'] = 0.0

    optimizer_class = resolve_class(tc.optimizer, native=True)
    scheduler_class = resolve_class(tc.lr_scheduler)
    gpa.pai_tracker.set_optimizer(optimizer_class)
    gpa.pai_tracker.set_scheduler(scheduler_class)
    optimizer_args: dict[str, Any] = {
        'params': groups,
        'lr'    : 0.0 if frozen else tc.lr,
    }
    if 'weight_decay' in inspect.signature(optimizer_class).parameters:
        optimizer_args['weight_decay'] = tc.weight_decay
    if optimizer_class is optim.AdamW:
        optimizer_args['fused'] = next(model.parameters()).is_cuda
    optimizer_args.update(tc.optimizer_kwargs)
    optimizer, _ = gpa.pai_tracker.setup_optimizer(
        model,
        optimizer_args,
        dict(tc.lr_scheduler_kwargs),
    )
    # Lightning's lr logging reads initial_lr, which narrowed groups can lack
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    return cast(optim.Optimizer, optimizer)

#
"""
EMA
"""
# PAI checkpoints the model it is handed, so the EMA weights that were scored
# are swapped into the live model around add_validation_score. 
# PAI also registers buffers lazily and resizes some, so the EMA update 
# pairs tensors by name and mirrors PAI's own buffers instead of averaging them

def replace_buffer(root: nn.Module, name: str, tensor: torch.Tensor) -> None:
    '''
    Re-register a nested buffer with a copy of tensor, for shape changes
    '''
    owner_name, _, attr = name.rpartition('.')
    owner = root.get_submodule(owner_name) if owner_name else root
    owner.register_buffer(attr, tensor.detach().clone())

def find_ema_callback(trainer: Trainer) -> RFDETREMACallback | None:
    '''
    The trainer's EMA callback, None when use_ema is off
    '''
    for callback in trainer.callbacks:
        if isinstance(callback, RFDETREMACallback):
            return callback
    return None

def install_named_ema_update(
    ema_cb   : RFDETREMACallback,
    pl_module: RFDETRModelModule,
) -> None:
    '''
    Replace AveragedModel.update_parameters with an update that pairs
    tensors by name

    Notes:
        - Buffers under a PAINeuronModule, non-float tensors and every
          tensor on the first update are copied, the rest averaged
        - Resized non-float buffers are re-registered, resized float ones
          skipped
    '''
    average_model = getattr(ema_cb, '_average_model', None)
    if average_model is None:
        return
    prefixes = [
        name + '.' for name, m in pl_module.model.named_modules()
        if type(m).__name__ == 'PAINeuronModule'
    ]
    # The averaged copy wraps the LightningModule, hence the model. prefix
    mirror = {
        'model.' + name for name, _ in pl_module.model.named_buffers()
        if any(name.startswith(p) for p in prefixes)
    }

    @torch.no_grad()
    def update_parameters(live: LightningModule) -> None:
        ema          = average_model.module
        n_averaged   = int(average_model.n_averaged.item())
        decay        = ema_cb._effective_decay(n_averaged)
        ema_tensors  = {
            **dict(ema.named_parameters()),
            **dict(ema.named_buffers()),
        }
        live_tensors = {
            **dict(live.named_parameters()),
            **dict(live.named_buffers()),
        }
        groups: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        for name, ema_t in ema_tensors.items():
            live_t = live_tensors.get(name)
            if live_t is None:
                continue
            if live_t.shape != ema_t.shape:
                if not ema_t.is_floating_point():
                    replace_buffer(ema, name, live_t)
                continue
            copy_over = name in mirror or not ema_t.is_floating_point()
            if copy_over or n_averaged == 0:
                ema_t.copy_(live_t)
                continue
            averaged, current = groups.setdefault(
                (ema_t.device, ema_t.dtype),
                ([], []),
            )
            averaged.append(ema_t)
            current.append(live_t.to(ema_t.dtype))
        for averaged, current in groups.values():
            torch._foreach_mul_(averaged, decay)
            torch._foreach_add_(averaged, current, alpha=1.0 - decay)
        average_model.n_averaged += 1

    average_model.update_parameters = update_parameters

def rebuild_ema(
    ema_cb   : RFDETREMACallback,
    pl_module: RFDETRModelModule,
) -> None:
    '''
    Rebuild the EMA copy from the restructured module, keeping n_averaged
    '''
    old        = getattr(ema_cb, '_average_model', None)
    n_averaged = int(old.n_averaged.item()) if old is not None else 0
    # PAI processor state does not survive the deepcopy inside AveragedModel
    gpa.pai_tracker.clear_all_processors()
    new = AveragedModel(
        model        = pl_module,
        device       = pl_module.device,
        use_buffers  = True,
        multi_avg_fn = ema_cb._multi_avg_fn,
    )
    new.n_averaged.fill_(n_averaged)
    new.eval()
    ema_cb._average_model = new
    install_named_ema_update(ema_cb, pl_module)

#
"""
Callback
"""
class PerforatedAICallback(Callback):
    '''
    Feed the validation metric to PerforatedAI and act on what it decides

    Notes:
        - Runs after COCOEvalCallback and BestModelCallback. 
          Swaps the EMA weights in, passes PAI the score, then stops, 
          restores the live weights, or rebuilds the optimizer and EMA 
          around the restructured model with the optimizer moments 
          restored by name

    Signature:
        monitor (str):
            - callback_metrics key scored in neuron phases
        observer (Callable[[ObserverRecord], None] | None):
            - Called after every scored validation with the epoch, mode,
              dendrite count, val/ metrics, correlations and PAI's decision
    '''

    def __init__(
        self,
        monitor : str = 'val/mAP_50_95',
        observer: Callable[[ObserverRecord], None] | None = None,
    ) -> None:
        super().__init__()
        self.monitor      = monitor
        self.observer     = observer
        self.moment_stash: dict[str, dict[str, Any]] = {}

    def set_observer(
        self,
        observer: Callable[[ObserverRecord], None] | None,
    ) -> None:
        self.observer = observer

    def on_fit_start(
        self,
        trainer  : Trainer,
        pl_module: LightningModule,
    ) -> None:
        module = cast('RFDETRModelModule', pl_module)
        ema_cb = find_ema_callback(trainer)
        if ema_cb is not None:
            install_named_ema_update(ema_cb, module)

    def on_train_batch_end(
        self,
        trainer  : Trainer,
        pl_module: LightningModule,
        outputs  : Any,
        batch    : Any,
        batch_idx: int,
    ) -> None:
        # Rebuild the EMA copy once PAI has registered its lazy buffers
        module    = cast('RFDETRModelModule', pl_module)
        ema_cb    = find_ema_callback(trainer)
        ema_inner = _get_ema_inner_module(ema_cb)
        if ema_inner is None:
            return
        live = {name for name, _ in module.model.named_buffers()}
        ema  = {name for name, _ in ema_inner.model.named_buffers()}
        if live != ema:
            rebuild_ema(cast(RFDETREMACallback, ema_cb), module)

    def on_train_epoch_end(
        self,
        trainer  : Trainer,
        pl_module: LightningModule,
    ) -> None:
        loss = trainer.callback_metrics.get('train/loss')
        if loss is not None:
            gpa.pai_tracker.add_extra_score_without_graphing(
                float(loss),
                'train_loss',
            )

    def on_validation_end(
        self,
        trainer  : Trainer,
        pl_module: LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        module      = cast('RFDETRModelModule', pl_module)
        tracker     = gpa.pai_tracker
        val_metrics = {
            key: float(value)
            for key, value in trainer.callback_metrics.items()
            if key.startswith('val/') and value.numel() == 1
        }
        for key, value in val_metrics.items():
            tracker.add_extra_score_without_graphing(value, key)

        score = float(trainer.callback_metrics[self.monitor])
        mode  = tracker.member_vars['mode']
        added = tracker.member_vars['num_dendrites_added']
        if mode == 'p':
            # Dendrite mode scores on the mean best dendrite correlation
            getter = getattr(tracker, 'get_current_pb_scores', None)
            if getter is not None:
                scores = list(getter().values())
                score  = sum(scores) / len(scores) if scores else 0.0
        if mode == 'n':
            # Keep the optimizer state by parameter name for after a switch
            optimizer = trainer.optimizers[0]
            names     = {id(p): n for n, p in module.model.named_parameters()}
            self.moment_stash = {
                names[id(p)]: optimizer.state[p]
                for group in optimizer.param_groups
                for p in group['params']
                if p in optimizer.state and id(p) in names
            }

        # Copy the scored EMA weights into the live model in place, since PAI
        # checkpoints the model it is handed
        ema_cb     = find_ema_callback(trainer)
        ema_inner  = _get_ema_inner_module(ema_cb)
        live_state = module.model.state_dict()
        saved: dict[str, torch.Tensor] = {}
        if ema_inner is not None:
            ema_state = ema_inner.model.state_dict()
            keys      = {n for n, _ in module.model.named_parameters()}
            keys     |= {
                n for n, _ in module.model.named_buffers()
                if n.endswith(('running_mean', 'running_var'))
            }
            with torch.no_grad():
                for key in keys:
                    saved[key] = live_state[key].clone()
                    live_state[key].copy_(ema_state[key])

        device = module.device
        model, restructured, training_complete = tracker.add_validation_score(
            score,
            module.model,
            force_switch = before_first_switch(),
        )
        if self.observer is not None:
            names  = [m.name for m in tracker.neuron_module_vector]
            best   = tracker.member_vars.get('best_scores', [])
            self.observer(
                {
                    'epoch'            : trainer.current_epoch,
                    'mode'             : mode,
                    'num_dendrites'    : added,
                    'metrics'          : val_metrics,
                    'correlations'     : {
                        name: float(best[i][-1])
                        for i, name in enumerate(names)
                        if i < len(best) and len(best[i]) > 0
                    },
                    'restructured'     : bool(restructured),
                    'training_complete': bool(training_complete),
                }
            )

        if training_complete:
            module.model = model.to(device)
            if ema_cb is not None:
                rebuild_ema(ema_cb, module)
            trainer.should_stop = True
            return
        if not restructured:
            with torch.no_grad():
                for key, tensor in saved.items():
                    live_state[key].copy_(tensor)
            if ema_inner is not None:
                # Re-register the EMA's non-float buffers PAI resized
                ema_buffers = dict(ema_inner.named_buffers())
                for name, live_t in module.named_buffers():
                    ema_t = ema_buffers.get(name)
                    if (
                        ema_t is not None
                        and not ema_t.is_floating_point()
                        and ema_t.shape != live_t.shape
                    ):
                        replace_buffer(ema_inner, name, live_t)
            return

        module.model = model.to(device)
        optimizer    = setup_perforated_optimizer(module)
        # The strategy rebuilds its LightningOptimizer wrappers from this list
        trainer.optimizers                    = [optimizer]
        trainer.strategy.lr_scheduler_configs = []
        if tracker.member_vars['mode'] == 'n' and self.moment_stash:
            # Put the stashed state back on parameters of the same name + shape
            params = dict(module.model.named_parameters())
            held   = {
                id(p) for g in optimizer.param_groups for p in g['params']
            }
            for name, state in self.moment_stash.items():
                param  = params.get(name)
                moment = state.get('exp_avg')
                if param is None or id(param) not in held:
                    continue
                if moment is not None and moment.shape != param.shape:
                    continue
                optimizer.state[param] = state
            self.moment_stash = {}
        if ema_cb is not None:
            rebuild_ema(ema_cb, module)

    def on_train_end(
        self,
        trainer  : Trainer,
        pl_module: LightningModule,
    ) -> None:
        # The whole cleaned LWDETR goes under 'module', since no constructor
        # rebuilds dendrite heads from a state dict
        if not trainer.is_global_zero:
            return
        module = cast('RFDETRModelModule', pl_module)
        clean  = clean_perforated_model(module.model)
        path   = Path(module.train_config.output_dir) / clean_model_name
        torch.save(
            {
                'module'      : clean,
                'model'       : clean.state_dict(),
                'model_config': module.model_config.model_dump(mode='json'),
                'epoch'       : trainer.current_epoch,
            },
            path,
        )

def clean_perforated_model(model: nn.Module) -> nn.Module:
    '''
    Deep copy of the detector with the PAI tracker removed, dendrites kept
    '''
    return cast(nn.Module, upa.prepare_final_model(model))
