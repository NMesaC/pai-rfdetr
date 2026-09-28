# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the PerforatedAI integration switched on by ``TrainConfig.perforate``.

Everything runs without ``perforatedai`` installed except the one test that wraps a real model.
"""

import copy
import warnings
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest
import torch
from torch import nn

from rfdetr._namespace import _TC_NAMESPACE_FIELDS
from rfdetr.config import PerforationTargetName, RFDETRNanoConfig, RFDETRSegNanoConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.perforated import (
    KEY_MAP_NAME,
    PERFORATION_TARGETS,
    RECIPES,
    PerforatedAICallback,
    PerforationTarget,
    build_perforated_save_name,
    build_tracked_leaf_ids,
    build_tracked_parameter_ids,
    check_target_ids,
    configure_dendrite_copy,
    extract_plain_state,
    get_perforation_target,
    list_perforable_modules,
    output_dimensions_for,
    record_forward_stats,
    resolve_forward_function,
    resolve_perforate_ids,
)
from rfdetr.training.perforated_ema import (
    ema_update_by_name,
    ema_weight_keys,
    restore_weights,
    swap_in_weights,
)
from rfdetr.training.perforated_restructure import (
    PostNormResidual,
    PreNormBranch,
    assert_equivalent,
    invert_key_map,
    remap_state_dict,
    restructure_model,
)

_PAI_FIELDS = [name for name in TrainConfig.model_fields if name.startswith("pai_")]


class _TinyDetector(nn.Module):
    """Module tree with the two decoder heads a target names plus a conv backbone with BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.class_embed = nn.Linear(4, 3)
        self.bbox_embed = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a 4D class output and a 3D box output; ``shared`` runs twice like the two-stage class-head copies."""
        tokens = self.shared(self.shared(self.backbone(x).flatten(2).transpose(1, 2)))
        return self.class_embed(tokens.unsqueeze(0)), self.bbox_embed(tokens)


@pytest.fixture
def model_config():
    """Nano architecture on CPU with no pretrained weights."""
    return RFDETRNanoConfig(pretrain_weights=None, device="cpu", num_classes=3)


@pytest.fixture
def train_config(tmp_path):
    """Factory for a loggerless TrainConfig under ``tmp_path``; call with **overrides."""

    def _make(**overrides):
        defaults = dict(
            dataset_dir=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            epochs=1,
            batch_size=2,
            num_workers=0,
            tensorboard=False,
            wandb=False,
            mlflow=False,
            clearml=False,
            amp_dtype=None,
        )
        defaults.update(overrides)
        return TrainConfig(**defaults)

    return _make


class TestPerforateConfig:
    """``TrainConfig`` exposes the PAI switches and enforces the recipe constraints."""

    def test_defaults_off(self, train_config):
        """A default config does not perforate and every pai_* knob keeps its documented default."""
        tc = train_config()
        assert tc.perforate is False
        assert tc.pai_target == "all"
        assert tc.pai_n_epochs_to_switch == 30
        assert tc.pai_load_folder is None

    def test_recipe_config_accepted(self, train_config):
        """The documented recipe (fp32, eval every epoch, no early stopping) validates."""
        tc = train_config(perforate=True, pai_target="bbox_head", early_stopping=False, eval_interval=1)
        assert tc.perforate is True
        assert tc.pai_target == "bbox_head"

    def test_rejects_eval_interval(self, train_config):
        """PAI counts switches in validations, so validation must run every epoch."""
        with pytest.raises(ValueError, match="eval_interval=1"):
            train_config(perforate=True, eval_interval=2)

    def test_rejects_early_stopping(self, train_config):
        """Early stopping would end the run before PAI decides."""
        with pytest.raises(ValueError, match="early_stopping"):
            train_config(perforate=True, early_stopping=True)

    def test_rejects_multi_device(self, train_config):
        """The callback swaps the trainer's optimizer in place, which is only exercised on one device."""
        with pytest.raises(ValueError, match="single device"):
            train_config(perforate=True, devices=2)

    def test_rejects_load_folder_without_perforate(self, train_config):
        """A PAI system folder cannot load into a model that was never perforated."""
        with pytest.raises(ValueError, match="pai_load_folder requires perforate=True"):
            train_config(pai_load_folder="some_folder")

    def test_rejects_unknown_target(self, train_config):
        """``pai_target`` is a closed literal."""
        with pytest.raises(ValueError):
            train_config(perforate=True, pai_target="decoder")

    def test_warns_on_lr_scheduler(self, train_config):
        """PAI owns the LR schedule, so a configured scheduler is ignored with a warning."""
        with pytest.warns(UserWarning, match="ignores lr_scheduler"):
            train_config(perforate=True, lr_scheduler="cosine")

    def test_warns_on_amp(self, train_config):
        """Mixed precision is warned about, not rejected."""
        with pytest.warns(UserWarning, match="amp_dtype"):
            train_config(perforate=True, amp_dtype="bf16")

    def test_plain_config_emits_no_pai_warning(self, train_config):
        """``perforate=False`` never triggers the PAI warnings, whatever the scheduler."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train_config(lr_scheduler="cosine")

    def test_round_trips_through_model_dump(self, train_config):
        """Every pai_* field serializes and reloads, which training_config.json relies on."""
        tc = train_config(perforate=True, pai_dendrite_lr=1e-5, pai_forward_function="relu", pai_global_candidates=2)
        restored = TrainConfig(**tc.model_dump())
        assert restored.perforate is True
        assert restored.pai_dendrite_lr == pytest.approx(1e-5)
        assert restored.pai_forward_function == "relu"
        assert restored.pai_global_candidates == 2

    def test_pai_fields_stay_out_of_namespace(self):
        """The legacy args namespace never carries PAI settings."""
        assert "perforate" not in _TC_NAMESPACE_FIELDS
        assert not any(name in _TC_NAMESPACE_FIELDS for name in _PAI_FIELDS)


class TestSaveName:
    """``build_perforated_save_name`` labels the PAI system folder."""

    def test_derives_save_name_from_config_class(self, model_config, train_config):
        """Without ``model_name`` the config class name (minus ``Config``) and the target label the PAI folder."""
        name = build_perforated_save_name(model_config, train_config(perforate=True, pai_target="bbox_head"))
        assert name.startswith("rfdetrnano_bbox_head_dendritic_")

    def test_explicit_save_name_wins(self, model_config, train_config):
        """``pai_save_name`` is used verbatim."""
        assert (
            build_perforated_save_name(model_config, train_config(perforate=True, pai_save_name="my_run")) == "my_run"
        )


class TestTargets:
    """The target registry and the parameter bookkeeping built from it."""

    def test_registry_matches_config_literal(self):
        """Every ``pai_target`` literal has a registered target and vice versa."""
        assert set(PERFORATION_TARGETS) == set(get_args(PerforationTargetName))

    def test_unknown_target_raises(self):
        """A name outside the registry is rejected with the known names."""
        with pytest.raises(ValueError, match="cls_head"):
            get_perforation_target("decoder")

    def test_id_list_builds_a_restructuring_target(self):
        """A list of ids is a custom target that restructures first; ids without the leading dot are rejected."""
        target = get_perforation_target([".class_embed", ".transformer.decoder.layers.0.ffn"])
        assert target.name == "custom"
        assert target.restructure
        assert target.perforate_ids == (".class_embed", ".transformer.decoder.layers.0.ffn")
        with pytest.raises(ValueError, match="class_embed"):
            get_perforation_target(["class_embed"])

    def test_block_targets_restructure(self):
        """Block targets exist only after restructuring, the leaf targets do not restructure."""
        assert all(PERFORATION_TARGETS[n].restructure for n in ("backbone_blocks", "decoder_blocks", "all_blocks"))
        assert not any(PERFORATION_TARGETS[n].restructure for n in ("all", "cls_head", "bbox_head", "cls_bbox_head"))

    def test_tracked_parameter_ids_exclude_perforated(self):
        """Parameters under a perforated module are PAI's; every other parameter is registered by name."""
        ids = build_tracked_parameter_ids(_TinyDetector(), get_perforation_target("cls_head"))
        assert ".class_embed.weight" not in ids
        assert ".bbox_embed.0.weight" in ids
        assert ".backbone.0.weight" in ids

    def test_tracked_leaf_ids_skip_perforated_subtree(self):
        """Leaves below a perforated id are skipped; other leaves with parameters are listed."""
        ids = build_tracked_leaf_ids(_TinyDetector(), get_perforation_target("bbox_head"))
        assert ".bbox_embed.0" not in ids
        assert ".class_embed" in ids
        assert ".backbone.1" in ids
        assert ".backbone" not in ids

    def test_check_target_ids_rejects_missing_module(self):
        """A target naming a module the model lacks fails before PAI is touched."""
        target = PerforationTarget(name="bad", perforate_ids=(".missing",))
        with pytest.raises(ValueError, match="missing"):
            check_target_ids(_TinyDetector(), target)

    def test_all_resolves_every_linear_and_conv(self):
        """``all`` wraps each Linear and Conv2d leaf and nothing else."""
        ids = resolve_perforate_ids(_TinyDetector(), get_perforation_target("all"))
        assert set(ids) == {".class_embed", ".bbox_embed.0", ".bbox_embed.1", ".backbone.0", ".shared"}

    def test_skip_ids_leave_a_module_out(self):
        """A skipped module drops out of the wrap set; PAI tracks it, so its parameters are not registered by name."""
        target = replace(get_perforation_target("all"), skip_ids=(".shared",))
        assert ".shared" not in resolve_perforate_ids(_TinyDetector(), target)
        assert set(build_tracked_parameter_ids(_TinyDetector(), target)) == {".backbone.1.weight", ".backbone.1.bias"}

    def test_list_perforable_modules(self):
        """Weight-carrying modules are listed with their type names and a leading dot."""
        modules = list_perforable_modules(_TinyDetector())
        assert modules[".class_embed"] == "Linear"
        assert modules[".backbone.0"] == "Conv2d"
        assert ".bbox_embed" not in modules


class TestOutputDimensions:
    """The neuron axis of every wrapped module comes from one measuring forward."""

    def test_records_rank_and_calls_per_module(self):
        """Heads applied to stacked outputs are 4D, encoder-style outputs 3D, convs 4D; a reused module counts twice."""
        ranks, calls = record_forward_stats(_TinyDetector(), torch.zeros(2, 3, 2, 2))
        assert ranks[".class_embed"] == 4
        assert ranks[".bbox_embed.1"] == 3
        assert ranks[".backbone.0"] == 4
        assert calls[".shared"] == 2
        assert calls[".class_embed"] == 1

    def test_forward_leaves_model_untouched(self):
        """The measuring pass restores BatchNorm statistics and the training flag."""
        model = _TinyDetector().eval()
        before = model.backbone[1].running_mean.clone()
        record_forward_stats(model, torch.ones(2, 3, 2, 2))
        assert torch.equal(model.backbone[1].running_mean, before)
        assert model.backbone[1].num_batches_tracked == 0
        assert not model.training

    def test_axis_by_module_type(self):
        """Convolutions keep channels at axis 1; linear outputs put neurons last; sub-blocks declare their axis."""
        assert output_dimensions_for(nn.Conv2d(3, 4, 1), 4) == [-1, 0, -1, -1]
        assert output_dimensions_for(nn.Linear(4, 4), 4) == [-1, -1, -1, 0]
        assert output_dimensions_for(nn.Linear(4, 4), 3) == [-1, -1, 0]
        assert output_dimensions_for(nn.Sequential(nn.Linear(4, 4)), 4) == [-1, -1, -1, 0]
        branch = PreNormBranch(nn.LayerNorm(4), nn.Linear(4, 4))
        assert output_dimensions_for(branch, 3) == [-1, -1, 0]
        block = PostNormResidual(nn.Linear(4, 4), nn.Dropout(0.0), nn.LayerNorm(4))
        assert output_dimensions_for(block, 3) == [-1, -1, 0]

    def test_hooks_named_ids_of_any_type(self):
        """Ids passed explicitly are measured even when their type is not in the perforable list."""
        ranks, calls = record_forward_stats(_TinyDetector(), torch.zeros(2, 3, 2, 2), module_ids=(".backbone",))
        assert ranks[".backbone"] == 4
        assert calls[".backbone"] == 1


class TestForwardFunction:
    """``pai_forward_function`` resolves names on torch, ``identity``, or a callable."""

    def test_torch_name(self):
        """A plain torch function name resolves to that function."""
        assert resolve_forward_function("relu") is torch.relu

    def test_functional_name(self):
        """A torch.nn.functional name resolves too."""
        assert resolve_forward_function("gelu") is torch.nn.functional.gelu

    def test_identity_and_callable(self):
        """``identity`` returns its input unchanged and a callable is used as is."""
        x = torch.ones(2)
        assert resolve_forward_function("identity")(x) is x
        assert resolve_forward_function(torch.tanh) is torch.tanh

    def test_unknown_name_raises(self):
        """A name torch does not have is rejected."""
        with pytest.raises(ValueError, match="pai_forward_function"):
            resolve_forward_function("not_a_torch_function")

    def test_fraction_is_bounded(self, train_config):
        """The auto warmup fraction must be a proper fraction of an epoch."""
        with pytest.raises(ValueError):
            train_config(perforate=True, pai_initial_correlation_fraction=1.0)


class TestExtractPlainState:
    """``extract_plain_state`` turns a PAI stage state dict back into plain ``LWDETR`` keys."""

    def test_unwraps_main_module_and_drops_dendrites(self):
        """Only main_module tensors of a wrapped module survive, renamed; unwrapped tensors pass through."""
        state = {
            "class_embed.main_module.weight": torch.zeros(3, 4),
            "class_embed.main_module.bias": torch.zeros(3),
            "class_embed.dendrite_module.0.weight": torch.zeros(3, 4),
            "class_embed.dendrites_to_neurons.0": torch.zeros(3),
            "class_embed.this_node_index": torch.tensor(3),
            "bbox_embed.layers.0.weight": torch.zeros(4, 4),
            "tracker_string": torch.zeros(7),
        }
        plain = extract_plain_state(state)
        assert set(plain) == {"class_embed.weight", "class_embed.bias", "bbox_embed.layers.0.weight"}
        assert plain["class_embed.weight"] is state["class_embed.main_module.weight"]

    def test_plain_state_passes_through(self):
        """A state dict without PAI keys is returned unchanged."""
        state = {"class_embed.weight": torch.zeros(3, 4), "backbone.0.weight": torch.zeros(4, 3, 1, 1)}
        plain = extract_plain_state(state)
        assert plain.keys() == state.keys()
        assert all(plain[key] is state[key] for key in state)


class TestBuildTrainerPerforate:
    """``build_trainer`` appends ``PerforatedAICallback`` last, only for a perforated training run."""

    def test_absent_by_default(self, model_config, train_config):
        """A plain config wires no PAI callback."""
        trainer = build_trainer(train_config(), model_config)
        assert not any(isinstance(cb, PerforatedAICallback) for cb in trainer.callbacks)

    def test_appended_after_coco_eval(self, model_config, train_config):
        """With ``perforate=True`` the callback follows COCOEvalCallback so it sees the logged metric."""
        trainer = build_trainer(train_config(perforate=True), model_config)
        types = [type(cb) for cb in trainer.callbacks]
        assert PerforatedAICallback in types
        assert types.index(PerforatedAICallback) > types.index(COCOEvalCallback)

    def test_monitor_follows_best_model_metric(self, model_config, train_config):
        """The callback scores the same task metric BestModelCallback monitors."""
        trainer = build_trainer(train_config(perforate=True, best_model_metric="mar"), model_config)
        callback = next(cb for cb in trainer.callbacks if isinstance(cb, PerforatedAICallback))
        assert callback.monitor.startswith("val/")
        assert "mAR" in callback.monitor

    def test_never_in_eval_mode(self, model_config, train_config):
        """The evaluation-only trainer keeps the metric callback and nothing PAI."""
        trainer = build_trainer(train_config(perforate=True), model_config, include_training_callbacks=False)
        assert not any(isinstance(cb, PerforatedAICallback) for cb in trainer.callbacks)


class TestModuleRefusals:
    """``RFDETRModelModule`` refuses combinations the integration does not support before importing PAI."""

    def test_refuses_compile(self, train_config):
        """torch.compile over PAI modules is untested."""
        from rfdetr.training import RFDETRModelModule

        mc = RFDETRNanoConfig(pretrain_weights=None, device="cpu", num_classes=3, compile=True)
        with pytest.raises(ValueError, match="compile=True"):
            RFDETRModelModule(mc, train_config(perforate=True))

    def test_missing_library_names_the_extra(self, model_config, train_config):
        """Without perforatedai installed the module fails with the install hint."""
        pytest.importorskip("rfdetr")
        try:
            import perforatedai  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            pytest.skip("perforatedai is installed; the ImportError path is unreachable")
        from rfdetr.training import RFDETRModelModule

        with pytest.raises(ImportError, match=r"rfdetr\[pai\]"):
            RFDETRModelModule(model_config, train_config(perforate=True))


@pytest.mark.integration
class TestPerforatedModule:
    """Wrapping a real Nano through the module (needs ``perforatedai``)."""

    def test_wraps_every_linear_and_conv(self, model_config, train_config):
        """With the default target every Linear and Conv2d becomes a PAINeuronModule with its own neuron axis."""
        pytest.importorskip("perforatedai")
        from rfdetr.training import RFDETRModelModule

        module = RFDETRModelModule(model_config, train_config(perforate=True, pai_testing_dendrite_capacity=True))
        wrapped = {m.name: m for m in module.model.modules() if type(m).__name__ == "PAINeuronModule"}
        # 155 Linear + 9 Conv2d minus the 13 two-stage class-head copies RF-DETR calls twice per forward.
        assert len(wrapped) == 151
        assert ".transformer.enc_out_class_embed.0" not in wrapped
        assert wrapped[".class_embed"].this_output_dimensions.tolist() == [-1, -1, -1, 0]
        assert wrapped[".transformer.decoder.layers.0.linear1"].this_output_dimensions.tolist() == [-1, -1, 0]
        conv = ".backbone.0.encoder.encoder.embeddings.patch_embeddings.projection"
        assert wrapped[conv].this_output_dimensions.tolist() == [-1, 0, -1, -1]

    def test_wraps_class_embed_only(self, model_config, train_config):
        """The head target wraps exactly the classification head."""
        pytest.importorskip("perforatedai")
        from rfdetr.training import RFDETRModelModule

        tc = train_config(perforate=True, pai_target="cls_head", pai_testing_dendrite_capacity=True)
        module = RFDETRModelModule(model_config, tc)
        wrapped = [m.name for m in module.model.modules() if type(m).__name__ == "PAINeuronModule"]
        assert wrapped == [".class_embed"]

    def test_wraps_decoder_blocks(self, model_config, train_config):
        """The decoder target restructures every decoder layer into three post-norm blocks and wraps those."""
        pytest.importorskip("perforatedai")
        from rfdetr.training import RFDETRModelModule

        tc = train_config(perforate=True, pai_target="decoder_blocks", pai_testing_dendrite_capacity=True)
        module = RFDETRModelModule(model_config, tc)
        wrapped = {m.name: m for m in module.model.modules() if type(m).__name__ == "PAINeuronModule"}
        assert len(wrapped) == 3 * model_config.dec_layers
        assert set(wrapped) == {
            f".transformer.decoder.layers.{i}.{part}"
            for i in range(model_config.dec_layers)
            for part in ("self_attn", "cross_attn", "ffn")
        }
        assert wrapped[".transformer.decoder.layers.0.ffn"].this_output_dimensions.tolist() == [-1, -1, 0]
        assert (Path(tc.output_dir) / KEY_MAP_NAME).is_file()

    def test_wraps_backbone_blocks(self, model_config, train_config):
        """The backbone target wraps two pre-norm branches per DINOv2 layer."""
        pytest.importorskip("perforatedai")
        from rfdetr.training import RFDETRModelModule

        tc = train_config(perforate=True, pai_target="backbone_blocks", pai_testing_dendrite_capacity=True)
        module = RFDETRModelModule(model_config, tc)
        wrapped = sorted(m.name for m in module.model.modules() if type(m).__name__ == "PAINeuronModule")
        assert wrapped
        assert all(name.endswith((".attention", ".mlp")) for name in wrapped)
        assert len(wrapped) % 2 == 0


class _TinyPreNormBlock(nn.Module):
    """``x + fc(norm(x))`` with the norm, branch, and residual as sibling calls, like a transformer block."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` plus the normalized branch."""
        return x + self.fc(self.norm(x))


class _RestructuredPreNormBlock(nn.Module):
    """``_TinyPreNormBlock`` rebuilt as a ``PreNormBranch`` with the residual add outside."""

    def __init__(self, block: _TinyPreNormBlock) -> None:
        super().__init__()
        self.branch = PreNormBranch(block.norm, block.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` plus the branch."""
        return x + self.branch(x)


class _TinyStack(nn.Module):
    """Two tiny blocks in a ModuleList plus a head, with ``head`` also reachable under a second name (tied)."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyPreNormBlock(), _TinyPreNormBlock()])
        self.head = nn.Linear(4, 2)
        self.alias = self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks and the head on ``[B, N, 4]`` tokens."""
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TestRestructure:
    """The generic walker rebuilds blocks from recipes and keeps the state dict loadable."""

    def test_replaces_blocks_and_maps_keys(self):
        """Every block is rebuilt, moved keys are mapped, unchanged and tied keys map to themselves."""
        model = _TinyStack()
        result = restructure_model(model, {_TinyPreNormBlock: _RestructuredPreNormBlock})
        assert result.replaced == ("blocks.0", "blocks.1")
        assert all(isinstance(b, _RestructuredPreNormBlock) for b in model.blocks)
        assert result.key_map["blocks.0.norm.weight"] == "blocks.0.branch.norm.weight"
        assert result.key_map["blocks.1.fc.bias"] == "blocks.1.branch.fn.bias"
        assert result.key_map["head.weight"] == "head.weight"
        assert result.key_map["alias.weight"] == "alias.weight"
        assert set(result.key_map.values()) == set(model.state_dict())

    def test_restructured_forward_matches(self):
        """The rebuilt model computes the same function in eval and train mode."""
        model = _TinyStack()
        reference = copy.deepcopy(model)
        restructure_model(model, {_TinyPreNormBlock: _RestructuredPreNormBlock})
        assert_equivalent(reference, model, [torch.randn(2, 3, 4)])

    def test_key_map_round_trips_state_dicts(self):
        """A plain state dict remapped forward loads into the restructured model, and the inverse maps back."""
        model = _TinyStack()
        plain_state = {k: v.clone() for k, v in model.state_dict().items()}
        result = restructure_model(model, {_TinyPreNormBlock: _RestructuredPreNormBlock})
        model.load_state_dict(remap_state_dict(plain_state, result.key_map))
        back = remap_state_dict(model.state_dict(), invert_key_map(result.key_map))
        assert set(back) == set(plain_state)
        assert all(torch.equal(back[k], plain_state[k]) for k in plain_state)

    def test_recipe_dropping_a_tensor_is_rejected(self):
        """A recipe that loses a parameter fails instead of silently shrinking the model."""

        def drop_norm(block: _TinyPreNormBlock) -> nn.Module:
            """Return only the block's Linear, losing its norm."""
            return block.fc

        with pytest.raises(ValueError, match="dropped"):
            restructure_model(_TinyStack(), {_TinyPreNormBlock: drop_norm})

    def test_in_place_recipe_is_applied_once(self):
        """A recipe that mutates its module and returns it is not applied again on the restarted walk."""
        calls: list[int] = []

        def tag(block: _TinyPreNormBlock) -> nn.Module:
            """Record the call and return the same block."""
            calls.append(1)
            return block

        result = restructure_model(_TinyStack(), {_TinyPreNormBlock: tag})
        assert len(calls) == 2
        assert result.replaced == ("blocks.0", "blocks.1")


class TestSubBlocks:
    """The sub-block modules recipes assemble."""

    def test_pre_norm_branch_selects_and_scales(self):
        """The branch normalizes, applies the body, indexes a tuple output, and scales."""
        norm = nn.LayerNorm(4)
        fn = nn.Linear(4, 4)

        class _Tupled(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
                return fn(x), None

        scale = nn.Linear(4, 4, bias=False)
        branch = PreNormBranch(norm, _Tupled(), scale, select=0)
        x = torch.randn(2, 3, 4)
        assert torch.allclose(branch(x), scale(fn(norm(x))))

    def test_post_norm_residual_keeps_skip_until_configured_as_dendrite(self):
        """The neuron computes ``norm(x + fn(x))``, a configured copy drops the skip, deepcopy keeps the flag."""
        block = PostNormResidual(nn.Linear(4, 4), nn.Dropout(0.0), nn.LayerNorm(4))
        x = torch.randn(2, 3, 4)
        assert torch.allclose(block(x), block.norm(x + block.fn(x)))
        dendrite = copy.deepcopy(block)
        assert dendrite.skip
        dendrite.configure_as_dendrite()
        assert torch.allclose(dendrite(x), dendrite.norm(dendrite.fn(x)))
        assert block.skip
        assert "skip" not in dendrite.state_dict()

    def test_dendrite_copy_inherits_norm_and_scale_parameters(self):
        """After PAI randomizes a copy, norm and scale-only parameters return from the neuron, weights stay random."""
        main = PostNormResidual(nn.Linear(4, 4), nn.Dropout(0.0), nn.LayerNorm(4))
        with torch.no_grad():
            main.norm.weight.fill_(2.0)
            main.norm.bias.fill_(0.5)
        dendrite = copy.deepcopy(main)
        with torch.no_grad():
            for param in dendrite.parameters():
                param.normal_()
        configure_dendrite_copy(dendrite, main)
        assert not dendrite.skip
        assert torch.equal(dendrite.norm.weight, main.norm.weight)
        assert torch.equal(dendrite.norm.bias, main.norm.bias)
        assert not torch.equal(dendrite.fn.weight, main.fn.weight)


class TestEMAHelpers:
    """The framework-free EMA update and weight swap."""

    def test_update_by_name_averages_mirrors_and_resizes(self):
        """Floats are averaged, mirrored names and ints are copied, and a resized int buffer is re-registered."""
        live = nn.Linear(2, 2)
        live.register_buffer("index", torch.tensor([1, 2, 3]))
        live.register_buffer("flag", torch.tensor(1.0))
        ema = copy.deepcopy(live)
        with torch.no_grad():
            live.weight.fill_(1.0)
            ema.weight.fill_(0.0)
            live.flag.fill_(5.0)
        live.index = torch.tensor([7, 8])
        ema_update_by_name(ema, live, decay=0.75, mirror={"flag"})
        assert torch.allclose(ema.weight, torch.full((2, 2), 0.25))
        assert torch.equal(ema.flag, torch.tensor(5.0))
        assert torch.equal(ema.index, torch.tensor([7, 8]))

    def test_first_update_copies_everything(self):
        """The seeding update copies every tensor instead of averaging."""
        live = nn.Linear(2, 2)
        ema = copy.deepcopy(live)
        with torch.no_grad():
            live.weight.fill_(3.0)
        ema_update_by_name(ema, live, decay=0.99, mirror=set(), first_update=True)
        assert torch.equal(ema.weight, live.weight)

    def test_swap_and_restore_round_trip(self):
        """Swapping the EMA weights in and restoring leaves the live model as it was."""
        live = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        ema = copy.deepcopy(live)
        with torch.no_grad():
            ema[0].weight.fill_(9.0)
            ema[1].running_mean.fill_(4.0)
        before = {k: v.clone() for k, v in live.state_dict().items()}
        keys = ema_weight_keys(live)
        assert "1.running_mean" in keys
        assert "1.num_batches_tracked" not in keys
        saved = swap_in_weights(live, ema, keys)
        assert torch.equal(live[0].weight, ema[0].weight)
        assert torch.equal(live[1].running_mean, ema[1].running_mean)
        restore_weights(live, saved)
        assert all(torch.equal(live.state_dict()[k], v) for k, v in before.items())


class TestRestructuredDetector:
    """The RF-DETR recipes reproduce the stock detector."""

    def test_nano_restructure_is_equivalent(self, model_config):
        """Restructured Nano matches the original in eval and train mode and exposes the block ids."""
        from rfdetr.models.lwdetr import build_model_from_config

        torch.manual_seed(0)
        model = build_model_from_config(model_config)
        num_layers = sum(1 for m in model.modules() if type(m).__name__ == "WindowedDinov2WithRegistersLayer")
        reference = copy.deepcopy(model)
        result = restructure_model(model, RECIPES)
        assert "transformer.decoder.layers.0.ffn.fn.linear1.weight" in result.key_map.values()
        assert result.key_map["transformer.enc_output_norm.0.weight"] == "transformer.enc_output.0.model.1.weight"
        backbone = resolve_perforate_ids(model, get_perforation_target("backbone_blocks"))
        decoder = resolve_perforate_ids(model, get_perforation_target("decoder_blocks"))
        enc_output = resolve_perforate_ids(model, get_perforation_target("enc_output"))
        assert len(backbone) == 2 * num_layers
        assert len(decoder) == 3 * model_config.dec_layers
        assert len(enc_output) == model_config.group_detr
        resolution = model_config.resolution
        assert_equivalent(reference, model, [torch.randn(2, 3, resolution, resolution)], atol=1e-4, rtol=1e-3)

    def test_seg_nano_restructure_is_equivalent(self):
        """The segmentation recipes reproduce the stock segmentation head."""
        from rfdetr.models.lwdetr import build_model_from_config

        torch.manual_seed(0)
        model_config = RFDETRSegNanoConfig(pretrain_weights=None, device="cpu", num_classes=3)
        model = build_model_from_config(model_config)
        reference = copy.deepcopy(model)
        result = restructure_model(model, RECIPES)
        assert any("segmentation_head" in name and ".branch." in name for name in result.key_map.values())
        segmentation = resolve_perforate_ids(model, get_perforation_target("segmentation_blocks"))
        assert segmentation
        resolution = model_config.resolution
        assert_equivalent(reference, model, [torch.randn(2, 3, resolution, resolution)], atol=1e-4, rtol=1e-3)
