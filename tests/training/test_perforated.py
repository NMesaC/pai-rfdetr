# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the PerforatedAI integration switched on by ``TrainConfig.perforate``.

Everything runs without ``perforatedai`` installed except the one test that wraps a real model.
"""

import warnings
from dataclasses import replace

import pytest
import torch
from torch import nn

from rfdetr._namespace import _TC_NAMESPACE_FIELDS
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.perforated import (
    PERFORATION_TARGETS,
    PerforatedAICallback,
    PerforationTarget,
    build_perforated_save_name,
    build_tracked_leaf_ids,
    build_tracked_parameter_ids,
    check_target_ids,
    extract_plain_state,
    get_perforation_target,
    list_perforable_modules,
    output_dimensions_for,
    record_forward_stats,
    resolve_forward_function,
    resolve_perforate_ids,
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
        assert set(PERFORATION_TARGETS) == {"all", "cls_head", "bbox_head", "cls_bbox_head"}

    def test_unknown_target_raises(self):
        """A name outside the registry is rejected with the known names."""
        with pytest.raises(ValueError, match="cls_head"):
            get_perforation_target("decoder")

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
        """Convolutions keep channels at axis 1; linear outputs put neurons last."""
        assert output_dimensions_for("Conv2d", 4) == [-1, 0, -1, -1]
        assert output_dimensions_for("Linear", 4) == [-1, -1, -1, 0]
        assert output_dimensions_for("Linear", 3) == [-1, -1, 0]
        assert output_dimensions_for("MLP", 4) == [-1, -1, -1, 0]


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
