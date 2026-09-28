# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Sequential sweep of PerforatedAI dendrite settings, every run from one plateaued checkpoint.

Each grid point is one ``perforate=True`` run through the custom training API, so the sweep can attach an observer to
``PerforatedAICallback``. PAI keeps its state in process globals, so ``--all`` starts one subprocess per point. The
config is the Lightning CLI YAML the arms train with. ``--start-weights`` is a plain checkpoint, or a PAI stage file
that is converted once with ``extract_start_weights``.

Usage::

    python scripts/perforated_sweep.py configs/rfdetr_nano_pills_pai.yaml --list
    python scripts/perforated_sweep.py configs/rfdetr_nano_pills_pai.yaml --stage 1 --all \\
        --start-weights <system folder>/best_model_beforeSwitch_0.pt --training-config output/<run>/training_config.json

A finished point writes ``sweep_done.json`` into its output folder and is skipped on restart.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import wandb
import yaml

import rfdetr
from rfdetr.detr import _prepare_run_config, _save_training_config
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
from rfdetr.training.perforated import KEY_MAP_NAME, PerforatedAICallback, extract_start_weights
from rfdetr.utilities.distributed import _is_launcher_main_process
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Metric the lift is measured on (the EMA score PAI scores neuron phases on).
LIFT_METRIC = "val/ema_mAP_50_95"

#: Validation metrics copied to W&B every epoch, beside the lift and the dendrite correlations.
LOGGED_METRICS = ("val/ema_mAP_50_95", "val/ema_mAP_75")

#: Marker a finished run writes into its output folder.
DONE_MARKER = "sweep_done.json"

# Stage 1: targets x forward functions x init, forced switch at the first validation, stop after the second dendrite.
STAGE1_TARGETS = ("cls_bbox_head", "bbox_head")
STAGE1_FORWARD_FUNCTIONS = ("sigmoid", "identity", "relu")
STAGE1_INIT_BY_MAIN = (False, True)
# More than one candidate is unusable on perforatedbp 3.2.8 (stacked backward, OOM or a failed deepcopy on save).
STAGE1_GLOBAL_CANDIDATES = (1,)

# Stage 2: the three best stage 1 points x candidate init multiplier x dendrite lr. The centre (0.01, 1e-4) of each
# base repeats its stage 1 run and gives the seed noise.
STAGE2_BASES = (
    ("bbox_head_relu_rand", "bbox_head", "relu", False),
    ("cls_bbox_head_identity_main", "cls_bbox_head", "identity", True),
    ("bbox_head_identity_main", "bbox_head", "identity", True),
)
STAGE2_CANDIDATE_INIT_MULTS = (0.001, 0.01, 0.1)
STAGE2_DENDRITE_LRS = (1e-5, 1e-4, 1e-3)


@dataclass(frozen=True)
class SweepConfig:
    """One point of the sweep.

    Args:
        index: Position in the grid, 0 is the control.
        name: Run name, also the output folder and the W&B run name.
        target: ``TrainConfig.pai_target``.
        forward_function: ``TrainConfig.pai_forward_function``.
        global_candidates: ``TrainConfig.pai_global_candidates``.
        candidate_init_by_main: ``TrainConfig.pai_candidate_init_by_main``.
        candidate_init_mult: ``TrainConfig.pai_candidate_init_mult``.
        dendrite_lr: ``TrainConfig.pai_dendrite_lr``.
    """

    index: int
    name: str
    target: str
    forward_function: str | None = None
    global_candidates: int | None = None
    candidate_init_by_main: bool | None = None
    candidate_init_mult: float | None = None
    dendrite_lr: float | None = None

    def train_kwargs(self) -> dict[str, Any]:
        """Return the ``TrainConfig`` fields this point sets on top of the shared config."""
        return {
            "perforate": True,
            "pai_target": self.target,
            "pai_n_epochs_to_switch": 30,
            "pai_max_dendrites": 2,
            "pai_max_dendrite_tries": 1,
            # PAI rejects a save_name with a slash.
            "pai_save_name": f"pai_{self.name}",
            "pai_force_first_switch": True,
            "pai_dendrite_lr": self.dendrite_lr,
            "pai_candidate_init_mult": self.candidate_init_mult,
            "pai_candidate_init_by_main": self.candidate_init_by_main,
            "pai_global_candidates": self.global_candidates,
            "pai_forward_function": self.forward_function,
        }


class SweepLogger:
    """Turn ``PerforatedAICallback`` observer records into the W&B stream of one sweep run.

    The baseline is the first validation's score, which with ``pai_force_first_switch`` is the loaded model itself.

    Args:
        config: Point of the sweep this run is.
        output_dir: Run output folder, where the done marker is written.
        project: W&B project.
        entity: W&B entity.
        enabled: ``False`` logs nothing to W&B and still writes the marker.
    """

    def __init__(self, config: SweepConfig, output_dir: Path, project: str, entity: str, enabled: bool) -> None:
        self.config = config
        self.output_dir = output_dir
        self.enabled = enabled
        self.baseline: float | None = None
        self.best_lift_d1 = float("-inf")
        self.best_lift = float("-inf")
        self.best_corr: dict[str, float] = {}
        self.last_epoch = -1
        self.run = None
        if enabled:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=config.name,
                config=asdict(config),
                dir=str(output_dir),
            )

    def observe(self, record: dict[str, Any]) -> None:
        """Log one scored validation.

        Args:
            record: Record ``PerforatedAICallback`` builds after each ``add_validation_score``.
        """
        metrics = record["metrics"]
        score = metrics.get(LIFT_METRIC)
        if score is None:
            raise RuntimeError(f"{LIFT_METRIC} missing from the validation metrics; the sweep cannot measure a lift.")
        if self.baseline is None:
            self.baseline = score
        lift = score - self.baseline
        self.best_lift = max(self.best_lift, lift)
        if record["mode"] == "n" and record["num_dendrites"] == 1:
            self.best_lift_d1 = max(self.best_lift_d1, lift)
        for name, corr in record["correlations"].items():
            self.best_corr[name] = max(self.best_corr.get(name, corr), corr)
        self.last_epoch = record["epoch"]

        row: dict[str, Any] = {
            "epoch": record["epoch"],
            "pai/mode": 1 if record["mode"] == "p" else 0,
            "pai/num_dendrites": record["num_dendrites"],
            "sweep/lift": lift,
        }
        for key in LOGGED_METRICS:
            if key in metrics:
                row[key] = metrics[key]
        if record["correlations"]:
            values = list(record["correlations"].values())
            row["pai/corr_mean"] = sum(values) / len(values)
            for name, corr in record["correlations"].items():
                row[f"pai/corr{name}"] = corr
        logger.info(
            "Sweep %s epoch %d mode %s dendrites %d lift %+.4f corr %s",
            self.config.name,
            record["epoch"],
            record["mode"],
            record["num_dendrites"],
            lift,
            {k: round(v, 4) for k, v in record["correlations"].items()},
        )
        if self.run is not None:
            self.run.log(row, step=record["epoch"])

    def finish(self, reference_score: float) -> dict[str, Any]:
        """Write the run summary to W&B and to the done marker.

        Args:
            reference_score: Score the start checkpoint was extracted at, so the summary shows how far the baseline
                drifted from it.

        Returns:
            The summary written to the marker.
        """
        summary = {
            "config": asdict(self.config),
            "baseline": self.baseline,
            "reference_score": reference_score,
            "baseline_gap": None if self.baseline is None else self.baseline - reference_score,
            "best_lift_d1": None if self.best_lift_d1 == float("-inf") else self.best_lift_d1,
            "best_lift": None if self.best_lift == float("-inf") else self.best_lift,
            "best_corr": self.best_corr,
            "last_epoch": self.last_epoch,
        }
        if self.run is not None:
            for key, value in summary.items():
                if key != "config":
                    self.run.summary[f"sweep/{key}"] = value
            self.run.finish()
        with open(self.output_dir / DONE_MARKER, "w") as handle:
            json.dump(summary, handle, indent=2)
        return summary


def build_stage1_grid() -> list[SweepConfig]:
    """Build the stage 1 grid: the control (the checkpoint's own recipe) first, then the full factorial."""
    grid = [SweepConfig(index=0, name="s1_00_control_cls_head", target="cls_head")]
    axes = itertools.product(STAGE1_TARGETS, STAGE1_FORWARD_FUNCTIONS, STAGE1_GLOBAL_CANDIDATES, STAGE1_INIT_BY_MAIN)
    for target, forward, candidates, by_main in axes:
        index = len(grid)
        init = "main" if by_main else "rand"
        grid.append(
            SweepConfig(
                index=index,
                name=f"s1_{index:02d}_{target}_{forward}_c{candidates}_{init}",
                target=target,
                forward_function=forward,
                global_candidates=candidates,
                candidate_init_by_main=by_main,
            )
        )
    return grid


def build_stage2_grid() -> list[SweepConfig]:
    """Build the stage 2 grid: every base crossed with both dendrite knobs, ordered base by base."""
    grid: list[SweepConfig] = []
    axes = itertools.product(STAGE2_BASES, STAGE2_CANDIDATE_INIT_MULTS, STAGE2_DENDRITE_LRS)
    for (tag, target, forward, by_main), init_mult, dendrite_lr in axes:
        index = len(grid)
        grid.append(
            SweepConfig(
                index=index,
                name=f"s2_{index:02d}_{tag}_im{init_mult:g}_lr{dendrite_lr:g}",
                target=target,
                forward_function=forward,
                global_candidates=1,
                candidate_init_by_main=by_main,
                candidate_init_mult=init_mult,
                dendrite_lr=dendrite_lr,
            )
        )
    return grid


def build_grid(stage: int) -> list[SweepConfig]:
    """Build the grid of one sweep stage (1: targets and forward functions, 2: dendrite knobs on the winners)."""
    if stage == 1:
        return build_stage1_grid()
    if stage == 2:
        return build_stage2_grid()
    raise ValueError(f"Unknown sweep stage {stage}; choose 1 or 2.")


def load_sweep_config(path: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Read the model class and the shared train arguments from a Lightning CLI config.

    Args:
        path: YAML in the ``rfdetr fit --config`` layout, e.g. ``configs/rfdetr_nano_pills_pai.yaml``.

    Returns:
        ``(model_name, model_kwargs, train_kwargs)``: the detector class name in the ``rfdetr`` package (the config
        class name minus ``Config``), the ``model_config.init_args``, and the ``train_config.init_args``.

    Raises:
        ValueError: If the file lacks the ``model.model_config`` / ``model.train_config`` sections.
    """
    with open(path) as handle:
        spec = yaml.safe_load(handle)
    try:
        model_section = spec["model"]["model_config"]
        train_section = spec["model"]["train_config"]
    except (KeyError, TypeError):
        raise ValueError(
            f"{path} is not a Lightning CLI config with model.model_config and model.train_config"
        ) from None
    model_name = str(model_section["class_path"]).rsplit(".", 1)[-1].removesuffix("Config")
    return model_name, dict(model_section.get("init_args") or {}), dict(train_section.get("init_args") or {})


def fit_sweep_point(
    model_name: str,
    model_kwargs: dict[str, Any],
    train_kwargs: dict[str, Any],
    pretrain_weights: str,
    observer: Any,
) -> None:
    """Train one perforated detector (``RFDETR.train()`` up to ``trainer.fit``) and stream PAI records to ``observer``.

    Args:
        model_name: Detector class name in the ``rfdetr`` package, e.g. ``"RFDETRNano"``.
        model_kwargs: ``ModelConfig`` overrides from the config file, e.g. ``num_classes``.
        train_kwargs: Keyword arguments of ``RFDETR.train()``, including the ``perforate`` / ``pai_*`` fields.
        pretrain_weights: Local checkpoint every sweep run starts from.
        observer: Callable handed to ``PerforatedAICallback.set_observer``.
    """
    detector = getattr(rfdetr, model_name)(pretrain_weights=pretrain_weights, **model_kwargs)
    config, accelerator, devices = _prepare_run_config(detector, **train_kwargs)
    detector._align_num_classes_from_dataset(str(config.dataset_dir))
    if _is_launcher_main_process():
        _save_training_config(config, detector.model_config, None)

    module = RFDETRModelModule(detector.model_config, config)
    datamodule = RFDETRDataModule(detector.model_config, config)
    trainer_kwargs: dict[str, Any] = {"accelerator": accelerator}
    if devices is not None:
        trainer_kwargs["devices"] = devices
    trainer = build_trainer(config, detector.model_config, **trainer_kwargs)
    callback = next(cb for cb in trainer.callbacks if isinstance(cb, PerforatedAICallback))
    callback.set_observer(observer)
    trainer.fit(module, datamodule)
    logger.info("Training finished. Outputs are in %s", config.output_dir)


def run_one(args: argparse.Namespace, point: SweepConfig) -> dict[str, Any]:
    """Train one point of the sweep in this process and return its summary."""
    output_dir = Path(args.output_root) / point.name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name, model_kwargs, train_kwargs = load_sweep_config(args.config)
    train_kwargs["run"] = point.name
    train_kwargs["output_dir"] = str(output_dir)
    train_kwargs["wandb"] = False
    if args.dataset_dir:
        train_kwargs["dataset_dir"] = args.dataset_dir
    if args.epochs:
        train_kwargs["epochs"] = args.epochs
    train_kwargs.update(point.train_kwargs())

    sweep_logger = SweepLogger(
        config=point,
        output_dir=output_dir,
        project=args.project,
        entity=args.entity,
        enabled=not args.no_wandb,
    )
    fit_sweep_point(model_name, model_kwargs, train_kwargs, args.start_weights, sweep_logger.observe)
    return sweep_logger.finish(args.reference_score)


def run_all(args: argparse.Namespace, grid: list[SweepConfig]) -> None:
    """Run every unfinished grid point in order, one subprocess each."""
    passthrough = [
        "--stage",
        str(args.stage),
        "--start-weights",
        args.start_weights,
        "--reference-score",
        str(args.reference_score),
        "--output-root",
        args.output_root,
        "--project",
        args.project,
        "--entity",
        args.entity,
    ]
    if args.dataset_dir:
        passthrough += ["--dataset-dir", args.dataset_dir]
    if args.epochs:
        passthrough += ["--epochs", str(args.epochs)]
    if args.no_wandb:
        passthrough += ["--no-wandb"]

    script = str(Path(__file__).resolve())
    failed = []
    for point in grid[args.from_index :]:
        marker = Path(args.output_root) / point.name / DONE_MARKER
        if marker.is_file():
            logger.info("Sweep skipping %s, done marker exists", point.name)
            continue
        command = [sys.executable, script, args.config, "--index", str(point.index)] + passthrough
        logger.info("Sweep starting %s: %s", point.name, " ".join(command))
        # Closed stdin makes a pdb prompt PAI opens on an error read EOF instead of holding the sequence.
        result = subprocess.run(command, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            logger.error("Sweep point %s exited with %d, moving on", point.name, result.returncode)
            failed.append(point.name)
    if failed:
        logger.error("Sweep finished with failed points: %s", failed)
    else:
        logger.info("Sweep finished, every point wrote its done marker")


def resolve_start_weights(start_weights: str, training_config: str) -> str:
    """Return a plain checkpoint path, converting a PAI stage file once when that is what was given.

    Args:
        start_weights: Plain ``.pth`` checkpoint, or a PAI stage file ending in ``.pt``.
        training_config: ``training_config.json`` whose ``model_config`` becomes the checkpoint ``args``, or empty.

    Returns:
        Path of a checkpoint ``RFDETR(pretrain_weights=...)`` can load.
    """
    start = Path(start_weights)
    if not start.is_file():
        raise FileNotFoundError(f"No checkpoint at {start}")
    if start.suffix != ".pt":
        return str(start)
    plain = start.with_name(f"{start.stem}_plain.pth")
    if plain.is_file():
        logger.info("Sweep reusing %s extracted earlier from %s", plain, start)
        return str(plain)
    model_args: dict[str, Any] = {}
    key_map: Path | None = None
    if training_config:
        with open(training_config) as handle:
            model_args = dict(json.load(handle).get("model_config", {}))
        # A run that restructured the detector into sub-blocks wrote its key map beside training_config.json.
        candidate = Path(training_config).parent / KEY_MAP_NAME
        key_map = candidate if candidate.is_file() else None
    return str(extract_start_weights(start, plain, model_args, key_map_path=key_map))


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments for the sweep."""
    parser = argparse.ArgumentParser(description="Sweep dendrite settings from one plateaued checkpoint")
    parser.add_argument("config", type=str, help="Lightning CLI YAML the arms train with (configs/*_pai.yaml).")
    parser.add_argument(
        "--start-weights",
        type=str,
        default="",
        help="Checkpoint every run starts from: a plain .pth, or a PAI stage .pt that is converted once beside itself.",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="",
        help="training_config.json of the run a PAI stage came from; its model_config becomes the checkpoint args.",
    )
    parser.add_argument(
        "--reference-score", type=float, default=0.7078, help="EMA mAP50:95 the start checkpoint scored."
    )
    parser.add_argument("--stage", type=int, default=1, help="Sweep stage: 1 (targets) or 2 (dendrite knobs).")
    parser.add_argument("--index", type=int, default=-1, help="Run this one grid index in this process.")
    parser.add_argument("--all", action="store_true", help="Run every grid index without a done marker, in order.")
    parser.add_argument("--from-index", type=int, default=0, help="First grid index --all considers.")
    parser.add_argument("--list", action="store_true", help="Print the grid and exit.")
    parser.add_argument("--output-root", type=str, default="output/cfg_search", help="One subfolder per run.")
    parser.add_argument("--dataset-dir", type=str, default="", help="Dataset root, overriding the config.")
    parser.add_argument("--epochs", type=int, default=0, help="Epoch ceiling for a smoke run; 0 keeps the config.")
    parser.add_argument("--project", type=str, default="cfg_search_rf_detr", help="W&B project.")
    parser.add_argument("--entity", type=str, default="perforated-ai", help="W&B entity.")
    parser.add_argument("--no-wandb", action="store_true", help="Log nothing to W&B, for smoke runs.")
    return parser.parse_args()


def main() -> None:
    """Entry point: list the grid, run one point, or run every unfinished point."""
    args = parse_args()
    grid = build_grid(args.stage)

    if args.list:
        for point in grid:
            print(f"{point.index:2d} {point.name}")
        return

    if not args.start_weights:
        raise ValueError("--start-weights is required")
    args.start_weights = resolve_start_weights(args.start_weights, args.training_config)
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    # Fail on a bad config before any subprocess starts.
    load_sweep_config(args.config)

    if args.index >= 0:
        if args.index >= len(grid):
            raise ValueError(f"--index {args.index} is outside the grid of {len(grid)}")
        summary = run_one(args, grid[args.index])
        print(json.dumps(summary, indent=2))
    elif args.all:
        run_all(args, grid)
    else:
        raise ValueError("Pass --index N, --all, or --list")


if __name__ == "__main__":
    main()
