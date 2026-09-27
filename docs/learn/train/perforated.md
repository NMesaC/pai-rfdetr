---
description: Train RF-DETR with PerforatedAI dendrites. Set perforate=True to wrap the decoder heads, let PAI switch between neuron and dendrite phases, and read the outputs.
---

# PerforatedAI Dendrites

RF-DETR can train with [PerforatedAI](https://www.perforatedai.com/) artificial dendrites. Set `perforate=True` on `TrainConfig` and the training loop wraps every `Linear` and `Conv2d` of the detector in PAI neuron modules, builds the optimizer and the learning-rate schedule through PAI, and lets PAI switch between neuron training and dendrite training on the validation metric. Everything else (data pipeline, EMA, COCO evaluation, checkpoints, loggers) is the standard RF-DETR stack.

## Install

```bash
pip install "rfdetr[train,pai]"
```

`perforatedbp` needs a license supplied through environment variables. Without it the first `perforate=True` run stops at PAI's license prompt.

## Run

=== "Python"

    ```python
    from rfdetr import RFDETRSmall

    model = RFDETRSmall()
    model.train(
        dataset_dir="<DATASET_PATH>",
        epochs=2000,  # Upper Bound (PAI ends the run on training_complete)
        batch_size=16,
        amp_dtype=None,  # fp32
        early_stopping=False,
        eval_interval=1,
        perforate=True,
        pai_n_epochs_to_switch=30,
    )
    ```

=== "CLI"

    ```bash
    rfdetr fit --config configs/rfdetr_nano_pills_pai.yaml
    ```

    `configs/rfdetr_nano_pills_pai.yaml` is a complete example.

The plain finetune to compare against is the same call with `perforate=False`.

## Settings

| `TrainConfig` field                  | Default             | Meaning                                                                                                              |
| ------------------------------------ | ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `perforate`                          | `False`             | Master switch.                                                                                                       |
| `pai_target`                         | `"all"`             | Modules that receive dendrites: `all` (every `Linear` and `Conv2d`), `cls_head`, `bbox_head`, or `cls_bbox_head`.   |
| `pai_n_epochs_to_switch`             | `30`                | Validations without improvement before PAI switches phase.                                                           |
| `pai_p_epochs_to_switch`             | `4`                 | Validations without a correlation improvement before PAI ends a dendrite phase.                                      |
| `pai_initial_correlation_batches`    | `0`                 | Correlation warmup batches before dendrite weights move. `0` resolves at train start to the fraction below.          |
| `pai_initial_correlation_fraction`   | `0.8`               | Fraction of one epoch used as the warmup when `pai_initial_correlation_batches` is `0`.                              |
| `pai_max_dendrites`                  | `2`                 | Hard cap on dendrite sets.                                                                                           |
| `pai_max_dendrite_tries`             | `2`                 | Failed cycles allowed before PAI ends the run.                                                                       |
| `pai_fixed_switch_every`             | `None`              | Switch phase every N epochs regardless of score. `None` keeps PAI's adaptive mode.                                   |
| `pai_testing_dendrite_capacity`      | `False`             | Run PAI's capacity probe instead of a real experiment (smoke tests).                                                 |
| `pai_track_leaves`                   | `False`             | Wrap every non-perforated leaf module as a PAI tracked module. Fallback for a `perforatedbp` `parameter_type` halt.  |
| `pai_save_name`                      | `None`              | PAI system folder name. `None` derives `<model>_<target>_dendritic_<timestamp>`.                                     |
| `pai_load_folder` / `pai_load_stage` | `None` / `"latest"` | Resume from a saved PAI system folder (see Outputs).                                                                 |
| `pai_force_first_switch`             | `False`             | Switch to dendrite training at the first validation; the one neuron epoch before it runs at lr 0.                    |
| `pai_dendrite_lr`                    | `None`              | Learning rate of the dendrite parameters during dendrite phases. `None` inherits the param group's rate.             |
| `pai_candidate_init_mult`            | `None`              | PAI `candidate_weight_initialization_multiplier`.                                                                    |
| `pai_candidate_init_by_main`         | `None`              | PAI `candidate_weight_init_by_main`.                                                                                 |
| `pai_global_candidates`              | `None`              | Candidates trained per dendrite. More than one enables PAI's `no_backward_workaround` and non-strict loading.        |
| `pai_forward_function`               | `None`              | Dendrite output nonlinearity: a `torch` / `torch.nn.functional` name (`relu`, `gelu`, ...), `identity`, or a callable. |

`TrainConfig` rejects `perforate=True` together with `eval_interval != 1`, `early_stopping=True`, multi-device settings, `compile=True`, or a keypoint model. It warns when `lr_scheduler` is set (PAI owns the schedule) and when `amp_dtype` is not `None` (mixed precision broke dendrite training on earlier arms).

## What the integration does

- **Wrapping.** `RFDETRModelModule` perforates the model after the pretrained weights load, so checkpoint keys match the plain model. With `pai_target="all"` every `Linear` and `Conv2d` is wrapped (151 modules on Nano); one measuring forward pass gives each module its neuron axis, since the decoder heads see 4D stacked outputs while the rest of the model sees 3D tokens, and turns any candidate called more than once per forward (the 13 two-stage class-head copies) into a PAI tracked module instead, because perforated backpropagation cannot pair two calls with one error. Parameters outside the wrapped modules (norms, embeddings) are registered with PAI by name rather than as tracked modules, because a tracked wrapper renames parameters that RF-DETR's parameter grouping reads.
- **Optimizer.** `configure_optimizers` returns an AdamW built through PAI's `setup_optimizer` from RF-DETR's usual layer-wise param groups. Dendrite parameters get weight decay 0. In dendrite phases only the perforated modules' parameters are handed to the optimizer, which is what keeps the backbone and transformer frozen. Lightning gets no scheduler: PAI steps a `ReduceLROnPlateau` (factor 0.1, patience 14, threshold 0.1% relative, floor `lr * 0.01`) inside `add_validation_score`, which is what its learning-rate search needs.
- **Callback.** `PerforatedAICallback` runs after `COCOEvalCallback` and `BestModelCallback`. It copies the EMA weights into the live model before handing PAI the score (validation scores the EMA model, and PAI checkpoints the model it is handed), restores them when PAI does not restructure, and rebuilds the optimizer and the EMA copy when it does. The AdamW moments of the model's own parameters survive a restructure by name. Neuron phases score on the task metric `BestModelCallback` monitors; dendrite phases score on the mean best dendrite correlation.

## Outputs

- `output_dir/` holds RF-DETR's usual `metrics.csv`, `last.ckpt`, and `checkpoint_best_*.pth`. These `.pth` files carry PAI-wrapped keys and only reload in a process that perforated the model the same way (same `perforate` and `pai_target`).
- `output_dir/final_clean_model.pth` is the EMA detector with the PAI tracker removed. Its perforated heads are `perforatedai.clean_perforatedai` modules holding the neuron and dendrite layers as plain `torch.nn` layers, so the file carries the whole `LWDETR` under `module`: reload it with `torch.load(path, weights_only=False)["module"]` in any process with `perforatedai` installed. `model` is that module's state dict; it does not load into a plain `LWDETR`, and `RFDETR(pretrain_weights=...)` cannot read it. `RFDETR.train()` syncs the same clean copy onto the detector so `predict()` and `export()` work after training.
- PAI writes its own system folder, `<model>_<target>_dendritic_<timestamp>`, in the working directory. Resume a dendrite run with `pai_load_folder` pointing at that folder, not with `resume=last.ckpt`.

## Extending

Custom target ids are not exposed through `TrainConfig`. To add a target, define a `PerforationTarget` in `rfdetr.training.perforated`, register it in `PERFORATION_TARGETS`, and add its name to `rfdetr.config.PerforationTargetName`. `list_perforable_modules(model)` prints every id a target can use.

## Sweeping dendrite settings

`scripts/perforated_sweep.py` trains a grid of `pai_*` settings, every point from the same plateaued checkpoint, one subprocess per point because PAI keeps its state in process globals. It reads the same Lightning CLI YAML the arms train with and logs its own W&B stream (lift over the loaded model, mode, dendrite count, per-layer correlations) through the callback's observer.

```bash
python scripts/perforated_sweep.py configs/rfdetr_nano_pills_pai.yaml --list
python scripts/perforated_sweep.py configs/rfdetr_nano_pills_pai.yaml --stage 1 --all \
    --start-weights <system folder>/best_model_beforeSwitch_0.pt \
    --training-config output/<run>/training_config.json
```

A PAI stage file passed as `--start-weights` is converted once with `extract_start_weights`, which strips the PAI wrapping and the dendrites from the saved state dict and writes `<stage>_plain.pth` beside it. `best_model_beforeSwitch_0` is the best neuron-only model PAI restored before it added the first dendrite. Each finished point writes `sweep_done.json` into its output folder and is skipped on restart.
