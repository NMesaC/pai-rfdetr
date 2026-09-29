---
description: Train RF-DETR with PerforatedAI dendrites. Set perforate=True, name the modules that get dendrites, and point pai_config at a PAIConfig JSON.
---

# PerforatedAI Dendrites

RF-DETR can train with [PerforatedAI](https://www.perforatedai.com/) artificial dendrites. Set `perforate=True` on `TrainConfig` and name the modules that should get dendrites. PAI trains the network, adds a dendrite set when the validation score stalls, trains that, and repeats until it decides to stop. Data loading, EMA, COCO evaluation, checkpoints, and loggers are all the same from RF-DETR.

## Install

```bash
pip install "rfdetr[train]" perforatedai perforatedbp
```

`perforatedbp` needs a license, supplied through environment variables.

## Run

=== "Python"

    ```python
    from rfdetr import RFDETRSmall

    model = RFDETRSmall()
    model.train(
        dataset_dir="<DATASET_PATH>",
        epochs=2000,  # An upper bound, PAI ends the run itself
        batch_size=16,
        amp_dtype=None,
        early_stopping=False,
        eval_interval=1,
        lr_scheduler="torch.optim.lr_scheduler.ReduceLROnPlateau",
        lr_scheduler_kwargs={"mode": "max", "factor": 0.1, "patience": 14, "threshold": 0.001, "min_lr": 1e-6},
        perforate=True,
        pai_target=[".bbox_embed"],
        pai_config="configs/pai/rfdetr_nano_pills.json",
    )
    ```

=== "CLI"

    ```bash
    rfdetr fit --config configs/rfdetr_nano_pills_pai.yaml
    ```

    `configs/rfdetr_nano_pills_pai.yaml` is a full example.

The baseline run is the same command with `perforate=False`.

A few settings are fixed by how PAI works.

- `eval_interval` = 1 (PAI counts epochs in validations).
- `early_stopping` must be off (PAI decides when the run ends).
- Training runs on one device, without `compile`, and not on a keypoint model.
- `lr_scheduler` must be a scheduler PAI can step once per validation with the score, given as a class path with its arguments in `lr_scheduler_kwargs`. The `step` and `cosine` presets are rejected.
- `optimizer` and `lr_scheduler` must be names, not callables (PAI rebuilds both after every restructure).

## Settings

| `TrainConfig` field                  | Default             | Meaning                                                                                                                                                        |
| ------------------------------------ | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `perforate`                          | `False`             | Perforation Switch.                                                                                                                                            |
| `pai_target`                         | `[".bbox_embed"]`   | Modules that get dendrites, as a list of module ids or `fnmatch` patterns. See Module ids.                                                                     |
| `pai_config`                         | `None`              | Path to a `PAIConfig` JSON (see below). `None` keeps PAI's defaults.                                                                                           |
| `pai_track_leaves`                   | `True`              | Wrap every leaf module outside the target as a PAI tracked module. Use alongside `perforatedbp`.                                                               |
| `pai_save_name`                      | `None`              | Name of the folder PAI writes its own state to. `None` derives `<model>_dendritic_<timestamp>`.                                                                |
| `pai_load_folder` / `pai_load_stage` | `None` / `"latest"` | Resume from a folder PAI wrote earlier.                                                                                                                        |
| `pai_dendrite_lr`                    | `None`              | Learning rate of the dendrite parameters while dendrites train. `None` inherits the rate of the parameter group they belong to.                                |
| `pai_forward_function`               | `None`              | Nonlinearity on the dendrite output: a `torch` / `torch.nn.functional` name (`relu`, `gelu`, ...), `identity`, or a callable. `None` is PAI's default, sigmoid. |

### The `PAIConfig` JSON

Configuration settings for the PAI library lives in a JSON file instead of `TrainConfig`. The file is a flat object, one key per `PAIConfig` setting, using PAI's own names. Only put in the keys you want to change. An example recipe for the RF-VL100 Pills Dataset is at `configs/pai/rfdetr_nano_pills.json`.

```json
{
  "testing_dendrite_capacity": false,
  "n_epochs_to_switch": 30,
  "p_epochs_to_switch": 4,
  "max_dendrites": 2,
  "max_dendrite_tries": 2,
  "improvement_threshold": [0.001, 0.0001, 0.0],
  "improvement_threshold_raw": 1e-05,
  "history_lookback": 1,
  "initial_history_after_switches": 0,
  "reset_best_score_on_switch": false,
  "find_best_lr": true,
  "dont_give_up_unless_learning_rate_lowered": true,
  "retain_all_dendrites": false,
  "candidate_weight_initialization_multiplier": 0.01,
  "candidate_weight_init_by_main": false,
  "global_candidates": 1,
  "verbose": false,
  "drawing_pai": true,
  "test_saves": true
}
```

## Module ids

`pai_target` names modules by their `named_modules` path with a leading dot. A pattern with `*`, `?`, or `[23]` expands to every id it matches, and an id or pattern that matches nothing is ignored. If one match lies inside another, only the outer module is wrapped.

```yaml
pai_target: [".bbox_embed"]                                          # The box head
pai_target: [".class_embed", ".bbox_embed"]                          # Both heads
pai_target: [".transformer.decoder.layers.[23].ffn"]                 # The FFN of decoder layers 2 and 3
pai_target: [".transformer.decoder.layers.*.ffn", ".class_embed"]    # Every decoder FFN plus the class head
```

Before wrapping, RF-DETR is rebuilt into the shape PAI requires for best performance. Each normalization layer moves inside the block it belongs to and each residual add stays outside. That is what makes the sub-block ids below exist. `{i}` is a zero-based index.

| Region          | Id                                                                             | What it is                                          |
| --------------- | ------------------------------------------------------------------------------ | --------------------------------------------------- |
| DINOv2 Backbone | `.backbone.0.encoder.encoder.encoder.layer.{i}.attention`                      | `norm1` + Attention + `layer_scale1`                |
| DINOv2 Backbone | `.backbone.0.encoder.encoder.encoder.layer.{i}.mlp`                            | `norm2` + MLP + `layer_scale2`                      |
| Projector       | `.backbone.0.projector.stages.{s}.{j}.cv1` / `.cv2`, `.m.{k}.cv1` / `.cv2`     | Conv + BatchNorm + Activation                       |
| Decoder         | `.transformer.decoder.layers.{i}.self_attn`                                    | Self-Attention + Dropout + Norm                     |
| Decoder         | `.transformer.decoder.layers.{i}.cross_attn`                                   | Deformable Cross-Attention + Dropout + Norm         |
| Decoder         | `.transformer.decoder.layers.{i}.ffn`                                          | FFN + Dropout + Norm                                |
| Decoder         | `.transformer.decoder.ref_point_head`                                          | Reference Point MLP                                 |
| Two-Stage       | `.transformer.enc_output.{g}`                                                  | Linear + LayerNorm (One per `group_detr` group)     |
| Heads           | `.class_embed` / `.bbox_embed`                                                 | The detection heads                                 |
| Segmentation    | `.segmentation_head.blocks.{i}.branch`                                         | Depthwise Conv + Norm + Pointwise Conv (No Residual) |
| Segmentation    | `.segmentation_head.query_features_block.branch`                               | Norm + MLP (No Residual)                            |

The leaves inside a block keep their names, so `.transformer.decoder.layers.0.ffn.linear1` is a valid id too (though not recommended). The two-stage class and box heads (`.transformer.enc_out_class_embed.{g}`, `.transformer.enc_out_bbox_embed.{g}`) are called twice per forward, which PAI cannot pair with one error signal, so they are tracked rather than perforated. `list_perforable_modules(model)` in `rfdetr.training.perforated` prints every id for the model at hand.

## Outputs

- `output_dir/` has the usual RF-DETR files: `metrics.csv`, `last.ckpt`, and `checkpoint_best_*.pth`. The `.pth` files carry PAI-wrapped keys and only load back into a model that was perforated the same way.
- `output_dir/pai_config.json` is a copy of the JSON the run loaded.
- `output_dir/final_clean_model.pth` is the trained detector with the PAI bookkeeping removed. The neuron and dendrite layers are plain `torch.nn` layers inside `perforatedai` clean modules, so reload it with `torch.load(path, weights_only=False)["module"]` in any environment that has `perforatedai` installed. After `RFDETR.train()` the same clean model is already loaded, so `predict()` and `export()` work right away.
- PAI writes its own folder, `<model>_dendritic_<timestamp>`, in the working directory. It holds the score graphs, the saved stages, and the full effective config. Resume from it with `pai_load_folder`, not with `resume=last.ckpt`.
