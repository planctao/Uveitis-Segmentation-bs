---
name: experiment-archive
description: Archive a completed segmentation experiment for later inference by bundling checkpoints, logs, configs, source code, model weights, and a Markdown results record into a dated ZIP while updating EXPERIMENT_LOG.md.
---

# Experiment Archive

Use this skill when the user wants to preserve a trained experiment so it can be moved to cloud storage and later used for inference. The skill creates a local ZIP only; the user is responsible for moving it to Aliyun Drive or another storage service.

## Workflow

1. Identify the project root and the exact completed run directory (or directories). Do not silently combine unrelated runs. If “current experiment” is ambiguous, inspect `runs/` and `outputs/` and ask which run should be archived.
2. Use a concise Backbone label and method label. The helper names the archive `YYYYMMDD_<backbone>_<method>.zip` (lowercase, hyphenated; `_02`, `_03`, … are added on collisions).
3. Collect the run's checkpoints, metrics, logs, TensorBoard events, configuration, and any run-local artifacts, together with the code needed to rebuild the model (`src/`, `scripts/`, `configs/`, `backbone/`, `weights/`, selected root documentation and dependency files).
4. Extract available `metrics.csv`, `fold_summary.csv`, and `summary.json` values. Pass important values explicitly with repeated `--result KEY=VALUE` arguments when a paper-specific metric or an external evaluation is not present in the run files.
5. Invoke the helper:

```bash
python .codex/experiment-archive/scripts/archive_experiment.py \
  --backbone DINOv3-ConvNeXt-Tiny \
  --method VSUBR-DINO-SAM \
  --run-dir runs/<run-name> \
  --result "fivefold_fixed_macro_dice=0.7806" \
  --result "fivefold_sweep_macro_dice=0.7848" \
  --result "fivefold_independent_sweep_macro_dice=0.7855"
```

Run with `--dry-run` first when the run contains multi-gigabyte checkpoints. A real run writes the ZIP under `outputs/experiment_archives/`, creates `ARCHIVE_README.md`, `ARCHIVE_RESULTS.md`, and `ARCHIVE_MANIFEST.json` inside the ZIP, and appends one record to the project-root `EXPERIMENT_LOG.md`. It never deletes checkpoints, logs, source files, or weights.

The archive intentionally excludes the dataset, `.git`, `.codex`, Python caches, virtual environments, intermediate prediction caches, and the archive output directory itself. If a required inference asset is outside the default paths, add it with `--include relative/path`; use `--no-weights` only when the model weights are already available elsewhere and that omission is intentional.

After creating an archive, report its absolute path, size, included runs, headline metrics, and the fact that no external upload was performed. Verify the skill and helper with:

```bash
python /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/experiment-archive
python .codex/experiment-archive/scripts/archive_experiment.py --help
```
