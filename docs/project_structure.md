# Project Structure

## Main Directories

- `configs/`: YAML experiment configs.
- `dataset/`: existing input dataset. Keep this read-only.
- `data/`: optional processed or intermediate data created by this project.
- `docs/`: environment notes, experiment notes, and project documentation.
- `notebooks/`: exploratory notebooks.
- `outputs/checkpoints/`: model checkpoints.
- `outputs/logs/`: TensorBoard logs, text logs, and metrics.
- `scripts/`: command-line helpers that can run from the project root.
- `src/bs/`: reusable Python package code.
- `tests/`: lightweight regression tests.

## Current Dataset Assumption

The default config expects this structure:

```text
dataset/葡萄膜炎_dataset/split_dataorigin/
  img/f1 ... img/f5              # observed as .jpg
  mask/f1 ... mask/f5            # observed as .nii.gz
  HRNet_Result/f1 ... f5         # observed as .png
```

`scripts/index_dataset.py` checks how many image-mask pairs are available in each fold.
