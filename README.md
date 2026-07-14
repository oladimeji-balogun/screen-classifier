# Scene Classifier

6-class scene classification (buildings, forest, glacier, mountain, sea, street) on the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset, built as a template for production-grade ML project structure.

## Stack
- PyTorch + torchvision (ResNet18 baseline)
- Hydra for configuration
- MLflow for experiment tracking
- DVC for data versioning
- pytest for testing

## Status
Under active development -- Milestone 0 (scaffolding) complete.

## Setup
\`\`\`bash
uv sync
\`\`\`

## Project structure
\`\`\`bash
configs/     — Hydra configs (data, model, train)
data/        — raw / interim / processed (DVC-tracked, not in git)
src/         — installable package (scene_classifier)
tests/       — unit tests, mirrors src/
scripts/     — CLI entrypoints (train.py, evaluate.py, predict.py)
models/      — saved checkpoints (not in git)
\`\`\`