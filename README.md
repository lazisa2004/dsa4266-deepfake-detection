# DSA4266 Deepfake Detection ? Experiment 1 CNN v2

Visual deepfake detection on LAV-DF-5000 using ResNet50 and EfficientNet-B0.
Develop Python source locally in VS Code, review changes in GitHub, and execute
one local notebook against a remote Kaggle Tesla T4 Jupyter kernel.

The maintained local notebook is `experiment1_cnn_v2_vscode_kaggle.ipynb`.
It is intentionally **not tracked or uploaded to GitHub**. Kaggle fetches the
reviewed Python source into a fresh directory under `/kaggle/working` using a
read-only repository token stored in Kaggle Secrets. No Python ZIP upload or
`codeeeee` dataset is needed.

## Workflow

See [VS Code + Kaggle setup, reconnecting and backups](docs/VSCODE_KAGGLE_WORKFLOW.md).
Keep these existing Kaggle input datasets attached:

- `lav-df-5000-dsa4266`: original videos and shared source metadata.
- `meta-dataa`: the existing `candidates.sqlite` and supporting reports.
- `weights`: the official pretrained ResNet50 and EfficientNet-B0 weights.

The notebook discovers inputs by file markers, verifies the source commit and
code version, runs lightweight tests and bounded GPU/loader checks automatically,
then provides separate training sections for the two CNNs. Executing a training
section starts that model; preparing this repository does not run training.
A new kernel/session uses the same local notebook and the same input datasets.

## Current protocol

| Setting | CNN v2 |
| --- | --- |
| Candidate schedule | 10 FPS, timestamp-based |
| TRAIN candidates | 315,058 frames: 302,240 real / 12,818 fake |
| Per-epoch TRAIN selection | 8,000 unique frames: 4,000 real / 4,000 fake |
| Sample rotation | Deterministic by experiment seed and epoch; family/video diversity |
| DEV | All fixed 41,746 frames: 40,115 real / 1,631 fake |
| Backbone schedule | 2 head-only warm-up epochs, then existing final-stage fine-tuning |
| Optimizer | AdamW; head LR 1e-4, backbone LR 1e-5, weight decay 1e-4 |
| Regularization | Dropout 0.3, TRAIN horizontal-flip probability 0.5 |
| Training limit | 10 epochs; early-stopping patience 3 |
| Video aggregation / thresholds | Top-5%; frame and video thresholds fixed at 0.5 |
| Selection | DEV video ROC-AUC; balanced DEV loss breaks cross-model ties |
| Precision | Safe AMP gradient handling; explicitly configurable FP32 fallback |

Boundary-near frames remain included. DEV is naturally imbalanced and is never
augmented or resampled. Images are decoded lazily for selected examples instead
of constructing a full decoded-image cache. No LSTM model is trained by this
notebook. Existing clip metadata and its finalized labeling rule are retained.

The fresh notebook defaults to `RESUME_FROM = {}`. Completed checkpoints also
retain safe resume support. See [numerical stability and resume details](docs/VSCODE_KAGGLE_WORKFLOW.md#unchanged-cnn-methodology).

## Repository contents

One maintained package replaces the overlapping legacy packages:

- `cnn_experiment/config.py`: the unchanged CNN v2 configuration.
- `cnn_experiment/data.py`: candidate metadata/rules, rotating sampling, bounded
  decoding, preprocessing and the importable spawned-worker dataset.
- `cnn_experiment/models.py`: original model definitions, pretrained-weight
  checks, staged freezing and optimizer construction.
- `cnn_experiment/training.py`: unchanged epoch loop, AMP safeguards and live progress.
- `cnn_experiment/checkpoints.py`: unchanged checkpoint schema, validation and resume state.
- `cnn_experiment/evaluation.py`: metrics, Top-5% aggregation, histories, plots and DEV comparison.
- `cnn_experiment/workflow.py`: input validation, automatic bounded diagnostics,
  unique sessions and verified output backups.
- `cnn_experiment/final_evaluation.py`: separately protected future final evaluation.
- `cnn_experiment/github.py`: credential-safe checkout, embedded in the local notebook.
- `tests/`: five test modules covering all 40 existing lightweight checks.
- `docs/VSCODE_KAGGLE_WORKFLOW.md`: setup, reconnecting, diagnostics and backup instructions.

The repository proposal contains **21 files**: 10 package files (including its
initializer), 6 test files (including its initializer), one workflow guide and
4 root supporting files. The single notebook stays local and is excluded.
Original `experiment1/`, `experiment1_v2/`, old tests, notebooks and historical
notes remain preserved locally and ignored; they are not runtime dependencies.
Only `cnn_experiment` is maintained by this workflow.

The strict `.gitignore` uses an explicit file allowlist. New files require an
intentional allowlist change and review. Datasets, weights, checkpoints,
predictions, outputs, caches, archives and credentials stay outside Git.

## Validation

In the existing Windows development environment:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_data tests.test_training_state tests.test_training_progress tests.test_workflow tests.test_github -v
```

The suite uses synthetic tensors/artifacts and temporary mounts. An optional
existing local TRAIN fixture reads eight frames to compare main-process and
spawned loading; it is skipped when absent. Private checkout security uses
mocked Git/Secrets. Actual Kaggle Secrets, CUDA/T4 and remote VS Code connectivity
are verified by notebook preflight on the user's Kaggle session. Dependencies
are listed in `requirements.txt`; keep Kaggle's matched CUDA torch/torchvision
pair rather than replacing it unnecessarily.

## Outputs and held-out isolation

Each run writes to a new session under `/kaggle/working/experiment1_cnn_v2/`, with
separate diagnostics, model directories and DEV comparison. Live progress,
epoch metrics, current/best DEV plots, numerical diagnostics, CSV/JSON histories,
`last.pt` and `best.pt` are saved automatically. Checkpoints/history save before
plotting. Existing run directories are never silently reused.

The notebook packages checkpoints and other SESSION outputs into a new ZIP with
CRC and SHA256 verification. Download it and its manifest outside Kaggle before
expiry. Saving the local notebook alone does not preserve remote checkpoints.

Final held-out evaluation remains disabled until explicit later approval. Its
approval check occurs before data access; it uses only the DEV-selected CNN and
frozen policy. No actual held-out evaluation or performance claim is made here.
No new model training or experiment results are claimed by this setup.
