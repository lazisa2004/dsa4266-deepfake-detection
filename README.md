# DSA4266 Experiment 1: fixed-pool CNNs and ResNet50-LSTM

One local VS Code notebook with a remote Kaggle GPU kernel:
experiment1_cnn_v2_vscode_kaggle.ipynb. The notebook stays local and ignored.
Reviewed Python source fetches from the private GitHub repository using the Kaggle
Secret named exactly Github Token. No credentials enter URLs or logs.

Keep existing video, meta-dataa and weights Kaggle inputs. The new CNN pipeline
builds one immutable20k TRAIN frame pool directly from TRAIN metadata/timestamps,
then rotates4k real+4k fake per epoch. Both CNNs use identical corresponding IDs.
Full41,746-frame /500-video DEV evaluation stays fixed. Default new combined
profile: fixed20k_lr15e6, max10/patience3, warm-up2, dropout0.3, AdamW head1e-4 /
final-stage1.5e-5 and DEV frame PR-AUC checkpoint selection. The matching LR1e-5
control and original historical baseline remain separately named.

The approved reusable combined TRAIN/DEV index is also available at
[`metadata/candidates.sqlite`](metadata/README.md). It contains frame/clip locations,
timestamps and visual labels, not images, TEST records or weights. The notebook
continues using the attached `meta-dataa` input by default. See the metadata README
for exact counts, verification and the optional repository-index path override.

The new frozen ResNet50-LSTM requires a selected TRAINED ResNet50 best.pt,
extracts pooled2048 features in microbatches and trains the LSTM/classifier only.
Separate16-frame /stride12 /10FPS clips;1,500 real+1,500 fake per epoch. One fake
frame is enough; audio-only is visually real. Existing full DEV clips are retained.

See [settings, baseline analysis, exact notebook order, resume and estimates](docs/RETRAINING_AND_BACKUPS.md)
and [remote workflow](docs/VSCODE_KAGGLE_WORKFLOW.md).

Live metrics/AMP diagnostics/checkpoints run every epoch. Graphs run only after
training. Final DEV family-separated F2 calibration/reporting does not alter full
ranking metrics or TEST policy. Automatic rolling/final ZIPs require manual download
before expiry. No mandatory download test, Kaggle API authentication or upload.

## Maintained files

32 tracked files:14 package files including initializer,10 test files including
initializer,2 guides,4 root support files, plus the approved index and its README.
New pool.py, temporal.py and
calibration.py reuse unchanged data.py decoder/preprocessing and models.py CNNs.
training.py is the shared CNN/temporal training/AMP/progress/checkpoint engine.
Original packages, notebooks, checkpoints, datasets, histories and backups stay
local, ignored and preserved, except for the explicitly approved metadata index.
The allowlist excludes all unreviewed files.

## Lightweight checks

From linked:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

Tests use bounded synthetic CPU fixtures, temporary videos and mocked remote APIs.
No full CNN/GPU training or TEST evaluation. Kaggle preflight measures actual CUDA
memory and bounded loading before explicit training-cell execution. Source changes
still require review/approval before committing/pushing; preparation never uploads.
