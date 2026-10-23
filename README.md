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

Use the existing Kaggle `meta-dataa` input for `candidates.sqlite`. The same
reusable index is explicitly approved and remains tracked, unchanged, under
`metadata/candidates.sqlite` in this public research repository. Its derived
annotations retain the [upstream attribution and non-commercial terms](metadata/README.md).
No raw videos, trained
weights, credentials, notebooks or generated results are committed. The source
ZIP remains source-only and omits the database.

The new frozen ResNet50-LSTM requires a selected TRAINED ResNet50 best.pt,
extracts pooled2048 features in microbatches and trains the LSTM/classifier only.
Separate16-frame /stride12 /10FPS clips;1,500 real+1,500 fake per epoch. One fake
frame is enough; audio-only is visually real. Existing full DEV clips are retained.

See [settings, baseline analysis, exact notebook order, resume and estimates](docs/EXPERIMENT_METHODOLOGY.md)
and [remote workflow](docs/KAGGLE_SETUP.md).

Live metrics/AMP diagnostics/checkpoints run every epoch. Graphs run only after
training. Final DEV family-separated F2 calibration/reporting does not alter full
ranking metrics or TEST policy. Automatic rolling/final ZIPs require manual download
before expiry. No mandatory download test, Kaggle API authentication or upload.

## Maintained files

The exact proposed source/test/documentation list is generated in the local
pre-push review. It excludes notebooks, model assets and all outputs; only the explicitly approved
metadata database and its README are retained.
The existing pool.py, temporal.py and
calibration.py reuse unchanged data.py decoder/preprocessing and models.py CNNs.
training.py is the shared CNN/temporal training/AMP/progress/checkpoint engine.
Original packages, notebooks, checkpoints, datasets, histories and backups stay
local, ignored and preserved. The approved metadata stays tracked.
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

## Approved new CNN plan

The local `notebooks/experiment1_face_cnn.ipynb` defaults to `face_mask` for the
two new face-focused models; the third new run is full-frame EfficientNet-B0 with
head LR2e-4. `crop_30` remains available for previews/reference, with no extra
scheduled model cells. All six comparison rows retain Frame versus Clip labels.
The internal `*_face_crop` IDs remain for compatibility; actual input identities
and display labels distinguish masks from historical crops. No completed model or
LSTM is retrained. See [face setup, assets and restore](docs/FACE_CNN_EXPERIMENTS.md).


Face preprocessing uses one OpenCV distribution, `opencv-contrib-python==4.13.0.92`,
satisfying MediaPipe0.10.32 without overlapping `cv2` providers. Fresh-kernel setup
downloads the wheel before removing conflicting providers and preserves NumPy and
PyTorch. All100 cached TRAIN audit frames produced identical YuNet records,
landmarks, crops, masks and alpha maps before/after the provider replacement.
See the face guide for Kaggle preflight and the local-only notebook settings.
