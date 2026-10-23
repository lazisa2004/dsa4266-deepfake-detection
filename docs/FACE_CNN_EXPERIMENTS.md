# Face-mask CNN experiments and EfficientNet head LR

Main interface for these NEW experiments: `notebooks/experiment1_face_cnn.ipynb`.
The original `experiment1_cnn_v2_vscode_kaggle.ipynb`, completed checkpoints,
predictions, DEV index and LSTM files are preserved. No TEST section is added to
the new notebook, and no TEST source is accepted by the face loader.

## The three comparisons

| Run ID | Architecture / input | Head LR | Backbone LR | Changed factor vs its actual original full-frame run |
|---|---|---:|---:|---|
| `resnet50_face_crop` | ResNet50 / face mask | 1e-4 | 1.5e-5 | Input preprocessing only |
| `efficientnet_b0_face_crop` | EfficientNet-B0 / face mask | 1e-4 | 1.5e-5 | Input preprocessing only |
| `efficientnet_b0_head_lr_2e4` | EfficientNet-B0 / full frame | 2e-4 | 1.5e-5 | Classifier/head LR only |

All recipes reuse `experiment_config(model, 'fixed20k_lr15e6')`: seed4266,
dropout0.3, batch32, AdamW/weight decay1e-4, max10 epochs, two head-only warm-up
epochs then the original final-stage fine-tuning with frozen BN statistics,
same scheduler/early stopping, AMP gradient guards and frame PR-AUC checkpoint
selection. Only recipe C changes `head_lr`. Each independent fresh run reconstructs
the original architecture and loads the SAME ImageNet weight file, with the same
seeded head initialization; it never starts from a completed trained baseline.
Cross-library-version bitwise identity is not promised: versions/source hashes
are recorded; use the same Kaggle torch/torchvision environment for comparisons.

The comparison references the actual completed fixed20k/LR15e6 full-frame CNN
runs (the saved final ZIPs), not older historical metrics-only baselines. Recorded
config, index and pool SHA256 checks detect an unmatched reference. Baseline
checkpoints are needed only for paired Grad-CAM, not for retraining or table metrics.

The two legacy `*_face_crop` IDs are retained for compatibility, not as extra
scheduled crop models. The final plan has six rows: original full-frame ResNet50,
original full-frame EfficientNet-B0, original ResNet50–LSTM, two new face-mask CNNs,
and the new full-frame EfficientNet-B0 head-LR control. Old crop results cannot fill
mask rows; incompatible saved input policies are excluded or fail clearly.

## Original crop policy, retained unchanged for preview/reference

The detector is the small official OpenCV YuNet **2023mar FP32 ONNX** model,
232,589 bytes, SHA256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
Use only `opencv-contrib-python==4.13.0.92`, OpenCV CPU DNN, one thread and no OpenCL.
See the [official YuNet model and licence](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
and [OpenCV detector interface](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
No face-recognition model or face identity matching is used.

* Confidence cutoff0.8; NMS0.3; top-k5000. Choose largest visible valid face box,
  then highest confidence, centre distance and coordinates to break ties.
* Default margin0.30 means **30% TOTAL expansion of the largest face-box side**,
  i.e.15% padding per side. Configurable range0.25–0.35 total (12.5%–17.5% each side).
  This definition is saved with the versioned policy. A square includes forehead,
  eyes, cheeks, mouth, jaw/chin context; this is not mouth-only cropping.
* Shift the square within native224x224 bounds, cap it at source size, no black
  padding or invented pixels. Bilinear resize to224x224 preserves aspect ratio.
  No JPEG or other lossy image cache is created. Resizing necessarily interpolates
  pixels, which is part of the declared experimental input change.
* No detected face -> **original full-frame fallback**, explicitly marked in the
  cache/reports. Never drop or relabel a DEV frame. A broken detector/weight file
  or video decoding error raises; it is not disguised as a no-face fallback.
* The detector receives RGB pixels converted to BGR for OpenCV. Neither labels,
  fake intervals, prediction probabilities nor TRAIN/DEV split influence the box.
  TRAIN and DEV use the identical policy. Existing LazyDataset handles the same
  ImageNet normalization and deterministic TRAIN flip AFTER cropping.

The first preview attempt used30% padding on each side and returned the whole
224x224 image on all12 close-up examples. It was preserved under the parent
project's `archive/face_preview_initial_margin_20261023`. The implemented version
uses the total-expansion definition above. This refinement was based on crop
geometry before training, not DEV performance. Already-tight source frames may
still produce large/full-frame crops; retention fractions and full-frame counts
are reported rather than claiming all images become tightly cropped.

## Index, cache and sampling

`verify_inputs` opens the EXISTING combined `candidates.sqlite` read-only through
`DevSchedule`:500 videos,41,746 frames,40,115 Real+1,631 Fake, exact timestamps and
source indices. No balancing, augmentation or new DEV schedule.

No separate `train_pool.json` upload is required. `resolve_train_pool` first reuses
a matching attached/working manifest, then a matching attached completed-CNN ZIP
(extracting **only** the small TRAIN manifest). If neither exists, it invokes the
original `cnn_experiment.pool.build_pool(DATA, output, PoolConfig())` unchanged:
seed4266,10,000 per class,10FPS, one-source-frame guard and preferred spacing0.2s.
The original notebook used SQLite for fixed DEV; this separate TRAIN builder
reads TRAIN annotations in `metadata.json` and the original video packet timelines,
then decode-verifies the20,000 selected frames. It does not derive this newer pool
from the older SQLite TRAIN frame table or rebuild candidates.sqlite.

Every existing, extracted or regenerated manifest must match the original SHA256
`8bcb5c9845f92b43e36213f22e410bcecedf983f7db87250f02a9968868eea92`
and exact original `PoolConfig`. A mismatch, changed source, corrupt ZIP or incomplete
build stops setup/training clearly; no seed, rule or pool is silently substituted.
An explicit `POOL_OVERRIDE` is also hash checked. Matching metadata in a saved ZIP
avoids another TRAIN timeline/decode audit and does not extract checkpoint weights.

`FixedPool` validates10,000 Real+10,000 Fake; `FixedPoolSampler` is reused directly
for8,000 frames per epoch,4,000 each class, class rotation and diversity. Three
disposable samplers verify identical corresponding IDs for all10 epochs, without
consuming the training sampler. Regeneration is only the original deterministic
TRAIN-pool procedure above; no new sampling protocol or DEV/index regeneration.

Face detection is lazy: only preview, sampled TRAIN and fixed DEV frames decode
when used. `FrameReader` keeps the existing exact PTS/source-index/native-RGB checks.
`FaceReader` adds crops, then existing LazyDataset performs normalization and flip.
The cache contains **only bounding boxes, confidence/status, crop coordinates and
source identities**, not decoded images. Keys use split-relative source path,
original frame index and rational timestamp. Cache identity binds the exact
index/pool hashes, detector SHA, OpenCV version and crop configuration/version.
Labels are not stored in the detector cache or used for its identity.

Each spawned worker opens its own lazy CPU detector/SQLite connection. Transactions
commit per completed detection, so interruption loses at most unfinished work.
WAL handles concurrent worker writes. Snapshots use SQLite's backup API, never a
copy of a live database without its WAL. A wrong detector/cache policy raises.

The same cache is shared by A/B. At most61,746 frame identities are needed for the
full20,000 TRAIN pool plus41,746 DEV; only those actually visited are materialized.
Expect tens of MiB of metadata, not a decoded-image dataset. No millions of images
are predecoded. Original videos remain reattached private Kaggle inputs.

## Kaggle execution order

1. Open the NEW notebook locally and select your Kaggle Tesla T4 remote kernel.
   Keep the existing video, `meta-dataa` and pretrained `weights` datasets. Source
   and inputs need to be on Kaggle; the remote kernel cannot read Windows folders.
2. Until a separate approved push, attach the supplied **face_cnn_source_*.zip** as
   a private code dataset. Notebook setup auto-discovers it with bounded searches
   that prune video split folders. Optional `SOURCE_ZIP` resolves ambiguity.
   No GitHub push/upload is performed by development here. For later GitHub use,
   select `USE_GITHUB_SOURCE=True` only after approval/push, with the existing
   Secret **`Github Token`**. Pin the reviewed commit for repeatability. No token
   is printed, stored in notebook outputs or placed in a Git URL.
3. No additional TRAIN-manifest dataset is needed. Existing manifests, the old
   `/kaggle/working/fixed_cnn_pool_v1/train_pool.json`, previous face-session pools,
   attached completed/rolling ResNet50/EfficientNet CNN ZIPs and
   `fixed_cnn_pool_manifest.zip` are discovered without entering video folders.
   Otherwise section3 reproduces and hash-verifies the original pool automatically.
   An optional `POOL_OVERRIDE` selects an explicit matching manifest. Original final
   ZIP paths with custom filenames can be supplied via `BASELINE_ZIPS`: `restore_local_zip` verifies
   the archive and restores into NEW session folders. Alternatively use existing
   extracted baseline directories via `BASELINE_RUNS`.
4. Run sections1–3: source/dependency/GPU checks, automatic regression tests,
   existing input checks, full DEV identity and disposable epoch-ID validation.
   Original Kaggle paths:
   - Videos: `/kaggle/input/datasets/leezile/lav-df-5000-dsa4266/LAV-DF-5000`
   - Index: `/kaggle/input/datasets/leezile/meta-dataa/candidates.sqlite`
   - ImageNet weights: `/kaggle/input/datasets/leezile/weights`
   Required weight filenames remain `resnet50-11ad3fa6.pth` and
   `efficientnet_b0_rwightman-7f5810bc.pth`.
5. Run section4 and **inspect the crop previews**. Only12 source frames are used.
   Setup explicitly downloads/verifies the tiny public detector if no
   `DETECTOR_OVERRIDE` is supplied. Training itself never downloads weights.
   For offline reuse, attach that ONNX file as a private input.
6. Run section5: bounded loader/spawn/GPU numerical preflight with disposable
   ImageNet models and **zero optimizer steps**. Finite logits/loss/gradients are
   checked in both freeze phases. No full-dataset detection prepass is performed.
7. Execute **A, then B, then C** individually. Each cell starts only its own new
   experiment. Do NOT Run All. Existing live first/every50-batch progress, DEV
   progress, per-epoch metrics, numerical JSONL, best/last/history and ZIP creation
   remain enabled. Final training graphs are generated only after each run.
8. Execute sections6–8: five-row comparison, graphs/confusions, matched detector
   failure cohorts, paired DEV Grad-CAM and printed ZIP download paths.
   Missing/untrained rows remain explicitly pending; no fabricated results.

Runtime is dominated by the existing decode/training/full DEV passes; the first
face-crop epoch additionally detects previously unseen frame boxes, subsequent
epochs reuse them. A/B share the cache, C has no detection/crop overhead. Startup
prints measured bounded timings; extrapolations are not promises of T4 throughput.
Allow space for all three independent runs and their rolling/final checkpoint ZIPs
(several GiB), alongside a small cache; no26GiB decoded-image cache is constructed.

## Outputs, recovery and comparison meaning

Outputs are under a unique `/kaggle/working/experiment1_face_cnn/session_*`:

```
resnet50_face_crop/run/                  # best.pt, last.pt, history, metrics, plots
resnet50_face_crop/downloads/            # rolling/final ZIP, independent parent
efficientnet_b0_face_crop/run/
efficientnet_b0_face_crop/downloads/
efficientnet_b0_head_lr_2e4/run/
efficientnet_b0_head_lr_2e4/downloads/
```

Cropped checkpoints add `input_pipeline` metadata; architecture/state-dict names
are unchanged. Resume rejects a different face policy or a face/full-frame swap,
for both `last.pt` and companion `best.pt`, then uses the existing optimizer,
scheduler, scaler, RNG, sampler and early-stop restoration. Download completed ZIPs
to Windows before the session expires. Kaggle browser Files -> printed downloads
folder -> refresh -> Download is reliable when VS Code remote FileLinks fail.
ZIP creation on `/kaggle/working` alone is not persistent backup.

Face-run ZIPs include `input_pipeline.json`, `face_boxes.sqlite` snapshot,
`face_detection_report.json`, tiny `face_detector.onnx`, existing `train_pool.json`
and experiment plan alongside normal checkpoints/results. These generated files
stay ignored by Git. No videos or original candidates.sqlite are newly bundled.
For recovery restore to a NEW folder and new session, use that run's `last.pt`
via `RESUME_FROM`, preserve companion `best.pt`, and set `CACHE_OVERRIDE` to the
snapshot to avoid redetection. Same pinned detector/crop policy is required.

The five-row CSV/JSON report contains best epoch, best TRAIN/DEV losses, training
duration, frame/video ROC/PR-AUC, precision/recall/F1/F2 at0.5 and separate DEV-family
F2-calibrated reporting metrics/thresholds. Frame/video confusion matrices are saved.
Video scores use existing `video_predictions`: mean of the top
`max(1,ceil(.05*N))` frame scores over the FULL schedule, not max pooling.
No thresholds/models are chosen with TEST. DEV-family reporting is not an untouched
holdout because full DEV already selected the best checkpoint. Threshold changes
cannot improve ranking AUC or BCE.

Detection failures keep the same complete full-DEV evaluation. Paired frame-only
detected/fallback cohorts compare the SAME frame IDs for original and crop models,
with counts and label balance. Detection success can correlate with pose, quality
or labels; those descriptive subsets do not replace full DEV and are not silently
used to change video aggregation. No complete failure rate is asserted from previews.

Paired Grad-CAM selects original-model DEV confusion examples and applies both
models to the same source records. Face checkpoints refuse full-frame Grad-CAM
unless the exact saved pipeline is provided. Cropped heatmaps overlay the actual
face-model input, not untransformed full frames. Both maps are post-hoc evidence,
not pixel ground truth or proof of manipulated facial regions.

## Changed existing functions

* `cnn_experiment.training.train/_train`: optional input-pipeline dependency,
  chosen loader, pipeline identity in new crop checkpoints, resume identity check.
* `training.commit_epoch`: optional cache snapshot before normal epoch ZIP;
  final snapshot is also taken before the existing final ZIP.
* `cnn_experiment.backup.OPTIONAL`: allow only the named crop artifacts in ZIPs.
  Original runs have no such files and keep the same member set.
* `cnn_experiment.gradcam.load_model/save_explanations`: carry recorded input
  metadata, require matching crop preprocessing and label the cropped input.

The original data/model/config/evaluation/calibration/checkpoint/LSTM modules and
original notebook are unchanged. Current source before these optional extensions
is preserved in the parent project's `archive/before_face_experiments_*.zip`.
New package modules, notebook, tests and this document are listed in the local
release manifest; no commit or push is made.

## Validation performed locally

*22 face-experiment tests pass*, including8 tests for automatic pool resolution,
ZIP-only reuse and fail-closed hash checks. The latest source-only release checkout
runs108 tests:105 pass,3 skip,0 errors/failures. Before this pool-workflow update,
the normal local project ran100:98 pass,1 skip,1 pre-existing error in `tests/test_github.py:93`, whose old local
notebook assertion calls `startswith` on a list-form cell source. That unrelated
test and the saved original notebook remain unchanged. Source-only checkouts do
not include the local notebook and intentionally skip that assertion.

Real existing metadata was verified: unchanged41,746 DEV records/500 videos and
the same original20,000-frame TRAIN pool; identical balanced sampling IDs across
all10 planned epochs for A/B/C. Original index SHA:
`ac3cd8c440f0bb1e53cf955bcc6a2596e9ee1bc40e508f254c82e10bde13cfcb`;
original pool SHA:
`8bcb5c9845f92b43e36213f22e410bcecedf983f7db87250f02a9968868eea92`.

Twelve real TRAIN/DEV crop previews were generated and visually inspected, taking
8.10 seconds including decode/detection/plotting on this CPU. All12 had detected
faces, none required fallback, and mean retained image area was83.64%. This small
preview does not establish the complete DEV failure rate. The12-entry metadata
cache was20KiB; no decoded-image cache was built.

Disposable ImageNet models passed finite forward/backward checks in both warm-up
and fine-tuning phases for all three configurations, with **zero optimizer steps**
and unchanged weights/BN buffers. Spawned-worker tensors matched single-worker
tensors exactly. These are CPU smoke checks; the notebook automatically repeats
bounded GPU/AMP checks on your T4 before any experiment training cell.

Local artifacts: `linked/face_dev_validation/crop_previews/`,
`face_dev_validation/validation_report.json` and
`face_dev_validation/comparison_before_training/`. The comparison already contains
the actual original full-frame results, with three new-run rows explicitly pending.
Logs/receipts live under the parent project's `outputs/face_*` and remain ignored.
No actual TEST data or model training was executed. The original TRAIN pool was
subsequently reproduced as described below; no decoded-image cache was retained.

## Original-pool reproduction verified (24 October 2026)

The unmodified original `build_pool` with the unmodified `PoolConfig()` was run
against the existing local TRAIN videos/annotations. All20,000 selected frames
were decode-verified. The resulting manifest matched the saved hash **byte for
byte**: `8bcb5c9845f92b43e36213f22e410bcecedf983f7db87250f02a9968868eea92`.
This took607.5 seconds locally. No model, face detector or TEST evaluation ran.

Separately, `resolve_train_pool` reused the original ResNet50 completed ZIP and
obtained the same hash while reading ONLY `train_pool.json`; the original ZIP
was unchanged and no weights were extracted. Regeneration was instrumented to
raise if called on that reuse path, confirming it was avoided.

Receipts in the parent project's `outputs/`:
`train_pool_reproduction_receipt.json`, `face_pool_zip_reuse_receipt.json`,
and `face_pool_source_tests.txt`. On Kaggle, section3 records the selected path,
required/actual hash and seed in `diagnostics/pool_resolution.json`. Existing
matching manifests bypass regeneration; otherwise the original reproducible
metadata-only pool is created automatically. Keep using the same original video,
meta-dataa and weights datasets; there is no extra TRAIN-manifest upload requirement.

## Completed-model metrics tables and unchanged graphs

Both local notebooks now display **one cumulative table after each model's training
call returns**, followed by the original saved figures in this order:
`training_curves.png`, `best_dev_confusion.png`, `best_dev_roc_pr.png`, then
`dev_f2_reporting_confusion.png`. Live batch progress and epoch text logs remain;
there is no per-epoch table. Existing best-checkpoint summaries and ZIP/download
outputs remain. No graph is regenerated during saved-result replay.

The 19-column table includes TRAIN/DEV and balanced DEV loss, frame (or clip) and
video ROC/PR-AUC, precision, recall, F1, F2, both learning rates, and the recorded
selected best epoch. Classification metrics use **0.5**, separately from any
DEV-calibrated operating-point reporting. Metrics show four decimal places;
learning rates use four-decimal scientific notation so small rates remain visible.
Scroll horizontally to see all columns; the selected best row is highlighted.

Video F2 is `5*precision*recall/(4*precision+recall)` (zero when the denominator is
zero), equivalently `5*TP/(5*TP+4*FN+FP)`. It is already present in the current
per-epoch video metrics and histories; legacy F2 can be recovered only from that
epoch's saved confusion counts or precision/recall. CNN video probabilities still
use the original Top-5% mean. No pooling, threshold calibration, selection rule,
training or model configuration changed.

To view a previously completed model, **do not execute its training cell**. After
loading the updated source, run this directly (no videos, index or weights needed):

```python
from cnn_experiment.reporting import display_completed_run, display_comparison
saved = '/kaggle/input/your-private-results/resnet50_final_example.zip'
display_completed_run(saved, name='ResNet50', kind='frame')
# For the saved temporal model use kind='clip'.
```

A run folder, a single-model final ZIP, or a historical CSV/JSON/JSONL history file
is accepted. A history file alone cannot supply missing saved graphs. Missing
historical values/figures show `N/A`; best-checkpoint predictions can fill missing
best-epoch metrics only. No best predictions are copied to other epochs, no model
is reselected, and no inference or training runs. Saved results are read-only.

In the original notebook, set `SAVED_MODEL_RUNS` and run section9 to replay
tables/figures, or section8 for the six-model comparison. In the face notebook,
use `BASELINE_RUNS`/`BASELINE_ZIPS`, `SAVED_EXPERIMENT_RUNS`, and optionally
`LSTM_REPORT_RUN`, then section6. ZIP/history replay skips the detailed matched
cohort export, which needs extracted run folders and the existing index/pool;
normal completed training retains that existing export. Already displayed runs
are skipped so the automatic report is not repeated.

The six-model comparison uses each model's saved selected best epoch, labels
Frame versus Clip explicitly, and puts Video F2 between Video F1 and DEV Loss.
Untrained models show `Pending`; unavailable historical metrics show `N/A`.
To save enriched historical tables without changing originals:

```python
from cnn_experiment.reporting import export_epoch_report
export_epoch_report(saved, '/kaggle/working/reports/resnet50_new_report')
```

The destination must be new. It contains CSV, JSON and scrollable HTML. Existing
history/checkpoint files are never patched in place. New completed runs retain
Video F2 in their normal history and best-checkpoint summary; calibrated video F2
continues in the existing face-comparison reports. The source changes are local
until separately approved for GitHub. Use the latest reviewed source ZIP with
these notebooks; the old GitHub source cannot supply the new reporting module.

## Two face preprocessing options — face_mask approved as default

Set `PREPROCESSING_MODE` in the face notebook's setup cell to exactly one of:

| Mode | Behavior |
|---|---|
| `crop_30` (reference option) | Calls the original `FacePipeline` with `CropConfig(margin=.30)`. The YuNet detector, primary-face selection, 30% TOTAL square expansion, in-frame shifting, bilinear resize, fallback, cache identity and pixels are unchanged. |
| `face_mask` (default) | Retains the original224x224 RGB frame coordinates and scale. A dense full-face oval/surface hull, expanded outward by6% of its largest dimension, preserves original facial pixels exactly. A4px cosine feather extends OUTSIDE that preserved region into neutral RGB(128,128,128). No crop, warp, central-face-only selection or label-dependent mask. |

There is **no tighter-crop mode**. Existing full-frame Experiment C remains the
separate original head-LR control; it is not a third face-preprocessing option.
No original CNN/LSTM definitions or schedules were changed. Both face-model cells
continue to receive the SAME `PIPELINE`; selecting the final mode before training
therefore applies it to both ResNet50 and EfficientNet-B0. The historical experiment
IDs remain unchanged; the saved `input_pipeline` identity records actual mode and
its complete detector/landmark/mask policy. Do not mix modes when resuming a run.
The100-frame TRAIN audit supported this default. Training still requires explicit
execution of individual cells after source review; this preparation trains nothing.

Dense landmarks come from [Google's MediaPipe Face Landmarker IMAGE task](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python),
pinned to MediaPipe0.10.32 and the version1 float16 task. The ordered full oval
indices follow the [official connections](https://github.com/google-ai-edge/mediapipe/blob/v0.10.32/mediapipe/tasks/python/vision/face_landmarker.py).
The primary landmark face must overlap the original YuNet primary face. Detection
uses RGB pixels and geometry only: no class labels or manipulation annotations.
IMAGE mode avoids video tracking/history, and no alignment/rescaling is applied.

The notebook's section4 keeps the original crop previews and adds a three-column
comparison on the SAME six already-selected TRAIN identities: original, crop_30,
face_mask. It saves PNGs plus `previews.csv`, `previews.json` and `summary.json` in
`SESSION/preprocessing_previews`. Diagnostics include crop area, alpha-weighted
mask area, fully preserved area, feather support, detector outcome, fallback reason,
per-region coverage and clipping warnings. Landmarks for forehead, brows, eyes,
nose, cheeks, mouth, jaw and chin are checked. Source-frame bounds and occlusion
can prevent complete anatomy from being visible; landmark checks are geometric
warnings, not proof that unobserved anatomy exists or is covered. An out-of-frame
estimated landmark produces a clipping warning even if all visible facial pixels
are preserved. Unknown checks remain null, not false.

No-face, missing/invalid-landmark and unmatched-primary-face cases keep the FULL
original frame and record a distinct fallback status in the metadata cache. No
frames are dropped or relabeled. Broken model setup/API/runtime errors raise;
they are not silently converted into detection failures. Mask geometry is cached
separately from original crop metadata, with incompatible identities rejected.
Existing checkpoint input-pipeline and Grad-CAM identity checks remain intact.

The new task is3,758,596 bytes (about3.6MiB), fetched only from its versioned Google
URL with SHA256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
Set `LANDMARK_OVERRIDE` to an attached matching `.task` to avoid downloading it.
It is ignored by Git and is not included in the source ZIP. Metadata caches contain
only landmark coordinates/diagnostics, not a full decoded-image cache.

Setup uses one pinned `opencv-contrib-python==4.13.0.92` distribution, satisfying
MediaPipe0.10.32's declared dependency. `ensure_opencv_runtime(install=True)` first
downloads a compatible wheel, then removes all installed OpenCV providers and
installs only contrib without changing NumPy/PyTorch. If cv2 was already imported
with a different provider, setup stops before changing packages: restart the remote
kernel and execute setup first. `ensure_landmark_runtime(install=True)` preserves
installed support libraries and adds the pinned landmark runtime if needed.
The exact dependency versions are saved in preprocessing diagnostics.

Local bounded verification used the six original TRAIN-preview identities in
6.76 seconds:5 successful masks,1 explicit no-landmarks full-frame fallback and1
landmark-based clipping warning. Successful masks retained about40.6% of the frame
by alpha-weighted area; including the fallback, mean retention was50.5% versus
84.8% for the unchanged crop. Every fully preserved pixel was byte-identical to
its original location, and all outputs were224x224. This tiny set does not estimate
full TRAIN/DEV failure rates. No DEV masking pass, CNN inference/training, pool
regeneration, TEST access, commit, push or external upload was performed.

Local review artifacts: `linked/face_mask_validation/previews/` and
`face_mask_validation/validation_report.json`; focused test log under the parent
project's `outputs/face_mask_focused_tests.txt`.

## Final source review and restoration (24 October 2026)

The approved default is now `PREPROCESSING_MODE='face_mask'`. The100-frame TRAIN
audit measured96 masks,4 full-frame fallbacks, and39.6% mean retained area among
successful masks. This decision changes no6% margin,4px feather, neutral gray,
detector confidence, primary-face rule or full-frame fallback. `crop_30` remains
unchanged and selectable for reference; no additional crop-training cells exist.

TRAIN and DEV share one versioned `PIPELINE.identity`, including every mask/crop
setting, both model hashes, OpenCV/MediaPipe versions, index hash and TRAIN-pool
hash. The same mask reader is used before the original TRAIN-only augmentation;
DEV retains its original no-augmentation full schedule. Both best/last checkpoints
record the identity, and resume compares it against BOTH checkpoint states before
restoring optimizer/scheduler/scaler/RNG/sampler state. Mismatches fail clearly.

New rolling/final ZIPs include `input_pipeline.json`, the SQLite metadata snapshot,
split-specific fallback/clipping summaries, `face_detector.onnx`, its licence, and
`face_landmarker.task`. They also retain all existing checkpoint/history/plot/DEV
prediction artifacts. Restoring a ZIP always creates a NEW directory and verifies
its manifest hashes/CRC. Originals stay untouched. In the new session set:

```python
PREPROCESSING_MODE = 'face_mask'
RESUME_FROM = {'resnet50_face_crop': '/kaggle/working/restored_resnet/last.pt'}
MASK_CACHE_OVERRIDE = '/kaggle/working/restored_resnet/face_boxes.sqlite'
DETECTOR_OVERRIDE = '/kaggle/working/restored_resnet/face_detector.onnx'
LANDMARK_OVERRIDE = '/kaggle/working/restored_resnet/face_landmarker.task'
```

Keep `best.pt` beside `last.pt`. `CACHE_OVERRIDE` is exclusively an original
crop_30 cache, not a mask cache. If no cache override is supplied, the exact same
geometry can be regenerated using the pinned assets without changing the policy.
For fresh independent runs leave `RESUME_FROM={}`; the three new CNN instances
load their official ImageNet files, never the completed CNN best.pt files.

Mask-success cohort reporting now recognizes status `masked`; it must not count
successful masks as fallbacks. Six-model reporting labels mask inputs explicitly,
keeps clip metrics distinct, retains F2, and refuses to fill a mask row with old
crop or unknown-preprocessing results. Old crop histories remain available through
the standalone read-only completed-run report. Graph order and selection rules
are unchanged. There are exactly three scheduled new models.

Kaggle source loading: after the approved push use `USE_GITHUB_SOURCE=True`,
`GITHUB_REF='<full approved commit SHA>'` and Kaggle Secret `Github Token`.
The notebook remains local and untracked. Git checks out the exact commit into a
fresh `/kaggle/working` folder. Keep the existing video, `meta-dataa` and ImageNet
weights inputs. Internet is required for GitHub and pinned dependencies/model
assets unless matching assets are supplied through overrides. The Windows folder
and old `codeeeee` input are not used by the remote kernel. A reviewed source ZIP
remains an optional alternative, with manifest verification.

Restart the remote kernel before changing source. Execute setup, lightweight tests,
fixed-metadata verification, previews, and zero-update GPU preflight in order.
Only explicitly executing A/B/C training cells starts those new models. The local
CPU/source review does not establish compatibility with an untested Kaggle CUDA
runtime: those automatic GPU/loader/numerical checks must still pass there. No
training or TEST execution occurs during this local release preparation.

Dependency verification: the provider replacement passed `pip check` and actual
CPU YuNet/MediaPipe imports. Across the existing100 cached TRAIN audit originals,
all detector records, landmark coordinates, crop and mask pixels and alpha maps
were exactly identical before/after replacement; all100 outputs also matched the
reviewed saved masks. No mask/crop policy or model asset changed. Kaggle/Linux T4
still requires the automatic notebook preflight: a Windows CPU check cannot
establish CUDA or Linux system-library compatibility. If an OpenCV shared library
cannot import, setup fails before training; do not install a second cv2 provider.
