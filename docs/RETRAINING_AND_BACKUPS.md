# Completed baseline, protected retraining and private recovery

## What survives from the completed CNN runs

The two originals in `C:\Users\flipp\Downloads` were found and matched against all
five reported results for each model, using full CSV precision. Neither original
was modified. Byte-identical copies and SHA256 provenance are in the ignored local
`linked/baselines/completed_cnn_20261021/` folder. Previous trained weights are lost.
Old, unrelated experiment checkpoints on local disk are not substitutes for them.

| Verified source | Model | Best epoch by video ROC-AUC | DEV frame ROC-AUC | DEV frame PR-AUC | DEV video ROC-AUC | Epoch 10 TRAIN ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `history.csv` | ResNet50 | 9 | 0.887169 | 0.373575 | 0.831584 | 0.969944 |
| `history (1).csv` | EfficientNet-B0 | 10 | 0.806348 | 0.212499 | 0.739200 | 0.837091 |

`config.HISTORICAL_BASELINE` records these verified metrics and source hashes for
future Kaggle sessions. No missing predictions, confidence intervals, thresholds,
trained weights or metrics have been reconstructed or invented.

The histories confirm rotation: each epoch has 8,000 unique candidates, 4,000 real
and 4,000 fake, from 4,000 videos and 1,000 families, with no within-epoch duplicates.
Both used the same selections. By Epoch 10, cumulative unique frames = 52,779
(16.75% of TRAIN); real coverage = 13.23%, fake coverage = 99.70%. Epoch 10 reused
49.49% previously seen candidates; overlap with Epoch 9 was 10.85%. This is not a
fixed 8,000-frame dataset, but the much smaller fake pool necessarily repeats.
Nearby frames and related videos can remain correlated despite broad coverage.

ResNet50's TRAIN–DEV frame AUC gap at Epoch 10 is 0.085652, compared with 0.030743
for EfficientNet-B0. TRAIN AUC is measured online with augmentation and changing
weights on balanced samples; DEV is fixed and naturally imbalanced. These gaps
are useful warning signals, not a controlled estimate of overfitting by themselves.
ResNet's TRAIN loss keeps falling while DEV fluctuates. Its best balanced DEV BCE
is at Epoch 6, whereas video ROC-AUC selects Epoch 9. At Epoch 9, natural DEV BCE
is 0.257764 but fake-class BCE is 0.970612. The 40,115 real versus 1,631 fake DEV
frames explain why natural loss alone can hide poor fake-class loss/calibration.
Ranking quality, probability calibration and a thresholded decision differ.

ResNet's final-stage trainable parameter count is 14,966,785 versus EfficientNet's
1,130,673. Two warm-up epochs train the head; later epochs train only the original
final stage while BN running statistics stay frozen. This is one staged training
procedure, not two competing experiments. Capacity, limited domain adaptation,
head-only dropout 0.3 and weight decay 1e-4 may affect generalization, but these
histories cannot isolate their causal effects. TRAIN flipping at probability 0.5
is already enabled. No new image compression, resizing or artifact-altering
augmentation was added. The split, family leakage checks and full DEV stay fixed.

ResNet had 2 isolated skipped AMP updates; EfficientNet had 4. Scales backed off
and subsequent epochs completed. ResNet's finite pre-clipping gradient maxima
increased substantially; max-norm-5 clipping remains intact. These observations
do not establish numerical overflow as the cause of the generalization gap.

## Finalized new protocol (supersedes earlier experiment proposals)

This is a combined change to CNN sampling, selection and backbone LR, not an
isolated LR ablation. Historical weights remain lost; existing local code,
checkpoints and results are preserved. The previous working source/notebook were
archived in archive/before_fixed_pool_lstm_20261021_154603.zip outside linked.

| Setting | ResNet50 | EfficientNet-B0 | ResNet50-LSTM |
| --- | --- | --- | --- |
| TRAIN source | Same immutable 20k frame pool | Identical pool and epoch IDs | Separate eligible clip manifest |
| Epoch samples | 4k real +4k fake | 4k real +4k fake | 1,500 real +1,500 fake clips |
| Batch | 32 | 32 | 8 clips; encoder microbatch32 |
| Max epochs / patience | 10 /3 | 10 /3 | 10 /3 |
| Optimizer | AdamW | AdamW | AdamW, temporal head only |
| LR | Head1e-4; final stage1.5e-5 | Head1e-4; final stage1.5e-5 | LSTM/classifier1e-4; encoder frozen |
| Weight decay / dropout | 1e-4 /0.3 | 1e-4 /0.3 | 1e-4 /0.3 |
| Warm-up | 2 head epochs, configurable | 2 head epochs, configurable | None; encoder always frozen |
| Scheduler monitor | DEV video ROC-AUC | DEV video ROC-AUC | DEV clip PR-AUC |
| Checkpoint / early stopping | DEV frame PR-AUC | DEV frame PR-AUC | DEV clip PR-AUC |
| Augmentation | Existing TRAIN flip p0.5 | Existing TRAIN flip p0.5 | Same TRAIN-only flip across all16 frames |
| Initial workers/cache | 2 /0MiB | 2 /0MiB | 0 /0MiB (avoid large shared-memory prefetch) |

All use unchanged safe AMP and max-norm5 gradient handling, finite forward/loss
checks, scale backoff/skipped updates and repeated-instability errors. FP32 fallback
is explicit; AMP->FP32 resume remains supported. CNN BatchNorm statistics stay
frozen; the original final-stage trainable parameter definitions are unchanged.
No dropout0.4, 20-epoch EfficientNet recipe or automatic hyperparameter search.

### Fixed CNN pool

pool.build_pool reads only the 4,000 TRAIN metadata rows and source PTS directly.
It does NOT build/query the full315,058 TRAIN frame index. It keeps one video's
temporary10FPS eligibility arrays at a time and two passes of small quotas/video
metadata. The original nearest-PTS/earlier-tie10FPS grid remains intact.

Guard = (last source PTS-first source PTS)/(source frame count-1), exactly one
frame at measured actual mean source FPS. Distance to every visual interval
boundary must be strictly greater than that guard. VFR uses the measured mean;
this is recorded, not a rounded header FPS. Selected frames are decoded with
FrameReader and checked against exact rational PTS/index, dimensions, corruption
and the half-open visual annotations. Audio-only frames are visually real.
No TEST annotation/path/video is processed after discarding its split marker.

Exactly10k/class is mandatory. Family/video quota sweeps, short-segment-first
selection and farthest temporal spacing spread the pool. Spacing is a preference,
not a hard filter that silently reduces counts. Reports include class/video/family
coverage, selected gaps, represented intervals and every interval without a safe
grid frame. If fewer than10k safely fake frames exist, construction stops with
eligibility_report.json; it never relaxes the guard or labels.

train_pool.json is immutable, SHA256-bound and sufficient for lazy video lookup.
No permanent images are saved. Reattach the SAME manifest in future sessions;
do not rebuild a different pool for the other CNN. Each run ZIP also includes it.

FixedPoolSampler uses family/video sweeps within each shuffled class cycle,
consumes every candidate before starting another cycle, defers within-epoch
boundary duplicates, and interleaves balanced class pairs. Queues, cycle counters,
coverage and seed are checkpointed. Epoch manifests retain exact selected IDs.
With10k/class and4k/class/epoch, all20k candidates are covered by epoch3 and each
is used exactly4 times over10 completed epochs. Repetition is deliberate after
exhaustion; the smaller pool does not guarantee improved generalization.

### Controls and selection

Default profile for both CNNs: fixed20k_lr15e6. The matched fixed20k_lr1e5 profile
uses the same pool/seed/selector and changes only backbone LR. Run it separately,
fresh, for a controlled LR comparison. baseline retains the original full-index
sampler, LR1e-5 and video ROC-AUC selection. resnet_weight_decay_3e-4 remains a
separate baseline regularization option; do not stack it into a claimed one-factor
comparison. Named plans/configs are saved before training. Changing sampling,
selector, model configuration or pool while resuming is rejected.

CNN DEV is exactly the original41,746 frames /500 videos, no augmentation or
resampling. DevSchedule reads only DEV tables from the existing attached index.
Ranking metrics and Top-5% video aggregation stay unchanged. Frame/clip F2 and
video PR-AUC are added to existing loss, class loss, ROC/PR, precision/recall/F1
and confusion reports. PR-AUC is the existing trapezoidal measure; AP is separate.

Final DEV F2 calibration partitions source families deterministically, stratified
by presence of visually fake scheduled inputs. MaxF2 threshold is selected on
calibration families only; metrics/confusion at that threshold and at0.5 are
reported on disjoint reporting families. Equal F2 chooses highest threshold.
There is no fallback to in-sample threshold reporting if separation is impossible.
Both subsets cover DEV, and full-DEV ranking/video results remain available.
Because checkpoint selection used full DEV, reporting families are NOT an untouched
final holdout. Threshold changes cannot improve AUC, PR-AUC or BCE. No threshold
is applied to TEST; that separate future pathway still requires explicit approval.

### Working ResNet50-LSTM

temporal.trained_resnet strictly loads a compatible selected trained ResNet50
best.pt (state keys network.*), removes its classifier AFTER loading, freezes all
encoder parameters and keeps encoder/BatchNorm in eval mode. Missing, mismatched
or non-finite CNN weights are refused; there is no ImageNet fallback. CNN SHA256,
selected epoch/config/criterion, preprocessing and feature microbatch are saved.
The pooled2048 vectors, NOT binary logits, feed LSTM(hidden128,layers1), dropout0.3
and a binary linear head. Hidden size/layers/dropout/microbatch are configurable.
The LSTM/classifier runs FP32 under the shared outer AMP/GradScaler safeguards.

TRAIN clips: separate10FPS source timeline,16 chronological frames, strictstride12,
no padding/cross-video clips, at least ONE fake frame labels fake; audio-only is
real. Class pools rotate without replacement, then reshuffle; duplicates within an
epoch are prohibited and family/video diversity is maintained by cyclic sweeps.
There is no12.5% exclusion. Full DEV clips are copied exactly from the existing
fixed index, including its legacy extra final tail clips. The verified v2 schedule
is3,551 clips (3,075 real /476 fake), not the earlier obsolete8,738 count. The
TRAIN strict grid and unchanged DEV tail exception are recorded explicitly.

Frames decode lazily by batch/video; encoder extraction is chunked. Only a batch's
feature sequence is retained; no full-frame/feature RAM cache. Initial workers0
avoids large clip prefetch in /dev/shm; no per-session cache is enabled by default.
preflight_temporal measures bounded decoding/GPU memory/backward with ZERO updates.
It saves exact clip counts and fails if the fake pool cannot provide1,500 unique
clips. Actual full-pool construction and CUDA execution are deferred to Kaggle.

The original models.EfficientNetLSTM and older local packages remain unchanged as
historical reference. They are not the new ResNet50 temporal pipeline.

## Notebook execution and manual persistence

Use the ONE local experiment1_cnn_v2_vscode_kaggle.ipynb with remote Kaggle Python.
Do NOT Run All. Source still needs a separately approved Git push before GitHub
checkout can fetch it; this work did not commit/push/upload anything.

1. Section1: fresh remote kernel, GitHub fetch (Secret exactly Github Token),
   configuration and automatic lightweight tests.
2. Section2: existing video/meta-dataa/weights inputs; no duplicate video uploads.
3. Section3: validate unchanged DEV; attach POOL_OVERRIDE or create/verify20k pool.
   Download fixed_cnn_pool_manifest.zip and reuse train_pool.json in future sessions.
4. Section4: verify all planned epoch IDs match across architectures.
5. Section4a: OPTIONAL tiny ZIP download check. No confirmation/authentication gate.
6. Section5: automatic bounded loader/head/final-stage/CUDA overflow preflight.
7. Section6 explicitly starts ResNet50; download rolling ZIPs while it runs.
8. Section7 explicitly starts EfficientNet-B0, with the SAME pool and epoch IDs.
9. Sections8-11: compare CNNs, display final-only plots/F2 analysis, inventory and
   manual rolling/final ZIP links. Training metrics remain visible each epoch.
10. Section11a: after a trained selected ResNet exists, validate its identity,
    create/reuse independent clip_manifest.json, then zero-update LSTM preflight.
    Download clip_manifest.zip for reuse. Section11b explicitly starts LSTM.
11. Section12: leave TEST_APPROVAL=None; no held-out access/evaluation.

To download, use Kaggle's Files browser at the printed /kaggle/working path or
the FileLink when the remote VS Code/Jupyter server supports it. Download the ZIP
to Windows, Extract All, optionally verify Get-FileHash -Algorithm SHA256 against
the printed receipt. No code can silently persist a ZIP to your laptop.

Every completed epoch saves best.pt/last.pt, history.csv/jsonl, config and essential
metrics/diagnostics before making a verified rolling ZIP. No epoch graphs. Each
model gets a separate immutable final ZIP including predictions and final figures.
Original videos, candidates.sqlite, pretrained input weights, credentials and source
trees are excluded. Pool/clip manifests ARE small reproducibility artifacts and
included. A failed ZIP build retains the previous ZIP and local checkpoints.

Download while training is still running using the browser Files interface; another
cell on the same busy kernel will wait. One rolling ZIP/model replaces the previous
rolling ZIP; download any epoch version you want to preserve externally. Files left
in /kaggle/working alone are ephemeral, not persistent backups.

### Restore and resume

Attach the downloaded ZIP as a PRIVATE Kaggle input, extract with the helper into
a NEW directory, and keep best.pt beside last.pt:

```python
restored = restore_local_zip('/kaggle/input/my-private-run/resnet50_latest_completed_epoch.zip',
                             SESSION/'restored_resnet50')
POOL_OVERRIDE = restored/'train_pool.json'  # Set before running section3.
RESUME_FROM['resnet50'] = restored/'last.pt'  # Set before section6.
```

Use matching config/profile/precision/DEV index and manifest SHA. Restore helper
checks flat paths, every member hash and CRC and refuses an existing destination.
Resume restores optimizer, scheduler, scaler, RNG, class queues/coverage, history,
early stopping and original best selection; next epoch is completed epoch+1. Output
goes to a NEW run, never the source checkpoint folder. AMP->FP32 is allowed and
recorded; FP32->AMP or a changed selector/pool/config is rejected. Stopped/completed
runs are refused rather than silently restarted. Lost old weights cannot be resumed.

For LSTM also retain/reattach the identical original CNN best.pt, set
CNN_BEST_FOR_LSTM to it, CLIP_MANIFEST_OVERRIDE to the extracted clip_manifest.json
and LSTM_RESUME_FROM to extracted last.pt before11a/11b. Embedded encoder weights
are saved in LSTM checkpoints, but this initial recovery path additionally verifies
the source CNN file identity. Do not silently substitute a newer CNN checkpoint.

Private Kaggle backend code is retained as an unused opt-in Python API; the main
notebook never initializes it or uploads datasets. No Kaggle API credentials,
dummy-download confirmation or remote recovery verification blocks training.

## Estimates and remaining verification

Historical TRAIN+DEV epoch totals were about51.2min ResNet and39.3min EfficientNet
for10 epochs on the former setup. The new pool keeps8k training frames and fullDEV,
so these are rough anchors, not benchmarked predictions. Pool construction has
two TRAIN packet-timeline passes and selected-frame decode verification; decoding
intervening frames may make it take minutes to tens of minutes, potentially longer
on slow inputs. Save the manifest to avoid repeating it. Reports/logs show progress.

Pool JSON should be a few MiB to roughly10MiB; clip JSON roughly10-30MiB depending
on source lengths. No20k-image cache (~2.8GiB uint8 before compression) is built.
The shared-memory/prefetch and GPU probes determine actual safe capacity. Tentative
T4 budgets: CNN batch32 roughly1-5GiB; frozen-LSTM microbatch32 roughly1-4GiB plus
CUDA workspaces. These are estimates, not measured here. Clip input batch8 alone
is~73.5MiB float32; microbatching bounds encoder activations, not the raw clip batch.

LSTM encodes48,000 TRAIN frames plus56,816 scheduled DEV frame positions per epoch
before overlap reuse within decoder batches. That is~2.1x CNN frame positions
(49,746), but frozen-encoder forward, decoding locality and microbatch throughput
prevent a simple2.1x runtime claim. Use11a timing and live first-epoch rates; budget
at least another comparable CNN-run session and allow extra I/O time.

ResNet best+last checkpoints roughly408MiB; EfficientNet~48MiB before logs. Frozen
ResNet-LSTM best+last roughly205MiB including encoder/LSTM/Adam states. Rolling ZIP,
final ZIP, source checkpoints and temporary ZIP each need disk space; reserve a few
GiB total and monitor. Final full DEV reevaluation adds one DEV pass per model.

Actual20k safe-pool feasibility and full temporal clip counts must be verified by
Kaggle construction. Insufficient counts stop clearly. GPU/CUDA/real remote source
authentication and Windows downloads were not tested locally. TEST was never opened.
