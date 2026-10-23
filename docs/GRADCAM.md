# Grad-CAM on saved CNN predictions (DEV now; TEST locked)

Use `cnn_experiment/gradcam.py` as both an importable utility and a small demonstration
script. No extra notebook is required. The original main notebook and its TEST
section are unchanged. Nothing trains, selects a model, changes thresholds, builds
an index, or reruns full inference here.

## Inputs and paths

Use **best.pt and best_dev_predictions.csv from the same completed model run**.
The downloaded final model ZIPs already contain both. `last.pt` is for resuming;
do not substitute it for the checkpoint that produced the best DEV predictions.

| Input | Local project | Existing Kaggle session/input |
|---|---|---|
| Existing frame index | `linked/metadata/candidates.sqlite` | `/kaggle/input/datasets/leezile/meta-dataa/candidates.sqlite` |
| Videos root | `LAV-DF-5000` beside `linked` | `/kaggle/input/datasets/leezile/lav-df-5000-dsa4266/LAV-DF-5000` |
| ResNet50 run | Extract selected files from your downloaded ResNet final ZIP | `/kaggle/working/experiment1_cnn_v2/session_20261021_161719_UTC_cc5b9e80/resnet50` |
| EfficientNet run | Extract selected files from your downloaded EfficientNet final ZIP | Same session, subfolder `efficientnet_b0` |
| Weights/predictions | `RUN/best.pt`, `RUN/best_dev_predictions.csv` | Same names beneath the model run |
| New figures | Any **new** output folder | e.g. `/kaggle/working/gradcam_dev_resnet50_tp` |

Session working directories disappear when Kaggle expires. In a new session,
attach your saved model ZIP/extracted checkpoint dataset as a private input and set
`RUN` accordingly. The old working path is not persistent storage. ImageNet weights
are unnecessary: model construction uses `pretrained=False` then strict trained
checkpoint loading. Dependencies are the existing `requirements.txt` (torch,
torchvision, PyAV, numpy, matplotlib, Pillow); no Grad-CAM library or downloads.

DEV remains the exact fixed 41,746 frames (40,115 Real + 1,631 Fake) from 500 videos.
`DevSchedule.records('dev', 'frame')` loads the existing combined SQLite index.
CSV IDs, rational timestamps, paths, labels, unique/full coverage and the original
checkpoint's index SHA256 are checked. This requires the original combined index,
not the smaller DEV-only export with a different file hash. Neither the 20,000-frame
TRAIN pool nor LSTM clip manifests are involved.

## Run the DEV demonstration

From `linked`, with its Python environment selected:

```bash
python -m unittest tests.test_gradcam -v
python -m cnn_experiment.gradcam --help
python gradcam_dev_demo.py --run RUN --index metadata/candidates.sqlite --data-root ../LAV-DF-5000 --output gradcam_dev_demo
python -m cnn_experiment.gradcam --split dev --checkpoint RUN/best.pt --predictions RUN/best_dev_predictions.csv --index metadata/candidates.sqlite --data-root ../LAV-DF-5000 --category fp --count 2 --seed 4266 --output gradcam_dev_resnet50_fp
```

Replace `RUN` with the ResNet50 or EfficientNet completed run; architecture/dropout
are read from the checkpoint. Optional `--device cuda` uses your GPU for explanation
only. The CPU default is adequate for a handful of examples. New output folders
are required so existing figures and results cannot be overwritten.

The small `gradcam_dev_demo.py` script picks one available example per category
(deduplicated), then explains all Top-5% contributors for the first selected video's
saved score. It has no TEST option. Repeat with the other model's run and a new folder.

Categories: `tp`, `tn`, `fp`, `fn`, `high_confidence_correct`,
`high_confidence_incorrect`. Confusion cells use `p >= threshold`; default threshold
is the checkpoint's recorded DEV threshold (0.5 for these runs). Ordinary confusion
cells use seeded sampling; high-confidence groups sort by confidence in the predicted
class, with seeded tie handling. Empty categories report clearly. This selects
examples for illustration; it does not estimate performance from a balanced subset.

For an optional already-frozen DEV threshold, pass `--frozen-policy POLICY.json`:

```json
{"selection_source": "dev", "checkpoint_sha256": "THE_SELECTED_BEST_PT_SHA256", "threshold": 0.5, "topk_fraction": 0.05}
```

Choose/record any alternate threshold using DEV only. This utility never optimizes it.

### In the existing VS Code notebook with the Kaggle kernel

After your normal source/input setup, use an additional DEV-only scratch cell:

```python
from pathlib import Path
from cnn_experiment.gradcam import load_explanation_data, select_examples, save_explanations

RUN = Path('/kaggle/working/experiment1_cnn_v2/session_20261021_161719_UTC_cc5b9e80/resnet50')
INDEX = Path('/kaggle/input/datasets/leezile/meta-dataa/candidates.sqlite')
VIDEOS = Path('/kaggle/input/datasets/leezile/lav-df-5000-dsa4266/LAV-DF-5000')
data = load_explanation_data(RUN/'best.pt', RUN/'best_dev_predictions.csv',
                             split='dev', index_path=INDEX)
selected = select_examples(data, 'fp', count=2, seed=4266)
folder = save_explanations(data, RUN/'best.pt', VIDEOS,
                          '/kaggle/working/gradcam_dev_resnet50_fp', selected,
                          device='cuda', selection='fp')
from IPython.display import Image, display
for image in sorted(folder.glob('frame_*.png')):
    display(Image(filename=str(image)))
```

The new module is currently **local only**. Your GitHub bootstrap cannot fetch an
uncommitted module. Until a separate approved push, copy `cnn_experiment/gradcam.py`
into `CODE_ROOT/'cnn_experiment'` on the remote kernel (for example using VS Code's
remote file upload). Only this new module is needed with the existing fetched package;
do not replace the training modules or local main notebook. Upload the new test file
under `CODE_ROOT/'tests'` if running its tests remotely. Never assume the Kaggle
kernel can import directly from a Windows folder.

## Top-5% video explanations

```python
from cnn_experiment.gradcam import top5_contributors
source = data.records[selected[0]]['source_file']
video = top5_contributors(data, source)
print(video['k'], 'of', video['total_frames'], 'frames; video Fake p=', video['probability'])
folder = save_explanations(data, RUN/'best.pt', VIDEOS,
                          '/kaggle/working/gradcam_dev_resnet50_video',
                          [p['record_index'] for p in video['contributors']],
                          device='cuda', selection='top5_contributors', video=video)
```

CLI equivalent: use `--video dev/EXACT_VIDEO.mp4` instead of `--category`.
This explains **all** contributors (the CLI ignores `--count` in video mode).
For N saved frames the video probability is the arithmetic mean of the largest
`k=max(1, ceil(0.05*N))` frame probabilities. Each selected frame has weight `1/k`.
This is verified against the existing `evaluation.video_predictions` implementation.
Ties use exact timestamp then record index; a different tied-frame choice has the
same video score. All scheduled frame scores enter ranking, not only the selected
example subset. It is not max pooling. The heatmaps explain each contributor's
Fake logit, not a single joint spatiotemporal/video Grad-CAM.

## Output and interpretation

Each PNG is a triptych: native original RGB, normalized heatmap, overlay. Titles
include model, relative video ID, timestamp, GT, Fake probability and frozen threshold.
`frames.csv` saves exact timestamps/source indices, IDs, labels, saved evaluation
probabilities, recomputed FP32 probabilities/differences, target layer, zero-map status
and Top-5% membership when requested. `provenance.json` records input hashes,
thresholds and module version hash. Video mode also writes `top5_contributors.csv`
and `top5_video.json` with **all** contributing scores/times and the full pooled score.
Partial failures raise normally and save `failure.json` when the output folder exists.

ResNet50 target: `model.network.layer4[-1]`; EfficientNet-B0 target:
`model.network.features[-1]`. Both provide 7x7 spatial features for 224x224 frames.
Gradients are taken with respect to the **Fake pre-sigmoid logit**, not the Real
class or a thresholded probability. Average spatial gradients weight channels;
ReLU keeps positive Fake evidence; bilinear upsampling gives a 224x224 display.
The reported probability is sigmoid(logit). The logit avoids sigmoid saturation.
This follows the [Grad-CAM paper](https://arxiv.org/abs/1610.02391).

`model.eval()`, FP32, a fresh gradient-enabled input, `torch.autograd.grad`, and
per-call forward-hook cleanup support frozen backbone layers without updating any
weights, BN buffers or parameter gradients. Original mixed training flags are restored
on success/failure. Normalization is delegated directly to the existing LazyDataset,
no resize/crop/flip is added. Decoder PTS, index and native-size checks remain strict.

A zero map is a valid absence of positive Fake evidence at the chosen layer (often
possible for Real predictions); it is recorded rather than stretched into a false
hotspot. Maps are normalized per image, so colours are not comparable evidence
magnitudes across images. These are coarse model explanations, not pixel-level
ground-truth manipulation masks or proof of a causal mechanism. Large discrepancies
between saved and recomputed probabilities warn of possible provenance problems;
small ones may arise from saved AMP inference versus FP32 explanation.

## TEST later: two independent conditions, no implementation edits

**Do not execute now.** Importing the utility or calling its default DEV interface
never reads TEST. Every TEST entry point checks the same explicit approval phrase
as the existing final evaluation **before filesystem or checkpoint reads**.

1. Freeze final checkpoint, DEV threshold and Top-5% rule using the existing project
   process; authorize and complete final evaluation separately.
2. Use its existing `final_test/frozen_policy.json`, `test_frame_predictions.csv`,
   `test_video_predictions.json`, and `test_metrics.json`. The approved model's
   checkpoint SHA256 and completed frame/video coverage must agree. Both CNNs are
   supported as architectures; only the final model actually authorized/evaluated
   can be explained on TEST. This utility does not authorize testing both models.
3. Call the same `load_explanation_data(..., split='test',
   frozen_policy_path=FINAL_TEST/'frozen_policy.json', approval=FINAL_TEST_APPROVAL)`
   after importing `FINAL_TEST_APPROVAL` from `cnn_experiment.final_evaluation`.
   Supply the resulting TEST prediction CSV and exact frozen checkpoint. No index
   generation or new frame selection occurs: persisted evaluation timestamps form
   the TEST frame schedule. Source indices for selected figures are resolved from
   those exact timestamps with the existing `packet_timeline` function.
4. Use `select_examples`, `top5_contributors`, `save_explanations` unchanged. Confusion
   analysis reads TEST labels only behind the approval gate. The existing final
   protocol has one frozen `threshold` for frames/videos; this utility honors that
   recorded value, never the optional DEV calibration output by assumption.
5. Save report figures. Do not use TEST heatmaps, errors or confidence to select
   models/thresholds, retune or retrain.

**Existing final-evaluation limitation found during inspection:**
`final_evaluation.evaluate_final_test` currently calls `data.make_loader`, whose
`FrameReader.read` rejects sources outside TRAIN/DEV. Thus that routine needs a
separately reviewed TEST-decoder correction before a real final evaluation can
complete. Its code and original notebook gate were deliberately **not modified**
in this task. Grad-CAM's narrow, explicitly approved TEST reader duplicates the
same exact-index/PTS/native-RGB checks and reuses LazyDataset normalization; it
does not relax the shared training/DEV decoder. No real TEST execution has been
validated or performed. This does not affect DEV Grad-CAM.

## Local validation results (23 October 2026)

The original downloaded final model ZIPs were read without modification; only
`best.pt` and the saved DEV prediction CSV were temporarily extracted. Both models
matched the original 41,746-frame/500-video index. Six seeded category examples per
model plus all contributors for one video per model produced **24 PNGs**, CSV
metadata and hash provenance under `linked/gradcam_dev_examples/`. The same files
are bundled in `linked/gradcam_dev_examples.zip` (about 10 MiB), containing no
checkpoints or raw videos. Example figures were visually inspected.

| Check | ResNet50 | EfficientNet-B0 |
|---|---:|---:|
| Category figures | 6 | 6 |
| Video contributor figures | 4 of 74 frames | 8 of 158 frames |
| Saved Top-5% video probability | 0.940603 | 0.947547 |
| Largest saved-vs-FP32 probability difference | 0.002642 | 0.006316 |
| CPU validation time including extraction/loading | 25.5 s | 15.4 s |

These times used two CPU threads and a small selected subset, not full inference.
The main notebook, original source modules, index and original final ZIPs were
hash-checked unchanged during validation. Full details: generated
`gradcam_dev_examples/validation_report.json` and
`outputs/gradcam_dev_validation_receipt.json` in the parent project.

All **9 new Grad-CAM tests passed**, covering the analytical formula, both real
architectures with fully frozen parameters, strict checkpoint round-trip, preserved
weights/gradients, zero maps, non-finite detection, hook cleanup, exact CSV/index
joins, category selection, Top-5% ceiling/mean/ties and authorization-before-I/O.
The authorized interface test uses synthetic artifacts only, no held-out files.
The complete lightweight suite ran **86 tests: 84 passed, 1 skipped, 1 existing
error**. The error is `test_github.py:93`: its local notebook test calls
`startswith` on a list-form notebook cell source. That unrelated test and the
saved notebook were left unchanged. Log:
`outputs/gradcam_full_lightweight_tests_final.txt` in the parent project.
