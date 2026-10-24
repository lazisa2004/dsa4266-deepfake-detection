# New experiments by model family

These are NEW independent runs. Historical A/B/C/D/E/F configurations, run
folders, checkpoints and comparison tables are not renamed or reused.
Importing this package does not run an experiment or access data.

| ID | Model | Explicit input | Head LR | Backbone LR | Selection / early stopping |
| --- | --- | --- | --- | --- | --- |
| 1a | ResNet50 | full_frame | 1e-4 | 4.5e-6 | DEV frame ROC-AUC |
| 1b | ResNet50 | face_mask | 1e-4 | 4.5e-6 | DEV frame ROC-AUC |
| 2a | EfficientNet-B0 | full_frame | 2e-4 | 1.5e-5 | DEV frame ROC-AUC |
| 2b | EfficientNet-B0 | face_mask | 2e-4 | 1.5e-5 | DEV frame ROC-AUC |
| 3a | ResNet50-LSTM | full_frame_clips | 3e-5 temporal | fully frozen | DEV clip ROC-AUC |

All have a maximum of 15 epochs and early-stopping patience 3. CNNs retain
two head-only warm-up epochs, final-stage fine-tuning, frozen BN statistics,
batch size 32, dropout 0.3, augmentation and independently loaded ImageNet
weights. The CNN LR scheduler still monitors DEV **video** ROC-AUC, after
warm-up, independently of frame ROC-AUC checkpoint selection.

3a preserves the original temporal architecture, batch size 8, hidden size
128, 1 LSTM layer, dropout 0.3, 16 frames, stride 12, 10 FPS, 1,500 clips per
class per epoch and frozen encoder. Its original scheduler monitor remains
DEV **clip PR-AUC**; checkpoint selection and early stopping use clip ROC-AUC.
The unused inherited `backbone_lr` field does not create an encoder optimizer
group: only the temporal LSTM/classifier parameters are trainable.

## Execution in the existing local face notebook

The original cells are preserved. New cells are appended under sections 11-12.
After a future separately approved source release, restart the remote kernel,
fetch that approved commit using the existing `GITHUB_REF` setting, and run
the existing setup sections 1-3. They resolve the original data, pretrained
weights and verified fixed TRAIN pool. Skip the historical preview/preflight
and A/B/C/D/E/F training cells. Do not use Run All.

1. Run **11. New experiment plan and automatic code tests**.
2. Run **11.1 1a preflight**, then explicitly run **1a training**.
3. Review 1a's cumulative table and graphs; download its ZIP before the session
   expires. Stop here before deciding on other experiments.
4. Optionally run **1b preflight**, then its separate training cell.
5. Optionally run **3a preflight**, then its separate training cell.
6. After ResNet review, run **2a preflight**, then its training cell.
7. Optionally run **2b preflight**, then its separate training cell.
8. Run **12. New-plan comparison** whenever desired; unrun models stay Pending.

Each preflight/training cell refers to exactly one ID. No loop launches
multiple experiments. The notebook requires a completed/saved new 1a result
before another new experiment can start; the review decision remains yours.

The setup's `ORIGINAL_RESNET_BEST` must be explicitly set before optional 3a
to the original trained full-frame ResNet50 `best.pt`, not 1a, F, ImageNet-only
weights or a masked model. Set `ORIGINAL_RESNET_SHA256` when its known digest
is available. The checkpoint's SHA is recorded and checked between preflight
and training. Reattach the original `clip_manifest.json`; set
`ORIGINAL_CLIP_MANIFEST` if automatic discovery is ambiguous. Missing artifacts
fail clearly rather than generating a replacement protocol or encoder.

The appended setup requires this new package. A stale source pin produces a
clear error. This local implementation does not change any existing notebook
source pin, commit or push source, or launch training. A future approved push
and explicit pin update are needed before fetching these new cells' source
from GitHub. Existing Kaggle video/metadata/weights inputs remain attached.

## Isolation, recovery and reporting

`new_session()` creates a unique session. Each `output_path(session, id)` is
`<session>/<id>/run`; its parent has independent assets/preflight, provenance,
plans and downloads. Existing runs/plans are refused. The original fixed pool
and metadata SHA identities are required. Full-frame runs use the original
loader, resizing and normalization and never initialize YuNet or MediaPipe.
Only 1b/2b explicitly request the unchanged face-mask policy for TRAIN and DEV.

The shared engine saves best/last checkpoints, optimizer/scheduler/scaler/RNG/
sampler state, epoch histories, live progress and numerical diagnostics.
Rolling per-epoch and final ZIPs are independent for each ID. Download them;
Kaggle working storage is not persistent protection by itself.

After each model finishes, `display_result` reuses the saved predictions and
displays one complete 19-column cumulative epoch table (frame or clip), yellow
selected-best highlighting, then the existing graphs in their existing order.
ROC-selected graphs retain all completed epochs and best-epoch markers.
Threshold-0.5 precision, recall, F1/F2 and frame/clip and video ROC/PR-AUC are
retained; no threshold or video pooling changes are made. Tables, graphs,
predictions and selection identity are included in the final ZIP.

For replay, set `SAVED_NEW_RUNS` to a mapping from exact new IDs to saved run
folders or completed ZIPs before the new setup, then use
`display_result(id, SAVED_NEW_RUNS[id])`. Config and saved ID checks reject
historical results instead of relabeling them. For resume, restore the ZIP to
a NEW directory using the existing recovery utility, set
`NEW_RESUME_FROM[id]` to its `last.pt`, run this session's preflight and execute
only that ID's training cell. The source checkpoint stays untouched.

`duplicate_audit()` compares actual old definitions. 2a shares C's architecture,
full-frame input and learning rates, but changes **10 to 15 maximum epochs**
and **frame PR-AUC to frame ROC-AUC selection**. It has a new ID/output folder.

## Validation

From the project root with the existing dependencies:

```powershell
python -B -m unittest tests.test_model_family_plan
python -B -m unittest discover -s tests -t .
python -B -m unittest experiment_f.test_runner
```

Tests use temporary synthetic metadata/tensors and mock epoch computation.
They do not train a real model or access held-out data. GPU/decode throughput,
real optional mask assets and the original trained encoder are verified only
when the respective bounded Kaggle preflight is explicitly executed.

No new TEST pathway is introduced; the existing protected TEST gate is unchanged.

## Additional comparison and interpretability sections

Seven further local notebook cells are appended after the original 50; those 50
cells, their execution counts, metadata and outputs remain intact. New sections:

- **13A. New Experiment Comparison:** five new IDs, full saved-metric columns.
- **13B. Overall Master Comparison:** thirteen distinct identities grouped by
  ResNet50, EfficientNet-B0, then ResNet50-LSTM. Includes original baselines,
  A/B/C/D/E and all five new IDs.
- **14. DEV winner Grad-CAM:** a separately executed bounded inference cell;
  never called by training or comparison.

After the original setup and new-plan setup (cell33), run **cell52** once for
comparison imports/tests, then **cell53** for the new table and **cell55** for
the master table. Re-run 53/55 as runs finish. **Cell57** explicitly requests
Grad-CAM only when you want it. Do not Run All or rerun completed models.
The earlier compact comparison cell50 stays intact; it is not required for
these new sections. No GitHub source pin is changed by this addition; source
release and pin updates still require separate approval.

`comparison.collect_sources(globals(), MASTER_SOURCES)` refreshes actual notebook
run references each time. `MASTER_SOURCES` overrides use these exact keys:

```python
MASTER_SOURCES = {
    # Supply actual restored folders or final ZIPs when available, for example:
    # 'original_resnet50': Path('/kaggle/input/<your-run>/resnet50_final.zip'),
    # 'D': Path('/kaggle/input/<your-run>/efficientnet_b0_face_mask_roc_d/run'),
    # '1a': NEW_RUNS['1a'],
}
```

Other keys: `original_efficientnet_b0`, `original_resnet50_lstm`, `A`, `B`, `C`,
`E`, `1b`, `2a`, `2b`, `3a`. One physical artifact/best checkpoint cannot be
assigned to two IDs. Historical F was a superseded planned standalone recipe;
it is preserved locally, not counted as a performed run or relabelled as 1a.
The requested master registry has exactly thirteen identities.

Historical folders are not guessed from similar names. The local notebooks
and nearby folders contain different experiment versions. The included one-time
historical import comes from the original main notebook's existing rendered
comparison (zero-based cell19, output1; completion logs cells15/17). It records
the source notebook SHA256, cell ID, output index, completion-log locations,
each raw table row and displayed precision. These two saved results have best
epochs8 and10; they are not silently mixed with the older reference best epoch9
or similarly named baseline directories with different metrics/protocols.
LR/config facts unavailable in the saved table remain N/A. Missing weights are
explicit and these metric-only rows cannot enter the verified ranking.

To import another compatible saved notebook table once, without retraining:

```python
from model_family_plan.comparison import import_notebook_table
path = import_notebook_table(
    'saved_original_notebook.ipynb', 19, 1,
    {'resnet50': 'original_resnet50',
     'efficientnet_b0': 'original_efficientnet_b0'},
    'NEW_audited_original_results.json',
    threshold=0.5, selection='frame_pr_auc', completed_log_cells=(15, 17))
```

Indices and mappings must match that notebook's ACTUAL saved outputs. The
importer refuses missing completion logs and never infers a best epoch. Keep
the source notebook so its hash/locator can be audited. The new local notebook
already embeds its generated audit JSON, so these two historical metric rows
need no extra dataset upload. Their standalone copy is local-only under
`local_comparison_inputs/`. Additional audited JSON inputs may be supplied as
`EXTRA_AUDITED_IMPORTS` before cell52. They are metric-only records; config,
checkpoint, index and preprocessing identities are verified from restored run
artifacts, which take precedence over historical metric imports.

`build_comparison` loads the selected epoch and actual saved best metrics,
derives each level's F2 from its OWN saved P/R or confusion counts, and uses
saved video probabilities to recompute missing/alternate-threshold video
metrics at0.5 without inference. Unit values at a different/unknown threshold
stay N/A in the0.5 table; separately labelled calibrated results retain their
own thresholds and scopes. Video F2 never comes from frame/clip F2.

Completed status needs a matching final summary and full completed history.
Partial runs remain Partial even with best.pt. Verification checks canonical
configuration/preprocessing, selected epoch/score, exact DEV index/manifest,
Top-5% policy, finite tensors and strict architecture/state loading. Original
historical completion cannot be guessed without evidence: unsupplied old runs
show `Pending verification / Unknown (not attached)`; new unrun IDs show Pending.
Missing/invalid weights cannot rank. Completed metric-only imports stay visible
with Missing checkpoint. All missing/inapplicable metrics are N/A.

Each comparison call saves a NEW CSV/JSON/HTML snapshot, with scrolling and
yellow best valid performance values (minimum losses; maximum scores). The
verified ranking uses video ROC-AUC then PR-AUC; PR-AUC, F1 and F2 leaders are
reported separately. A leading reported completed result with unavailable
weights blocks substitution. Missing PR-AUC in a highest-ROC tie blocks an
invented tie-break. Original checkpoint selectors are never rewritten by ranking.

Grad-CAM automatically loads the overall winning actual best.pt, verifies saved
DEV predictions against the existing index and uses TP/TN/FP/FN selections at
0.5 (one per available category by default). Missing categories are logged.
Face-mask checkpoints require their saved assets and exact saved policy, and
only a NEW interpretation cache is written. No detector settings are tuned.
ResNet targets `network.layer4[-1]`; EfficientNet targets `network.features[-1]`.

The temporal attribution forward alone bypasses the original encoder no_grad
block: input gradients pass through the frozen encoder into the unchanged
LSTM/classifier's Fake logit. Encoder requires_grad flags remain frozen, no
parameter .grad is accumulated, and no optimizer is created. All16 frames retain
chronology. Temporal maps use one common positive maximum per clip; clip GT and
clip probability are labelled as CLIP quantities, not frame predictions.
Weight/state and original forward-probability equivalence are CPU-tested.

PNG triptychs (actual input, heatmap, transparent overlay), CSV timestamps,
categories, level-specific GT, probabilities, gradient norms, zero-map flags
and provenance go to a NEW interpretability folder. Zero positive maps are
reported, not replaced with made-up evidence. Probabilities differing from
saved predictions by more than0.01 stop for provenance review. Hooks and mixed
evaluation/training flags are restored. Existing checkpoint bytes and source
run directories are never written. No real attribution is generated until you
explicitly run cell57 with an available verified winner.
