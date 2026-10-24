# Experiment F: independent ResNet50 full frame

Prepared locally while D/E are running. No existing notebook, shared module,
configuration, source pin, checkpoint, run directory or cache is edited. Do not
execute this experiment or change the active kernel/source until current training
finishes and all rolling/final ZIPs have been downloaded.

F uses the original `cnn_experiment` full-frame loader/decoder and ImageNet
normalization. It does not import or use the face package, YuNet or MediaPipe.
`train(..., input_pipeline=None)` explicitly selects the original full-frame path.

| Setting | F |
|---|---|
| ID | `resnet50_full_frame_head5e5_roc_f` |
| Architecture / input | ResNet50 / full_frame |
| Head / backbone LR | 5e-5 / 1.5e-5 |
| Epoch cap / patience | 15 / 5 |
| Checkpoint and early-stopping monitor | DEV frame ROC-AUC |
| LR scheduler monitor | DEV video ROC-AUC, unchanged |
| Head-only warm-up / batch / dropout | 2 epochs / 32 / 0.3 |
| Initialization | Independent original ImageNet weights |

All other baseline `fixed20k_lr15e6` settings are inherited unchanged: seed4266,
fixed20k TRAIN pool, 4k/class per epoch, augmentation, final-stage fine-tuning,
frozen BN, AMP safeguards, full DEV, labels, timestamps, and Top-5% mean video
pooling. F verifies the exact approved pool/index hashes; it never rebuilds them.
Experiment E remains face_mask and is never renamed or reused as F.

## CPU verification now (separate local process)

From `linked`, using the project Python environment:

```powershell
..\.venv\Scripts\python.exe -B -m unittest experiment_f.test_runner -v
```

Tests use synthetic arrays/metadata and mocked GPU/train functions. They do not
execute an actual CNN training loop or contact Kaggle. No existing notebook is
opened and saved. `-B` avoids changing local bytecode caches.

## Integration AFTER current training finishes and after source approval

No notebook cells have been added. Later, after backing up D/E, approve a source
release containing these four files and the necessary narrow `.gitignore`
allowlist entries. They are currently ignored by the existing allowlist. No
existing file needs changing for local validation. No shared module change is
needed for F, including its reports or ZIP recovery.

The future approved `.gitignore` addition would be:

```gitignore
!/experiment_f/
/experiment_f/*
!/experiment_f/__init__.py
!/experiment_f/runner.py
!/experiment_f/test_runner.py
!/experiment_f/README.md
```

Do not apply or stage these entries during active training. The current GitHub
commits and notebook pins remain unchanged.

After that approved push, change the existing local notebook's `GITHUB_REF` to
the new FULL release SHA with `USE_GITHUB_SOURCE=True`, preserving outputs. This
must happen only after current training finishes. The currently pinned D/E commit
does not include F. Restart into a fresh remote kernel, load the approved source,
and reuse existing input/pool setup to establish `INDEX_PATH`, `POOL_PATH`, `DATA`,
`WEIGHTS` and `PROVENANCE`. Skip all completed-model training cells. Do not reload
modules or pull source into the currently running session.

For an F-only run, skip the face-preview, face-GPU-preflight and D/E setup cells;
F's own preflight below uses the original full-frame data path. Do not pass the
notebook's `PIPELINE` or change `PREPROCESSING_MODE` to repurpose an existing run.

Add this NEW setup/preflight cell later. It uses original full-frame checks and
disposable models with zero optimizer steps; no face pipeline is supplied:

```python
from experiment_f.runner import create_f_output, preflight_f, train_f, display_f_run

F_OUTPUT = create_f_output('/kaggle/working/experiment1_f')
F_PREFLIGHT = preflight_f(INDEX_PATH, POOL_PATH, DATA, WEIGHTS, F_OUTPUT)
print('F output:', F_OUTPUT)
```

Then add a separate, explicitly executed training cell:

```python
F_RUN = train_f(
    INDEX_PATH, POOL_PATH, DATA, WEIGHTS, F_OUTPUT,
    preflight=F_PREFLIGHT, provenance=PROVENANCE,
)
```

This starts F from ImageNet, displays one complete19-column epoch table after
completion, yellow best-row highlighting, then the existing graphs in their
existing order. The shared frame-ROC branch marks best epochs and saves all
completed epochs, frame/video ROC/PR, precision/recall/F1/F2 at0.5, losses and LRs.
Each completed epoch creates its own rolling checkpoint/history ZIP. The final
ZIP additionally contains `epoch_metrics.csv/json/html`, plots and best-checkpoint
DEV artifacts. No extra inference is introduced.

Files live at:
`/kaggle/working/experiment1_f/<F_ID>/session_<unique>/run/` and sibling `downloads/`.
Download the ZIPs before expiry. ZIP files on Kaggle disk are not an external backup.

Replay existing F results without retraining:

```python
display_f_run('/kaggle/input/<your-completed-f-dataset>/<f-final.zip>')
```

For future resume, restore ONLY a verified F ZIP into a new directory using
`cnn_experiment.backup.restore_local_zip`, create a new F output and matching
preflight, then pass `resume_from=restored/'last.pt'` to `train_f`. Keep companion
`best.pt`. Existing optimizer/scheduler/scaler/RNG/sampler/early-stop restoration
is reused. F checks experiment identity and full-frame preprocessing first, so
masked E or other models cannot be relabelled as F.

## Standalone entry point (future Kaggle execution only)

```bash
python -B -m experiment_f.runner --index /path/candidates.sqlite \
  --pool /path/train_pool.json --data /path/LAV-DF-5000 \
  --weights /path/imagenet_weights
```

Default is preflight only. Add `--train` explicitly to train after it passes, and
optionally `--resume-from /path/restored_F/last.pt` for an F-only resume. Use a
separate process only after active training finishes. No TEST execution is exposed.
