# VS Code, GitHub and Kaggle workflow

The maintained notebook is `experiment1_cnn_v2_vscode_kaggle.ipynb`. Keep it locally
in VS Code; it is deliberately excluded from Git. Older notebook files, outputs,
checkpoints and backups remain on disk and are not part of the proposed upload.
The GitHub repository holds the small, reviewed Python dependency set and docs.
No code ZIP or `codeeeee` dataset is needed.

## First session and reconnecting

1. After the reviewed source is approved and pushed, create/start a Kaggle notebook
   session with **Internet enabled** and the **Tesla T4** accelerator.
2. Attach these existing inputs; do not upload duplicate copies:

   | Existing dataset | Required markers |
   | --- | --- |
   | `lav-df-5000-dsa4266` | `metadata.json`, sibling `train/` and `dev/` video directories |
   | `meta-dataa` | `candidates.sqlite` and existing supporting reports |
   | `weights` | `resnet50-11ad3fa6.pth`, `efficientnet_b0_rwightman-7f5810bc.pth` |

   Marker discovery supports nested mount paths. If more than one copy is attached,
   set the relevant path override in the first local notebook cell. The notebook
   requires the existing SQLite file and never invokes the index builder.
3. Create a fine-grained GitHub token owned by `lazisa2004`, restricted to
   `dsa4266-deepfake-detection`, with **Contents: Read-only** and an expiry. Add it
   under **Kaggle Secrets** as `GITHUB_TOKEN`; enable access for the Kaggle session's
   notebook. Never paste it into VS Code source, a shell command, notebook output
   or chat. See [GitHub's token guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
4. In Kaggle's **Run > Kaggle Jupyter Server**, copy the **VSCode Compatible URL**.
   In the local notebook's VS Code kernel picker, choose **Select Another Kernel >
   Existing Jupyter Server > Enter URL**, paste it, then choose the remote Python
   kernel. The server URL can contain a session credential: enter it in the kernel
   picker only, not in notebook cells or Git files. See
   [VS Code's remote Jupyter instructions](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_connect-to-a-remote-jupyter-server).
5. Execute sections 1–5. These are automatic preflight checks. Section 1 verifies
   `/kaggle` paths, CUDA and T4, fetches source into a new working directory and
   prints the commit plus `cnn_v2_amp_safe_resume_v1`. It preserves Kaggle's
   installed torch/torchvision pair and installs only PyAV if missing.
6. Executing sections **6 and 7 starts CNN training**. No full training is launched
   by preparing these files. All diagnostics and epoch plots run by default;
   there are no diagnostic activation flags. The final held-out section is gated.

When a Kaggle session expires, start another session, reattach the same three
inputs, enable the secret again, obtain its new server URL and reconnect VS Code.
Run the same local notebook from section 1 with a fresh remote kernel. No local
videos, local virtual environment or VS Code terminal process are used by its
remote cells. Local source edits affect Kaggle only after a reviewed Git push and
a fresh source checkout. The notebook never uploads local edits automatically.

## Source loading and reproducibility

`GITHUB_REF = 'main'` fetches the latest approved source on that branch. For repeat
runs, replace it with the full approved commit SHA recorded in
`SESSION/git_provenance.json`. A new code directory is used every time. Existing
experiment/test imports cause a clear restart-kernel error rather than using
stale modules. A missing internet connection, bad secret or missing ref stops
setup; there is no fallback to an old attached code dataset.

The token is retrieved from Kaggle Secrets and supplied only to the Git fetch
child's environment via a temporary askpass executable containing no token. Git
credential helpers/global config are disabled for that operation. The executable
is removed and the temporary token environment cleared on success or failure.
Captured Git output is not printed, including error output. Remote URLs remain
credential-free. Only the source URL, requested ref and commit are persisted.

## Unchanged CNN methodology

- LAV-DF-5000, full-frame RGB preprocessing and 10-FPS candidate selection.
- Current audited TRAIN: 315,058 frame candidates, 302,240 real / 12,818 fake.
  The older 785,622 count belongs to an earlier candidate schedule; it is not
  silently substituted for the verified 10-FPS v2 index.
- Each CNN epoch: 4,000 real + 4,000 fake, rotating deterministically from that
  index using the experiment seed and epoch. No duplicate candidate within an epoch.
- Fixed DEV: 41,746 frames, 40,115 real / 1,631 fake, without augmentation/resampling.
- Boundary-near frame inclusion and visual interval labels remain unchanged.
  Existing clip metadata is preserved; no LSTM training is performed.
- ResNet50 and EfficientNet-B0; two frozen-backbone head warm-up epochs, then
  original partial final-stage tuning with frozen BN running statistics.
- AdamW head LR `1e-4`, backbone LR `1e-5`, weight decay `1e-4`, dropout `0.3`,
  TRAIN flip probability `0.5`, batch size 32, at most 10 epochs, patience 3.
- DEV video ROC-AUC selects the best epoch; Top-5% aggregation and thresholds
  `0.5` remain fixed. Balanced DEV loss breaks cross-model selection ties.
- The existing gradient guard preserves AMP skip/back-off behavior. Finite
  gradients use FP64 norm accumulation and max-norm-5 clipping. Invalid FP32
  gradients or repeated AMP instability stop visibly without unsafe updates.

The fresh default is `RESUME_FROM = {}`. `MODEL_AMP` exposes an explicit FP32
fallback. It is never changed automatically. If a later run needs to resume,
reattach its completed `last.pt` and matching `best.pt`, use a new SESSION, and
set the model's `RESUME_FROM` path. Existing resume checks restore optimizer,
scheduler, scaler or inactive scaler provenance, RNG, sampler and early stopping.
Changing precision is supported but not a bitwise-equivalent continuation.
The consolidated `cnn_experiment/checkpoints.py` retains the original resume
validation and restoration logic. Existing state-dictionary checkpoints use the
same parameter names, optimizer ordering, config schema and protocol identifier.

## Automatic diagnostics and outputs

Preflight runs the selected lightweight unit tests, read-only metadata checks,
two metadata-only epoch selections with label checks, a small 64-sample loader
benchmark, the two-element CUDA overflow check, one batch in main/spawned loader
modes per model, and one partial-fine-tuning GPU probe per model. Disposable CNN
probes perform no optimizer updates. These do not decode the full candidate pool.
The loader timing estimate excludes training compute and is not an epoch ETA.

Outputs use this layout:

```text
/kaggle/working/experiment1_cnn_v2/session_<UTC timestamp>_<unique suffix>/
  git_provenance.json
  planned_configs.json
  diagnostics/                  # preflight reports and smoke-test logs
  resnet50/
    best.pt, last.pt
    history.csv, history.jsonl
    progress.jsonl, numerics.jsonl, run_status.json
    training_curves.png, numerical_curves.png
    epoch_diagnostics/epoch_001/ # current DEV confusion/ROC plots and metrics
    best_dev_*                  # best-checkpoint metrics, predictions and plots
    training_summary.json
  efficientnet_b0/              # same structure
  comparison/
    comparison.json, model_comparison.csv
```

Progress prints first/periodic/final batches, TRAIN/DEV losses, gradient norms,
AMP scale, skipped updates, batch/load times and CUDA/parent-process memory.
Epoch summaries include TRAIN and frame/video DEV ROC-AUC, precision/recall/F1,
natural/balanced/class losses, LRs, duration, best status and early-stop count.
Checkpoints/history save before plotting. Plots display inside the remote VS
Code notebook after each completed epoch, including non-best epochs. Candidate
IDs stay in diagnostic files instead of large notebook output lists.

Existing run directories cause an error; rerunning a training cell never resets
or overwrites a run. You can rerun display/inventory cells. Comparison display
reuses its in-memory result; a separate attempt to overwrite a saved comparison
is rejected. Restarting section 1 creates a fresh session, after a kernel restart.
On failure, preserve the run and read `failure_traceback.txt`, `run_status.json`,
`progress.jsonl` and `numerics.jsonl`. A hard kernel/OS kill cannot save a traceback.

## Back up and retrieve the outputs

Run section 11 before expiry, including after an interrupted run. It packages
only that SESSION, including actual checkpoint files; it never packages videos,
input SQLite, weights, `.git` or secrets. It creates a unique ZIP under
`/kaggle/working/experiment1_cnn_v2_backups/`, verifies CRCs and writes a SHA256
manifest. This needs roughly the current SESSION size again in free disk space.

Download the ZIP and manifest using the running Kaggle session's Output/file
browser (or its remote Jupyter file browser). A notebook `FileLink` is offered,
but VS Code may not resolve remote file links; use the remote file browser then.
Store the ZIP on your computer before the session ends. On Windows you can check:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'C:\path\to\downloaded_session.zip'
```

Compare it with the printed/manifest SHA256. Alternatively, explicitly preserve
and verify the files as a private Kaggle output/Dataset for later reattachment.
Do not rely on saving the local `.ipynb`, leaving a ZIP on ephemeral disk, or a
Kaggle Save & Run action that unexpectedly reruns training. Downloading is the
direct backup route for this externally executed workflow.

## Future held-out evaluation

The final notebook section defaults to `TEST_APPROVAL = None` and does nothing
with held-out data. Importing the supported final-evaluation module also performs
no data access. Only after explicit final-evaluation approval may the approval
value be set to the exact phrase documented in that module. The function checks
approval before any file access, requires the completed DEV selection and matched
best checkpoint, then evaluates only the selected CNN with its frozen config.
It never updates weights, tunes thresholds, selects models on TEST, or modifies
the TRAIN/DEV index. A separate new `final_test/` output directory is required.
This pathway has not been exercised on actual held-out data. After any approved
future evaluation, package/download the updated SESSION again.

## What has and has not been verified locally

Lightweight tests cover AMP skips/resume, persistence, epoch artifact generation,
path discovery, credential handling with mocked Git/Secrets, output ZIP integrity
and the final-evaluation approval guard. Notebook cells are syntax checked.
The private remote was reachable with existing Windows Git Credential Manager.
Kaggle's actual secret, T4 smoke checks and external VS Code connection must be
verified by running sections 1–5 after the approved source is pushed. No full CNN
training, real held-out evaluation, Kaggle session creation or push is performed
as part of preparation.


## Consolidated package and local preservation

The only active source package is `cnn_experiment`. The 21-file GitHub proposal
includes 10 package files, 6 test files, this workflow guide and 4 root supporting
files. The local notebook is not uploaded. Its automatic test cell runs
`tests.test_data`, `tests.test_training_state`, `tests.test_training_progress`,
`tests.test_workflow` and `tests.test_github`; these retain all 40 prior checks.
The GitHub helper is `cnn_experiment/github.py` and its standard-library source
is embedded in the notebook so it can fetch the package before project imports.

Original code, old tests and historical documentation stay local and ignored.
A timestamped ZIP under `archive/before_compact_refactor_*` preserves their exact
pre-refactor contents, including the previous main notebook. Additional local
before/after audits live in ignored `outputs/`; these are not needed to run the
uploaded source. No legacy package or old notebook is required by the new package.

The refactor moves code rather than changing algorithms. Sampling sequences,
labels, preprocessing, parameter names, optimizer parameter ordering, AMP behavior,
resume state and DEV metric calculations are compared against the preserved code.
The models' `network.*` state-dictionary namespace and saved configuration schema
remain unchanged. Original class-module paths change, but existing pipeline
checkpoints contain state dictionaries rather than pickled model instances.

The combined data module imports PyTorch when imported, including for metadata
helpers. This changes import overhead only; it performs no CUDA work, image decoding
or index building on import. Final evaluation still checks approval before importing
that module or reading metadata. Its import alone remains free of data/GPU access.
