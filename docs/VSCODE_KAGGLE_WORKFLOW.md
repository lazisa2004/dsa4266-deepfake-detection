# VS Code, private GitHub and Kaggle

Open linked/experiment1_cnn_v2_vscode_kaggle.ipynb locally. Keep only this main
interface; Python executes on the selected Kaggle remote Python3 kernel.

1. After separate approval/push of the reviewed source, enable Kaggle Internet/T4.
2. Reattach existing lav-df-5000-dsa4266, meta-dataa and weights datasets. Required
   markers: metadata.json beside train/dev videos, candidates.sqlite, and official
   resnet50-11ad3fa6.pth / efficientnet_b0_rwightman-7f5810bc.pth files.
3. Enable read-only repository access through Kaggle Secret exactly Github Token.
   Never paste it into source, Git remote URLs, outputs or chat. The bootstrap
   supplies it only to a temporary Git fetch child environment, suppresses captured
   Git output and clears it afterward. Credential helpers are disabled.
4. Connect VS Code's existing Jupyter Server picker to your Kaggle VSCode Compatible
   server URL. Do not put its token-bearing URL in notebook cells or Git files.
5. Restart the remote kernel before source checkout if project modules are loaded.
   Section1 fetches main into a fresh /kaggle/working/dsa4266_source_* directory and
   records the actual commit. Missing new modules/tests stops clearly; there is
   no fallback to stale codeeeee source. Pin the recorded approved commit for repeats.
6. Follow [the exact section order and settings](RETRAINING_AND_BACKUPS.md#notebook-execution-and-manual-persistence).
   Do NOT Run All. Only executing sections6,7 or11b starts the respective training.

Kaggle cannot read the Windows linked folder. Local changes become available to
the remote GitHub fetch only after an explicitly approved push. No push was made
by this implementation. The source ZIP is for review; bootstrap uses GitHub.

Section3 consumes only existing DEV from SQLite and creates/reuses the immutable
TRAIN pool separately. Save its small manifest ZIP, then reattach train_pool.json
in future sessions to avoid repeat auditing. LSTM has its own portable clip manifest.
Neither manifest replaces the original video input: videos remain required for
lazy decoding. Source video sizes/metadata and exact PTS are checked.

Diagnostics and plots need no activation flags. Tests/dependencies/GPU, pool balance,
labels/rotation, spawned loading and zero-update AMP backward are automatic preflight.
Losses/ranking/precision/recall/F1/F2, numerical logs and checkpoints save each epoch.
Final-only curves/confusion/predictions and family-calibrated DEV F2 analysis save
after the selected checkpoint's final full DEV pass. CPU full training is disabled.

Use Kaggle Files to download rolling ZIPs while training runs; a second cell on the
same busy kernel waits. Optional section4a creates a tiny test ZIP, with no mandatory
confirmation. No Kaggle API credentials or automatic private dataset uploads.
Verify ZIP hashes and retain the original CNN best.pt before a temporal experiment.
See the main guide for restore_local_zip, RESUME_FROM and LSTM source identity.

## Preserved definitions and TEST isolation

RGB/ImageNet normalization, native224 dimensions,10FPS nearest-PTS selection,
half-open visual labels, TRAIN/DEV family separation, CNN architecture/freezing,
BatchNorm policy, AMP fixes and Top-5% video aggregation remain unchanged.
The authorized new pool guard/sampling, LR1.5e-5 and frame-PR selector are recorded
as a combined protocol, not an isolated ablation. No permanent decoded-image cache.

final_evaluation.py is unchanged. Section12 remains TEST_APPROVAL=None. The future
gate rejects absent approval before file access. The current gate supports CNN
evaluation only; temporal TEST evaluation and use of a calibrated F2 TEST threshold
need separate future review/explicit authorization. Do not infer either is approved.
