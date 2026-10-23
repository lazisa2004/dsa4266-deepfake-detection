# Reusable TRAIN/DEV candidate index

`candidates.sqlite` is the exact combined index recorded by the completed CNN
experiments at code commit `5609c907380b57ff92d57ed96f6bf0f04947875e`.
It is shared in this private repository with the project owner's explicit approval.

| Split | Videos | Frames | Real frames | Fake frames | Clips |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 4,000 | 315,058 | 302,240 | 12,818 | 26,817 |
| DEV | 500 | 41,746 | 40,115 | 1,631 | 3,551 |

- Size:32,714,752 bytes (31.20 MiB).
- SHA256:`ac3cd8c440f0bb1e53cf955bcc6a2596e9ee1bc40e508f254c82e10bde13cfcb`.
- Protocol:`cnn_v2_pts10_clip16_stride12_one_fake`.
- Tables:`info`, `videos`, `frames`, `clips`.
- Contains relative TRAIN/DEV video references, exact source PTS, 10 FPS selections,
  merged visual intervals, frame/clip labels and source-size checks.
- No TEST records, raw video/image/audio data, trained weights, credentials,
  personal absolute paths or notebook outputs.

The original generator is `cnn_experiment.data.build_index`; `choose_indices`
selects nearest source frames to0.1-second targets, with earlier-source ties.
Fake frame labels use visual intervals `[start,end)`; audio-only is visually real.
DEV boundary-near frames are retained. Never silently rebuild or alter this index.

This full index is separate from the new immutable20,000-frame TRAIN pool
(`train_pool.json`). The pool manifest is not included here. Original videos are
still required to decode images; metadata alone contains no pixels.

The current notebook still uses its existing attached `meta-dataa` input. To use
the identical repository copy in a future session, after source checkout and before
the input-discovery cell, explicitly set:

```python
INDEX_OVERRIDE = CODE_ROOT / 'metadata' / 'candidates.sqlite'
```

No sampling, label, training or TEST behavior changes from this path override.
Cloning the repository now supplies both the preprocessing source and this exact
index. Authorized access to the original dataset is still needed for decoding.
The main VS Code notebook remains local and Git-ignored.
