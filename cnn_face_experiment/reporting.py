"""Read-only DEV comparison and matched-cohort reporting; no inference or TEST."""
import csv
from contextlib import closing
from fractions import Fraction
import json
from pathlib import Path
import sqlite3

from cnn_experiment.calibration import calibrated_f2
from cnn_experiment.checkpoints import digest
from cnn_experiment.evaluation import binary_metrics, evaluate_predictions
from cnn_experiment.pool import DevSchedule
from .crops import frame_key
from .experiments import EXPERIMENTS, experiment_configurations


def saved_predictions(run, records):
    run = Path(run)
    by_key = {(r['source_file'], Fraction(r['timestamps_exact'][0])): i for i, r in enumerate(records)}
    result = []
    with (run/'best_dev_predictions.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = row['source_file'], Fraction(row['timestamp_exact'])
            if key not in by_key:
                raise ValueError('Prediction outside the fixed DEV index')
            i = by_key[key]
            if records[i]['label'] != int(row['label']) or records[i]['id'] != int(row['candidate_id']):
                raise ValueError('DEV identity/label mismatch in comparison')
            result.append({'record_index': i, 'target': records[i]['label'], 'probability': float(row['probability'])})
    if sorted(p['record_index'] for p in result) != list(range(len(records))):
        raise ValueError('Matched comparison requires complete, unique fixed DEV predictions')
    return sorted(result, key=lambda p: p['record_index'])


def verify_baseline(run, architecture, index_sha256, pool_sha256):
    """Check the ACTUAL recorded config, not a historical metrics-only reference."""
    config = json.loads((Path(run)/'config.json').read_text(encoding='utf-8'))
    expected = next(c for c in experiment_configurations().values() if c.model == architecture)
    runtime = {'workers', 'prefetch_factor', 'cache_mib', 'loader_timeout_seconds', 'train_progress_every',
               'dev_progress_every', 'heartbeat_seconds'}
    mismatch = {key: (config.get(key), value) for key, value in expected.as_dict().items()
                if key not in runtime and config.get(key) != value}
    if mismatch or config.get('index_sha256') != index_sha256 or config.get('pool_sha256') != pool_sha256:
        raise ValueError(f'Original {architecture} is not the matched fixed20k/LR15e6 baseline: {mismatch}')
    if (Path(run)/'input_pipeline.json').exists():
        raise ValueError('Expected the original FULL-FRAME baseline')
    return config


def summarize_run(name, run, records):
    run = Path(run)
    summary = json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    if config['topk_fraction'] != .05:
        raise ValueError('Expected existing Top-5% pooling')
    predictions = saved_predictions(run, records)
    report = evaluate_predictions(records, predictions,
        {'aggregation': 'topk', 'topk_fraction': .05, 'prediction_threshold': .5, 'video_threshold': .5})
    history = [json.loads(line) for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    best = history[summary['best_epoch']-1]
    frame_calibration = calibrated_f2(records, predictions, config['seed'])
    videos = report['videos']
    video_records = [{'id': i, 'kind': 'frame', 'split': 'dev', 'source_file': v['source_file'],
                      'source_family': v['source_family'], 'label': v['target']} for i, v in enumerate(videos)]
    video_scores = [{'record_index': i, 'probability': v['probability']} for i, v in enumerate(videos)]
    video_calibration = calibrated_f2(video_records, video_scores, config['seed'])
    input_path = run/'input_pipeline.json'
    pipeline = json.loads(input_path.read_text(encoding='utf-8')) if input_path.is_file() else None
    row = {'experiment': name, 'status': 'completed', 'architecture': config['model'],
        'native_level': 'frame',
        'input': pipeline['kind'] if pipeline else 'full_frame',
        'head_lr': config['head_lr'], 'backbone_lr': config['backbone_lr'], 'best_epoch': summary['best_epoch'],
        'best_train_loss': best['train']['loss'], 'best_dev_loss': best['dev']['loss'],
        'training_and_epoch_dev_seconds': sum(r['train_seconds']+r['dev_seconds'] for r in history),
        'frame_dev_f2_threshold': frame_calibration['threshold'], 'video_dev_f2_threshold': video_calibration['threshold']}
    for prefix, metrics in (('frame', report['unit']), ('video', report['video'])):
        for metric in ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'f2'):
            row[f'{prefix}_{metric}_at_0_5' if metric not in ('pr_auc', 'roc_auc') else f'{prefix}_{metric}'] = metrics[metric]
    for prefix, calibration in (('frame', frame_calibration), ('video', video_calibration)):
        for metric in ('precision', 'recall', 'f1', 'f2'):
            row[f'{prefix}_{metric}_calibrated_reporting'] = calibration['reporting'][metric]
    return row, {'full_dev_at_0_5': report, 'frame_calibration': frame_calibration,
                 'video_calibration': video_calibration}, predictions


def failure_cohorts(face_run, records, baseline_predictions, face_predictions):
    path = (Path(face_run)/'face_boxes.sqlite').resolve()
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
        statuses = {key: json.loads(metadata)['status'] for key, metadata in db.execute("SELECT frame_key,metadata FROM boxes WHERE split='dev'")}
        policy = json.loads(db.execute("SELECT value FROM info WHERE key='policy'").fetchone()[0])
    face_kind = 'face_mask' if policy.get('kind') == 'face_mask' else 'face_crop'
    keys = [frame_key(r) for r in records]
    if any(k not in statuses for k in keys):
        raise ValueError('Detection metadata does not cover every fixed DEV frame')
    report = {'full_dev_frames': len(records), 'dev_frames_excluded': 0,
              'scope': 'Descriptive matched frame cohorts; no replacement of full DEV/video protocol', 'cohorts': {}}
    for cohort, predicate in [('detected', lambda status: status in ('detected', 'masked')),
                              ('fallback', lambda status: status not in ('detected', 'masked'))]:
        ids = [i for i, key in enumerate(keys) if predicate(statuses[key])]
        entry = {'n': len(ids), 'real': sum(records[i]['label'] == 0 for i in ids), 'fake': sum(records[i]['label'] == 1 for i in ids)}
        for name, predictions in [('original', baseline_predictions), (face_kind, face_predictions)]:
            by_index = {p['record_index']: p['probability'] for p in predictions}
            entry[name] = binary_metrics([records[i]['label'] for i in ids], [by_index[i] for i in ids], .5) if ids else None
        report['cohorts'][cohort] = entry
    return report


def compare_experiments(baselines, new_runs, index_path, pool_path, output, *, lstm_run=None, preprocessing_mode='face_mask'):
    """Six labelled experiments, including saved ResNet50-LSTM; no winner re-selection."""
    from cnn_experiment.data import write_json
    if preprocessing_mode not in ('face_mask', 'crop_30'): raise ValueError('Unknown face preprocessing mode')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    schedule = DevSchedule(index_path)
    try:
        records = schedule.records('dev', 'frame')
    finally:
        schedule.close()
    order = [('original_resnet50', baselines.get('resnet50')),
             ('resnet50_face_crop', new_runs.get('resnet50_face_crop')),
             ('original_efficientnet_b0', baselines.get('efficientnet_b0')),
             ('efficientnet_b0_face_crop', new_runs.get('efficientnet_b0_face_crop')),
             ('efficientnet_b0_head_lr_2e4', new_runs.get('efficientnet_b0_head_lr_2e4'))]
    rows, predictions = [], {}
    for name, path in order:
        if path is None or not (Path(path)/'training_summary.json').is_file():
            rows.append({'experiment': name, 'status': 'Pending', 'native_level': 'frame'}); continue
        if name.startswith('original_'):
            verify_baseline(path, name.removeprefix('original_'), digest(index_path), digest(pool_path))
        else:
            saved_config = json.loads((Path(path)/'config.json').read_text(encoding='utf-8'))
            if saved_config.get('model') != EXPERIMENTS[name][0]: raise ValueError('Saved experiment architecture mismatch')
            if saved_config.get('head_lr') != EXPERIMENTS[name][2]: raise ValueError('Saved experiment head LR mismatch')
            input_path = Path(path)/'input_pipeline.json'
            pipeline = json.loads(input_path.read_text(encoding='utf-8')) if input_path.is_file() else None
            expected_kind = ('face_mask' if preprocessing_mode == 'face_mask' else 'face_crop') if EXPERIMENTS[name][1] else None
            if (pipeline or {}).get('kind') != expected_kind:
                raise ValueError('Saved preprocessing does not match this six-model plan; keep crop references separate from mask results')
        row, report, scores = summarize_run(name, path, records)
        rows.append(row); predictions[name] = scores
        write_json(output/(name+'_metrics.json'), report)
        # Confusion matrices for both levels at 0.5 and family-separated F2 thresholds.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        for ax, title, metrics in zip(axes.flat,
            ('Frame @0.5', 'Video @0.5', 'Frame calibrated reporting families', 'Video calibrated reporting families'),
            (report['full_dev_at_0_5']['unit'], report['full_dev_at_0_5']['video'], report['frame_calibration']['reporting'], report['video_calibration']['reporting'])):
            cm = metrics['confusion_matrix']; ax.imshow(cm, cmap='Blues', vmin=0)
            for i in range(2):
                for j in range(2): ax.text(j, i, str(cm[i][j]), ha='center', va='center')
            ax.set(title=title, xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], xlabel='Predicted', ylabel='GT')
        fig.suptitle(name); fig.tight_layout(); fig.savefig(output/(name+'_confusion.png'), dpi=130); plt.close(fig)
    for architecture in ('resnet50', 'efficientnet_b0'):
        face = architecture+'_face_crop'; original = 'original_'+architecture
        if face in predictions and original in predictions:
            write_json(output/(face+'_matched_cohorts.json'), failure_cohorts(new_runs[face], records, predictions[original], predictions[face]))
    from cnn_experiment.reporting import best_result
    temporal = best_result(lstm_run, name='resnet50_lstm', kind='clip')
    temporal_row = {'experiment': 'original_resnet50_lstm',
        'status': 'completed' if temporal['Status'] == 'Completed' else temporal['Status'],
        'architecture': 'resnet50_lstm', 'native_level': 'clip', 'best_epoch': temporal['Best Epoch'],
        'best_train_loss': temporal['TRAIN Loss'], 'best_dev_loss': temporal['DEV Loss'],
        'balanced_dev_loss': temporal['Balanced DEV Loss'],
        'clip_pr_auc': temporal['Frame/Clip PR-AUC'], 'clip_roc_auc': temporal['Frame/Clip ROC-AUC'],
        'video_pr_auc': temporal['Video PR-AUC'], 'video_roc_auc': temporal['Video ROC-AUC']}
    for prefix, title in (('clip', 'Frame/Clip'), ('video', 'Video')):
        for metric in ('precision', 'recall', 'f1', 'f2'):
            temporal_row[f'{prefix}_{metric}_at_0_5'] = temporal[f'{title} {metric.title() if metric not in ("f1", "f2") else metric.upper()}']
    rows.insert(1, temporal_row)
    # CSV ordering only: Video F2 is immediately between Video F1 and DEV Loss.
    leading = ['experiment', 'status', 'native_level', 'best_epoch', 'frame_pr_auc', 'clip_pr_auc',
        'frame_roc_auc', 'clip_roc_auc', 'video_roc_auc', 'video_pr_auc',
        'frame_precision_at_0_5', 'clip_precision_at_0_5', 'frame_recall_at_0_5', 'clip_recall_at_0_5',
        'frame_f1_at_0_5', 'clip_f1_at_0_5', 'frame_f2_at_0_5', 'clip_f2_at_0_5',
        'video_precision_at_0_5', 'video_recall_at_0_5', 'video_f1_at_0_5', 'video_f2_at_0_5', 'best_dev_loss']
    rows = [{key: row.get(key) for key in leading} | {key: value for key, value in row.items() if key not in leading} for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) if row.get(key) is not None else 'N/A' for key in fields} for row in rows)
    write_json(output/'comparison.json', {'rows': rows, 'test_used': False,
        'notes': ['Full fixed DEV is primary; detected-only cohorts are descriptive and can be biased.',
                  'Threshold calibration/reporting is family-separated within DEV; checkpoint selection used full DEV.',
                  'Durations include online TRAIN and per-epoch DEV, excluding final plots/inference/ZIP overhead.',
                  'Original references are the actual fixed20k completed full-frame runs, not older historical metrics.']})
    return rows
