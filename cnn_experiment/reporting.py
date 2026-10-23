"""Read-only completed-run presentation. No model loading, inference, training or TEST.

Accept a run folder, a completed model ZIP, or a historical history CSV/JSON.
Derived reports are written only to an explicitly NEW destination.
"""
import csv
from html import escape
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

GRAPH_ORDER = ('training_curves.png', 'best_dev_confusion.png', 'best_dev_roc_pr.png',
               'dev_f2_reporting_confusion.png')
MODEL_ORDER = ('resnet50', 'resnet50_lstm', 'efficientnet_b0', 'resnet50_face_crop',
               'efficientnet_b0_face_crop', 'efficientnet_b0_head_lr_2e4')
MODEL_LABELS = {'resnet50': 'Original ResNet50', 'resnet50_lstm': 'ResNet50–LSTM',
    'efficientnet_b0': 'Original EfficientNet-B0', 'resnet50_face_crop': 'Face-masked ResNet50',
    'efficientnet_b0_face_crop': 'Face-masked EfficientNet-B0',
    'efficientnet_b0_head_lr_2e4': 'Full-frame EfficientNet-B0: head LR 2e-4'}


class SavedRun:
    """Read only allowlisted DEV/history artifacts, including flat/nested run ZIPs."""
    def __init__(self, source):
        self.source = Path(source)
        if any(p.lower() in ('test', 'final_test') for p in self.source.parts):
            raise PermissionError('This reporting interface accepts saved TRAIN/DEV artifacts only')
        self.archive = self.source.suffix.lower() == '.zip'
        self.single_history = self.source.is_file() and self.source.suffix.lower() in ('.csv', '.json', '.jsonl')
        if self.archive:
            with zipfile.ZipFile(self.source) as archive:
                names = archive.namelist()
            roots = {str(PurePosixPath(n).parent) for n in names if PurePosixPath(n).name in ('history.jsonl', 'history.json', 'history.csv')
                     and not {'test', 'final_test', '..'} & set(PurePosixPath(n).parts)}
            if len(roots) != 1:
                raise ValueError('Choose a single completed model ZIP, not a mixed-session archive')
            prefix = next(iter(roots))
            self.prefix = '' if prefix == '.' else prefix+'/'

    def read(self, name):
        allowed = set(GRAPH_ORDER) | {'history.jsonl', 'history.json', 'history.csv', 'config.json',
            'training_summary.json', 'input_pipeline.json', 'best_dev_metrics.json', 'best_dev_video_predictions.json',
            'best_dev_predictions.csv', 'dev_f2_calibration.json', 'video_dev_f2_calibration.json'}
        if name not in allowed:
            raise PermissionError('Not an allowlisted DEV reporting artifact')
        if self.single_history:
            expected = 'history'+self.source.suffix.lower()
            return self.source.read_bytes() if name == expected else None
        if self.archive:
            with zipfile.ZipFile(self.source) as archive:
                try:
                    return archive.read(self.prefix+name)
                except KeyError:
                    return None
        path = self.source/name
        return path.read_bytes() if path.is_file() else None

    def json(self, name):
        data = self.read(name)
        return json.loads(data.decode('utf-8-sig')) if data is not None else None


def _scalar(value):
    if value is None or value == '' or isinstance(value, str) and value.lower() in ('nan', 'n/a', 'none', 'null'):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        if value in ('True', 'False'):
            return value == 'True'
        return value


def _unflatten(row):
    result = {}
    for field, value in row.items():
        target = result; parts = field.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _scalar(value)
    return result


def load_history(source):
    run = source if isinstance(source, SavedRun) else SavedRun(source)
    data = run.read('history.jsonl')
    if data is not None:
        rows = [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
    elif run.read('history.json') is not None:
        value = run.json('history.json'); rows = value.get('history', []) if isinstance(value, dict) else value
    else:
        data = run.read('history.csv')
        rows = [_unflatten(row) for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig')))] if data is not None else []
    epochs = [int(row['epoch']) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError('Expected unique completed epochs in increasing order')
    rows = [dict(row, epoch=epoch) for row, epoch in zip(rows, epochs)]
    return rows


def _selected_epoch(history, summary):
    if summary.get('best_epoch') is not None:
        return int(summary['best_epoch'])
    improved = [int(r['epoch']) for r in history if _scalar(r.get('is_best')) is True]
    return improved[-1] if improved else None  # no AUC maximization or re-selection


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def at_half(metrics, fallback_threshold=None):
    """Keep AUC/loss; hide classification values if their saved threshold is unknown/non-0.5."""
    from .evaluation import metrics_with_f2
    metrics = metrics_with_f2(metrics)
    threshold = metrics.get('threshold', fallback_threshold)
    if not _numeric(threshold) or not math.isclose(threshold, .5, abs_tol=1e-12):
        metrics = {key: value for key, value in metrics.items() if key not in ('precision', 'recall', 'f1', 'f2')}
    return metrics


def _best_metrics(run, config, kind):
    """Re-threshold saved best probabilities only; never reuse them for other epochs."""
    from .evaluation import binary_metrics
    saved = run.json('best_dev_metrics.json') or {}
    unit = at_half(saved.get('unit', {}), config.get('threshold'))
    video = at_half(saved.get('video', {}), config.get('threshold'))
    videos = run.json('best_dev_video_predictions.json')
    if videos and any(video.get(key) is None for key in ('f2', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1')):
        computed = binary_metrics([v['target'] for v in videos], [v['probability'] for v in videos], .5)
        video = {**computed, **{k: v for k, v in video.items() if v is not None}}
    raw = run.read('best_dev_predictions.csv') if any(unit.get(key) is None for key in ('f2', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1')) else None
    if raw is not None:
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        # Legacy clip CSVs do not record metric eligibility: never assume all clips qualify.
        if rows and (kind == 'frame' or saved.get('unit', {}).get('n') == len(rows)):
            computed = binary_metrics([int(r['label']) for r in rows], [float(r['probability']) for r in rows], .5)
            unit = {**computed, **{k: v for k, v in unit.items() if v is not None}}
    return unit, video, saved.get('losses', {})


def epoch_table(source, *, kind=None):
    """One cumulative row per completed epoch; missing facts remain None/N/A."""
    run = source if isinstance(source, SavedRun) else SavedRun(source)
    config, summary = run.json('config.json') or {}, run.json('training_summary.json') or {}
    history = load_history(run)
    kind = kind or ('clip' if 'lstm' in config.get('model', '').lower() else 'frame')
    if kind not in ('frame', 'clip'):
        raise ValueError('Choose explicit frame or clip reporting')
    label = kind.title(); best_epoch = _selected_epoch(history, summary)
    saved_unit, saved_video, saved_losses = _best_metrics(run, config, kind)
    rows = []
    for entry in history:
        selected = entry['epoch'] == best_epoch
        unit = at_half(entry.get('dev', {}), config.get('threshold'))
        video = at_half(entry.get('dev_video', {}), config.get('threshold'))
        if selected:
            unit = {**saved_unit, **{k: v for k, v in unit.items() if v is not None}}
            video = {**saved_video, **{k: v for k, v in video.items() if v is not None}}
        lr = entry.get('learning_rates', {})
        rows.append({'Epoch': entry['epoch'], 'TRAIN Loss': entry.get('train', {}).get('loss'),
            'DEV Loss': unit.get('loss') if unit.get('loss') is not None else saved_losses.get('loss') if selected else None,
            'Balanced DEV Loss': unit.get('balanced_loss') if unit.get('balanced_loss') is not None else saved_losses.get('balanced_loss') if selected else None,
            f'{label} PR-AUC': unit.get('pr_auc'), f'{label} ROC-AUC': unit.get('roc_auc'),
            'Video ROC-AUC': video.get('roc_auc'), 'Video PR-AUC': video.get('pr_auc'),
            f'{label} Precision': unit.get('precision'), f'{label} Recall': unit.get('recall'),
            f'{label} F1': unit.get('f1'), f'{label} F2': unit.get('f2'),
            'Video Precision': video.get('precision'), 'Video Recall': video.get('recall'),
            'Video F1': video.get('f1'), 'Video F2': video.get('f2'),
            'Backbone LR': lr.get('final_stage'), 'Classifier/Head LR': lr.get('head', lr.get('temporal_head')),
            'Best Checkpoint': 'Selected best' if selected else ('—' if best_epoch is not None else 'N/A')})
    rows = [{key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in row.items()} for row in rows]
    return rows, best_epoch, kind


def table_html(rows, *, title='Completed epoch DEV metrics (threshold 0.5)'):
    """Four-decimal metrics, scientific LR values, sticky header and horizontal scrolling."""
    if not rows:
        return '<p>'+escape(title)+': N/A (no saved history)</p>'
    fields = list(dict.fromkeys(key for row in rows for key in row))
    result = ['<div style="max-width:100%;overflow-x:auto;max-height:650px;overflow-y:auto;">',
              '<table style="border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums;">',
              '<caption style="text-align:left;">'+escape(title)+'</caption><thead><tr>']
    for field in fields:
        result.append('<th style="position:sticky;top:0;background:#e9eef5;padding:7px;border:1px solid #aaa;">'+escape(field)+'</th>')
    result.append('</tr></thead><tbody>')
    for row in rows:
        selected = row.get('Best Checkpoint') == 'Selected best'
        result.append('<tr'+(' style="background:#fff0b3;font-weight:bold;"' if selected else '')+'>')
        for field in fields:
            value = row.get(field)
            if value is None or isinstance(value, float) and not math.isfinite(value):
                rendered = 'N/A'
            elif field.endswith('LR') and _numeric(value):
                rendered = f'{value:.4e}'  # 1.5e-5 must not misleadingly display as 0.0000
            elif _numeric(value) and field not in ('Epoch', 'Best Epoch'):
                rendered = f'{value:.4f}'
            else:
                rendered = str(value)
            result.append('<td style="padding:6px;border:1px solid #ccc;">'+escape(rendered)+'</td>')
        result.append('</tr>')
    return ''.join(result)+ '</tbody></table></div>'


def display_completed_run(source, *, name=None, kind=None, graph_names=GRAPH_ORDER):
    """Call AFTER train() returns, or independently to replay saved results. Read-only."""
    from IPython.display import HTML, Image, display
    run = SavedRun(source)
    rows, best_epoch, kind = epoch_table(run, kind=kind)
    print(name or str(source), '| completed history; classification metrics at threshold 0.5')
    display(HTML(table_html(rows)))
    for filename in graph_names:  # original images in their original order, no regeneration
        image = run.read(filename)
        if image is not None:
            display(Image(data=image))
        else:
            print(filename+': N/A (saved figure unavailable)')
    selected = next((row for row in rows if row['Epoch'] == best_epoch), {})
    print('Selected best epoch:', best_epoch if best_epoch is not None else 'N/A',
          '| Video F2 @0.5:', f"{selected['Video F2']:.4f}" if _numeric(selected.get('Video F2')) else 'N/A')
    calibration = run.json('dev_f2_calibration.json')
    if calibration:
        op = calibration.get('reporting', {})
        formatted = {k: f'{v:.4f}' if _numeric(v) else 'N/A' for k, v in
            {'threshold': calibration.get('threshold'), **{k: op.get(k) for k in ('precision', 'recall', 'f1', 'f2')}}.items()}
        print('DEV family-calibrated F2 reporting:', formatted)
        print('Checkpoint selection used full DEV: reporting is not an untouched final holdout.')
        print('Threshold changes cannot improve ROC/PR-AUC or BCE. TEST remains isolated.')
    video_calibration = run.json('video_dev_f2_calibration.json')
    if video_calibration:
        reporting = video_calibration.get('reporting', {})
        from .evaluation import metrics_with_f2
        f2 = metrics_with_f2(reporting).get('f2')
        print('DEV-calibrated video F2:', f'{f2:.4f}' if _numeric(f2) else 'N/A',
              '| saved threshold:', video_calibration.get('threshold', 'N/A'))
    return rows


def best_result(source, *, name, kind='frame', expected_preprocessing=None):
    """Best-epoch metrics only, with saved best artifacts as missing-field fallback."""
    face_model = name in ('resnet50_face_crop', 'efficientnet_b0_face_crop')
    if expected_preprocessing not in (None, 'face_mask', 'crop_30'):
        raise ValueError('Expected preprocessing must be face_mask or crop_30')
    label = MODEL_LABELS.get(name, name)
    if face_model and expected_preprocessing == 'crop_30': label = label.replace('Face-masked', 'Face-cropped')
    base = {'Model': label, 'Status': 'Pending', 'Level': kind.title(), 'Best Epoch': None,
            'Input': expected_preprocessing if face_model else 'full_frame' if kind == 'frame' else 'frozen ResNet50 features'}
    names = ['Frame/Clip PR-AUC', 'Frame/Clip ROC-AUC', 'Video ROC-AUC', 'Video PR-AUC',
             'Frame/Clip Precision', 'Frame/Clip Recall', 'Frame/Clip F1', 'Frame/Clip F2',
             'Video Precision', 'Video Recall', 'Video F1', 'Video F2', 'DEV Loss', 'Balanced DEV Loss', 'TRAIN Loss']
    base.update({key: None for key in names})
    if source is None:
        return base
    run = SavedRun(source)
    config, summary = run.json('config.json') or {}, run.json('training_summary.json') or {}
    expected = ('resnet50_lstm' if name == 'resnet50_lstm' else
                'resnet50' if name.startswith('resnet50') else
                'efficientnet_b0' if name.startswith('efficientnet_b0') else None)
    recorded = config.get('model', summary.get('model'))
    if expected and recorded and recorded != expected:
        raise ValueError(f'Saved model {recorded} cannot be reported as {name}; choose the matching run.')
    if face_model or name == 'efficientnet_b0_head_lr_2e4':
        expected_lr = 2e-4 if name == 'efficientnet_b0_head_lr_2e4' else 1e-4
        if config.get('head_lr') != expected_lr:
            base['Status'] = 'Pending (saved head LR missing or differs from this experiment)'
            return base
    pipeline = run.json('input_pipeline.json')
    saved_mode = {'face_mask': 'face_mask', 'face_crop': 'crop_30'}.get((pipeline or {}).get('kind'))
    if face_model:
        if expected_preprocessing is not None and saved_mode != expected_preprocessing:
            base['Status'] = f'Pending (saved {saved_mode or "unknown preprocessing"} excluded)'
            return base  # Never present crop results as the newly scheduled mask model.
        base['Input'] = saved_mode or 'N/A (preprocessing unknown)'
        if saved_mode == 'crop_30': base['Model'] = label.replace('Face-masked', 'Face-cropped')
    elif pipeline is not None and kind == 'frame':
        raise ValueError('Original/full-frame comparison row cannot use a face-preprocessed run')
    rows, selected, _ = epoch_table(run, kind=kind)
    row = next((row for row in rows if row['Epoch'] == selected), {})
    unit, video, losses = _best_metrics(run, config, kind)
    base.update(Status='Completed' if summary or rows or unit or video else 'N/A (saved results unavailable)', **{'Best Epoch': selected})
    for column in names:
        history_column = column.replace('Frame/Clip', kind.title())
        if column.startswith('Frame/Clip'):
            metric = {'PR-AUC': 'pr_auc', 'ROC-AUC': 'roc_auc'}.get(column.split(' ', 1)[1], column.split(' ', 1)[1].lower())
            fallback = unit.get(metric)
        elif column.startswith('Video'):
            metric = {'ROC-AUC': 'roc_auc', 'PR-AUC': 'pr_auc'}.get(column[6:], column[6:].lower())
            fallback = video.get(metric)
        else:
            fallback = losses.get({'DEV Loss': 'loss', 'Balanced DEV Loss': 'balanced_loss'}.get(column, ''))
        value = row.get(history_column)
        base[column] = value if value is not None else fallback
    return base


def comparison_rows(runs, *, face_mode='face_mask'):
    if face_mode not in ('face_mask', 'crop_30'): raise ValueError('Choose one of the two face modes')
    return [best_result(runs.get(name), name=name, kind='clip' if name == 'resnet50_lstm' else 'frame',
                        expected_preprocessing=face_mode if name in ('resnet50_face_crop', 'efficientnet_b0_face_crop') else None)
            for name in MODEL_ORDER]


def display_comparison(runs, *, face_mode='face_mask'):
    from IPython.display import HTML, display
    rows = comparison_rows(runs, face_mode=face_mode)
    display(HTML(table_html(rows, title='Selected best DEV results; classification threshold 0.5')))
    return rows


def export_epoch_report(source, destination, *, kind=None):
    """Export enriched historical CSV/JSON/HTML into a NEW folder; never overwrite a run."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows, selected, native_kind = epoch_table(source, kind=kind)
    with (destination/'epoch_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['Epoch']); writer.writeheader(); writer.writerows(rows)
    (destination/'epoch_metrics.json').write_text(json.dumps({'rows': rows, 'selected_epoch': selected, 'level': native_kind,
        'threshold': .5, 'missing_values': 'null / N/A; never fabricated'}, indent=2, allow_nan=False), encoding='utf-8')
    (destination/'epoch_metrics.html').write_text(table_html(rows), encoding='utf-8')
    return destination
