"""Future held-out evaluation, inaccessible without explicit final approval.

Importing this module performs no filesystem reads, CUDA work or data access.
The CNN training notebook leaves the approval value unset.
"""
FINAL_TEST_APPROVAL = 'I explicitly approve final held-out TEST evaluation'


def evaluate_final_test(session, data_root, weights_root, *, approval=None):
    # This guard MUST precede imports, path discovery, metadata reads or CUDA.
    if approval != FINAL_TEST_APPROVAL:
        raise PermissionError('TEST remains isolated. Obtain explicit final-evaluation approval first.')
    import csv
    import gc
    import json
    from pathlib import Path
    import torch
    from torch import nn
    from cnn_experiment.config import Config
    from cnn_experiment.data import make_loader
    from cnn_experiment.data import PROTOCOL, choose_indices, merge_periods, packet_timeline, write_json
    from cnn_experiment.models import StagedCNN
    from cnn_experiment.training import RunProgress, should_log_batch
    from cnn_experiment.checkpoints import digest
    from cnn_experiment.training import require_cuda, seed_everything
    from cnn_experiment.evaluation import class_loss_metrics, evaluate_predictions, binary_curves

    require_cuda()
    session, data_root = Path(session), Path(data_root)
    comparison_path = session/'comparison/comparison.json'
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    selected = comparison['selected_cnn']
    if selected not in ('resnet50', 'efficientnet_b0'):
        raise ValueError('Freeze the CNN winner using DEV before final evaluation.')
    run = session/selected
    checkpoint_path = run/'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['config']['model'] != selected:
        raise ValueError('The selected checkpoint protocol/model differs from the DEV selection.')
    config = Config(**checkpoint['config']).validate()
    # Require completion and the exact same best epoch/DEV winner. No threshold tuning.
    summary = json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    selected_row = next(r for r in comparison['models'] if r['model'] == selected)
    if checkpoint['epoch'] + 1 != summary['best_epoch'] or summary['best_epoch'] != selected_row['best_epoch']:
        raise ValueError('DEV comparison and selected checkpoint do not match.')
    if digest(checkpoint_path) != selected_row['best_checkpoint_sha256']:
        raise ValueError('The best checkpoint changed after DEV selection was frozen.')
    output = session/'final_test'
    output.mkdir(parents=True, exist_ok=False)
    with RunProgress(output, selected, config.heartbeat_seconds) as progress:
        write_json(output/'frozen_policy.json', {'approval': approval, 'model': selected,
            'checkpoint_sha256': digest(checkpoint_path), 'comparison_sha256': digest(comparison_path),
            'threshold': config.threshold, 'topk_fraction': config.topk_fraction, 'fps': 10,
            'selection_source': 'DEV only; selected CNN only; no TEST model/threshold selection'})
        progress.stage('approved_test_metadata_and_timeline', announce=True)
        rows = [r for r in json.loads((data_root/'metadata.json').read_text(encoding='utf-8')) if r['split'] == 'test']
        by_file = {r['file']: r for r in rows}
        if not rows or len(by_file) != len(rows):
            raise ValueError('Nonempty, unique final-evaluation video metadata required.')
        records = []
        for row in sorted(rows, key=lambda r: r['file']):
            source = row['file']
            original = by_file.get(row['original'] or source)
            if (not source.startswith('test/') or original is None or
                    original['modify_video'] or original['modify_audio']):
                raise ValueError('Invalid held-out source family.')
            path = (data_root/source).resolve()
            if data_root.resolve() not in path.parents:
                raise ValueError('Held-out source path escaped the dataset root.')
            times = packet_timeline(path, row['video_frames'])
            periods = merge_periods(row['fake_periods']) if row['modify_video'] else []
            for idx in choose_indices(times, fps=10):
                records.append({'id': len(records), 'split': 'test', 'kind': 'frame',
                    'source_file': source, 'source_family': row['original'] or source,
                    'video_label': int(row['modify_video']), 'indices': [idx],
                    'timestamps_exact': [str(times[idx])], 'metric_eligible': True,
                    'label': int(any(a <= times[idx] < b for a, b in periods))})
        seed_everything(config.seed)
        model = StagedCNN(config, weights_root, pretrained=False).cuda()
        model.load_state_dict(checkpoint['model'], strict=True); model.eval()
        del checkpoint
        loader = make_loader(records, data_root, config, training=False)
        predictions, labels, scores = [], [], []
        sums, counts = {0: 0., 1: 0.}, {0: 0, 1: 0}
        with torch.no_grad():
            for batch, (images, targets, indices) in enumerate(loader, 1):
                progress.stage('approved_test_inference', batch=batch, total_batches=len(loader))
                images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16, enabled=config.amp):
                    logits = model(images).squeeze(1)
                    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError('Non-finite final-evaluation logits/loss; no model update or retuning is allowed.')
                for idx, target, score, loss in zip(indices.tolist(), targets.int().cpu().tolist(),
                                                   logits.float().sigmoid().cpu().tolist(), losses.float().cpu().tolist()):
                    predictions.append({'record_index': idx, 'target': target, 'probability': score})
                    labels.append(target); scores.append(score); sums[target] += loss; counts[target] += 1
                if should_log_batch(batch, len(loader), config.dev_progress_every):
                    progress.note(f'Final TEST evaluation: Batch {batch}/{len(loader)}')
        report = evaluate_predictions(records, predictions,
            {'aggregation': 'topk', 'topk_fraction': config.topk_fraction,
             'prediction_threshold': config.threshold, 'video_threshold': config.threshold}, 'frame')
        report['losses'] = class_loss_metrics(sums, counts) | {'loss': sum(sums.values())/sum(counts.values())}
        write_json(output/'test_metrics.json', {k: v for k, v in report.items() if k != 'videos'})
        write_json(output/'test_video_predictions.json', report['videos'])
        with (output/'test_frame_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['source_file', 'timestamp_exact', 'label', 'probability'])
            writer.writeheader()
            for prediction in predictions:
                record = records[prediction['record_index']]
                writer.writerow({'source_file': record['source_file'], 'timestamp_exact': record['timestamps_exact'][0],
                                 'label': record['label'], 'probability': prediction['probability']})
        write_json(output/'test_curves.json', {'frame': binary_curves(labels, scores),
            'video': binary_curves([v['target'] for v in report['videos']], [v['probability'] for v in report['videos']])})
        progress.note('Final evaluation complete. Metrics saved under ' + str(output))
        del model, loader
        gc.collect(); torch.cuda.empty_cache()
    return output
