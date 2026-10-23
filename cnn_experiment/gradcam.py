"""Post-hoc CNN Grad-CAM. No training, index creation, or implicit TEST access.

Run ``python -m cnn_experiment.gradcam --help`` for the bounded DEV demo.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
import argparse
import csv
import json
import math
import random
import warnings

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import digest
from .config import Config
from .data import FrameReader, LazyDataset, PROTOCOL, packet_timeline
from .evaluation import video_predictions
from .final_evaluation import FINAL_TEST_APPROVAL
from .models import StagedCNN
from .pool import DevSchedule


def guard_split(split, approval=None):
    """Call before any checkpoint, prediction, metadata or video access."""
    if split not in ('dev', 'test'):
        raise ValueError('Explainability supports dev or test only')
    if split == 'test' and approval != FINAL_TEST_APPROVAL:
        raise PermissionError('TEST Grad-CAM is locked; explicit final-evaluation approval required.')


def target_layer(model, name):
    """Last spatial feature output (7x7 for native 224x224 inputs)."""
    network = model.network if isinstance(model, StagedCNN) else model
    if name == 'resnet50':
        return network.layer4[-1], 'network.layer4[-1]'
    if name == 'efficientnet_b0':
        return network.features[-1], 'network.features[-1]'
    raise ValueError('Grad-CAM supports ResNet50 and EfficientNet-B0 CNNs only')


class GradCAM:
    """Fake-logit Grad-CAM; autograd.grad never accumulates parameter gradients.

    Fresh input gradients make frozen backbones work. Hooks exist only during a
    call and are removed even on failure. FP32 avoids half-precision overflow.
    """
    def __init__(self, model, layer):
        self.model, self.layer = model, layer

    def __call__(self, images):
        states = [(module, module.training) for module in self.model.modules()]
        captured = []
        handle = self.layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            self.model.eval()
            with torch.inference_mode(False), torch.enable_grad(), torch.autocast(images.device.type, enabled=False):
                inputs = images.detach().clone().float().requires_grad_(True)
                logits = self.model(inputs).reshape(-1)
                if len(captured) != 1 or captured[0].ndim != 4 or logits.numel() != len(inputs):
                    raise ValueError('Expected one spatial activation and one binary Fake logit per image')
                activation = captured[0]
                gradient, = torch.autograd.grad(logits.sum(), activation)
                if not all(bool(torch.isfinite(t).all()) for t in (logits, activation, gradient)):
                    raise FloatingPointError('Non-finite Grad-CAM logit, activation or gradient')
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                raw = (weights * activation).sum(dim=1, keepdim=True).relu()
                maps = F.interpolate(raw, size=inputs.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
                maxima = maps.flatten(1).amax(dim=1).view(-1, 1, 1)
                maps = torch.where(maxima > 0, maps / maxima.clamp_min(torch.finfo(maps.dtype).tiny), torch.zeros_like(maps))
                return {'heatmaps': maps.detach().cpu().numpy(),
                        'probabilities': logits.detach().sigmoid().cpu().numpy(),
                        'logits': logits.detach().cpu().numpy(),
                        'zero_maps': (maxima.reshape(-1) == 0).cpu().tolist(),
                        'activation_shape': list(activation.shape)}
        finally:
            handle.remove()
            captured.clear()
            # Direct flags preserve mixed BN/dropout states without invoking custom train().
            for module, training in states:
                module.training = training


def load_model(checkpoint, device='cpu', *, split='dev', approval=None):
    guard_split(split, approval)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('protocol') != PROTOCOL:
        raise ValueError('Expected an existing CNN frame-protocol checkpoint')
    config = Config(**state['config']).validate()
    # Keep the caller's RNG unchanged: constructing an uninitialized model uses RNG.
    with torch.random.fork_rng(devices=[]):
        model = StagedCNN(config, '.', pretrained=False)
    model.load_state_dict(state['model'], strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()):
        raise FloatingPointError('Non-finite checkpoint weights')
    model.to(device=device, dtype=torch.float32).eval()
    model.explanation_preprocessing = state.get('input_pipeline')
    return model, config, state.get('index_sha256')


def _valid_source(source, split):
    parts = PurePosixPath(source)
    if '\\' in source or parts.is_absolute() or '..' in parts.parts or len(parts.parts) < 2 or parts.parts[0] != split:
        raise ValueError('Frame reference must be a relative path within the requested split')


@dataclass
class ExplanationData:
    split: str
    records: list
    predictions: list
    frame_threshold: float
    video_threshold: float
    checkpoint_sha256: str
    predictions_sha256: str
    approval: str = None


def load_explanation_data(checkpoint, predictions_csv, *, split='dev', index_path=None,
                          frozen_policy_path=None, approval=None):
    """Bind saved predictions to the existing schedule; never sample new frames.

    TEST requires the existing final_test/frozen_policy.json plus completed
    test_metrics.json and test_video_predictions.json. No TEST data is read before
    the explicit guard. The final TEST CSV itself is its persisted frame schedule.
    """
    guard_split(split, approval)
    checkpoint_hash = digest(checkpoint)
    predictions_csv = Path(predictions_csv)
    if split == 'test':
        if frozen_policy_path is None:
            raise ValueError('Completed final TEST frozen_policy.json required')
        policy_path = Path(frozen_policy_path)
        if policy_path.name != 'frozen_policy.json' or predictions_csv.resolve() != (policy_path.parent/'test_frame_predictions.csv').resolve():
            raise ValueError('Use predictions and frozen policy from the same final TEST evaluation')
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
        if (policy.get('approval') != FINAL_TEST_APPROVAL or policy.get('checkpoint_sha256') != checkpoint_hash
                or policy.get('topk_fraction') != .05 or policy.get('fps') != 10):
            raise ValueError('Final TEST checkpoint, approval or frozen Top-5%/10 FPS policy mismatch')
        metrics = json.loads((policy_path.parent/'test_metrics.json').read_text(encoding='utf-8'))
        videos = json.loads((policy_path.parent/'test_video_predictions.json').read_text(encoding='utf-8'))
        video_map = {v['source_file']: v for v in videos}
        threshold = float(policy['threshold'])
        records = []
    else:
        if index_path is None:
            raise ValueError('DEV requires the existing candidates.sqlite (no rebuild)')
        schedule = DevSchedule(index_path)
        try:
            records = schedule.records('dev', 'frame')
        finally:
            schedule.close()
        # Only config and model state are read; no optimizer restoration or mutation.
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('protocol') != PROTOCOL or state.get('index_sha256') != digest(index_path):
            raise ValueError('Checkpoint does not match the exact original DEV index')
        threshold = float(state['config']['threshold'])
        del state
        if frozen_policy_path is not None:
            policy = json.loads(Path(frozen_policy_path).read_text(encoding='utf-8'))
            if policy.get('checkpoint_sha256') != checkpoint_hash or policy.get('topk_fraction') != .05 or policy.get('selection_source') != 'dev':
                raise ValueError('Optional DEV policy must be frozen on DEV for this checkpoint')
            threshold = float(policy['threshold'])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid frozen classification threshold')
    by_key = {(r['source_file'], Fraction(r['timestamps_exact'][0])): i for i, r in enumerate(records)}
    predictions, seen = [], set()
    with predictions_csv.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            source, stamp = row['source_file'], Fraction(row['timestamp_exact'])
            _valid_source(source, split)
            key = (source, stamp)
            probability, label = float(row['probability']), int(row['label'])
            if key in seen or not math.isfinite(probability) or not 0 <= probability <= 1 or label not in (0, 1):
                raise ValueError('Duplicate or invalid saved frame prediction')
            seen.add(key)
            if split == 'dev':
                if key not in by_key:
                    raise ValueError('Prediction timestamp is outside the fixed DEV schedule')
                i = by_key[key]
                record = records[i]
                if label != record['label'] or int(row['candidate_id']) != record['id']:
                    raise ValueError('Prediction labels/candidate IDs differ from the DEV index')
            else:
                if source not in video_map:
                    raise ValueError('Missing final video prediction for TEST frame')
                video = video_map[source]
                i = len(records)
                records.append({'id': i, 'split': split, 'kind': 'frame', 'source_file': source,
                    'source_family': video['source_family'], 'video_label': video['target'],
                    'label': label, 'timestamps_exact': [str(stamp)], 'metric_eligible': True})
            predictions.append({'record_index': i, 'target': label, 'probability': probability})
    if not records or len(predictions) != len(records):
        raise ValueError('Exactly one prediction per fixed scheduled frame is required')
    predictions.sort(key=lambda p: p['record_index'])
    if split == 'test':
        if metrics['scheduled_predictions'] != len(records) or set(video_map) != {r['source_file'] for r in records}:
            raise ValueError('Incomplete final TEST evaluation artifacts')
        # Verify that saved full video aggregation agrees, rather than inferring video labels.
        aggregated = video_predictions(records, predictions, {'aggregation': 'topk', 'topk_fraction': .05})
        for video in aggregated:
            saved = video_map[video['source_file']]
            if saved['samples'] != video['samples'] or not math.isclose(saved['probability'], video['probability'], abs_tol=1e-7):
                raise ValueError('Final TEST frame/video predictions disagree')
    return ExplanationData(split, records, predictions, threshold, threshold, checkpoint_hash,
                           digest(predictions_csv), approval)


def select_examples(data, category='tp', count=2, seed=4266):
    """Seeded confusion-cell sampling; high-confidence groups use predicted confidence.

    Correct/incorrect confidence = p for predicted Fake, otherwise 1-p. Thus the
    high-confidence selections reflect the frozen threshold, including non-0.5 rules.
    """
    guard_split(data.split, data.approval)
    if category not in ('tp', 'tn', 'fp', 'fn', 'high_confidence_correct', 'high_confidence_incorrect') or count < 1:
        raise ValueError('Unknown selection category or nonpositive count')
    candidates = []
    for p in data.predictions:
        r = data.records[p['record_index']]
        target = r.get('label')
        if target not in (0, 1):
            continue
        predicted = int(p['probability'] >= data.frame_threshold)
        cell = {(1, 1): 'tp', (0, 0): 'tn', (0, 1): 'fp', (1, 0): 'fn'}[target, predicted]
        if category == cell or category == ('high_confidence_correct' if target == predicted else 'high_confidence_incorrect'):
            candidates.append(p['record_index'])
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if category.startswith('high_confidence'):
        scores = {p['record_index']: p['probability'] for p in data.predictions}
        candidates.sort(key=lambda i: -(scores[i] if scores[i] >= data.frame_threshold else 1-scores[i]))
    return candidates[:count]


def top5_contributors(data, source_file):
    """Use ALL saved frames: mean of ceil(0.05*N), with deterministic tie ordering."""
    guard_split(data.split, data.approval)
    _valid_source(source_file, data.split)
    items = [p for p in data.predictions if data.records[p['record_index']]['source_file'] == source_file]
    if not items:
        raise ValueError('Video absent from saved evaluation')
    items.sort(key=lambda p: (-p['probability'], Fraction(data.records[p['record_index']]['timestamps_exact'][0]), p['record_index']))
    k = max(1, math.ceil(.05 * len(items)))
    contributors = items[:k]
    policy = {'aggregation': 'topk', 'topk_fraction': .05}
    report = next(v for v in video_predictions(data.records, data.predictions, policy) if v['source_file'] == source_file)
    mean = sum(p['probability'] for p in contributors)/k
    if not math.isclose(mean, report['probability'], abs_tol=1e-12):
        raise AssertionError('Top-5% contributor mean differs from existing evaluator')
    return {'source_file': source_file, 'total_frames': len(items), 'k': k, 'probability': mean,
            'video_threshold': data.video_threshold, 'predicted_fake': mean >= data.video_threshold,
            'contributors': [p | {'timestamp_exact': data.records[p['record_index']]['timestamps_exact'][0],
                                  'weight': 1/k} for p in contributors]}


class _ApprovedTestReader(FrameReader):
    """Narrow adapter: same strict PyAV index/PTS/RGB checks as FrameReader.

    The original reader intentionally rejects TEST; its defaults are untouched.
    This adapter can only read already-selected TEST evaluation records after approval.
    """
    def __init__(self, root, approval):
        guard_split('test', approval)
        super().__init__(root, cache_mib=0)
        self.approval = approval

    def read(self, records):
        guard_split('test', self.approval)
        import av
        result = []
        for row in records:
            if row['split'] != 'test' or len(row['indices']) != 1:
                raise ValueError('Approved single TEST frames only')
            source = row['source_file']
            _valid_source(source, 'test')
            path = (self.root/source).resolve()
            if self.root not in path.parents:
                raise ValueError('Source escaped dataset root')
            wanted, expected = row['indices'][0], Fraction(row['timestamps_exact'][0])
            value = None
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                origin = Fraction(container.start_time, av.time_base) if container.start_time is not None else Fraction(0)
                for i, frame in enumerate(container.decode(stream)):
                    if i == wanted:
                        if frame.is_corrupt or frame.pts is None or Fraction(frame.pts * frame.time_base)-origin != expected:
                            raise ValueError('Decoded TEST PTS mismatch')
                        if (frame.width, frame.height) != (224, 224):
                            raise ValueError('Expected native 224x224; no silent resize')
                        value = frame.to_ndarray(format='rgb24')
                        break
            if value is None:
                raise ValueError('Missing requested TEST frame')
            result.append([value])
        return result


def _selected_records(data, indices, data_root):
    guard_split(data.split, data.approval)
    records = [dict(data.records[i]) for i in indices]
    if data.split == 'test':
        # Resolve ONLY persisted timestamps, not a new 10 FPS sampling protocol.
        timelines = {}
        for r in records:
            source = r['source_file']
            _valid_source(source, 'test')
            if source not in timelines:
                path = (Path(data_root)/source).resolve()
                if Path(data_root).resolve() not in path.parents:
                    raise ValueError('Source escaped dataset root')
                import av
                with av.open(str(path)) as container:
                    count = container.streams.video[0].frames
                if not count:
                    raise ValueError('Source frame count is unavailable; require an existing exact frame-index manifest')
                timeline = packet_timeline(path, count)
                timelines[source] = {stamp: i for i, stamp in enumerate(timeline)}
                if len(timelines[source]) != len(timeline):
                    raise ValueError('Duplicate source timestamps cannot be resolved uniquely')
            stamp = Fraction(r['timestamps_exact'][0])
            if stamp not in timelines[source]:
                raise ValueError('Saved TEST timestamp missing from source timeline')
            r['indices'] = [timelines[source][stamp]]
    return records


def save_explanations(data, checkpoint, data_root, output, indices, *, device='cpu', selection='manual', video=None, input_pipeline=None):
    """Write a NEW folder with triptych PNGs, CSV, provenance and optional Top-5% table."""
    guard_split(data.split, data.approval)
    if digest(checkpoint) != data.checkpoint_sha256:
        raise ValueError('Checkpoint changed after prediction selection')
    indices = list(indices)
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(data.records) for i in indices):
        raise ValueError('Select unique, valid frame indices; requested category may be empty')
    if video is not None:
        valid = {p['record_index'] for p in video['contributors']}
        if not set(indices) <= valid:
            raise ValueError('Video figures must be selected from the full Top-5% contributor set')
    model, config, _ = load_model(checkpoint, device, split=data.split, approval=data.approval)
    layer, layer_name = target_layer(model, config.model)
    records = _selected_records(data, indices, data_root)
    preprocessing = getattr(model, 'explanation_preprocessing', None)
    if preprocessing is not None:
        if input_pipeline is None or input_pipeline.identity != preprocessing:
            raise ValueError('Face-crop checkpoint requires its exact saved input pipeline; full-frame explanations refused')
        if data.split != 'dev':
            raise PermissionError('Face experiment explainability is DEV-only; no TEST authorization in this experiment')
        dataset = input_pipeline.dataset(records, data_root, config, training=False, cache_mib=1)
    else:
        if input_pipeline is not None:
            raise ValueError('Cannot apply face preprocessing to an original full-frame checkpoint')
        dataset = LazyDataset(records, data_root, flip_probability=0, cache_mib=1)
    if data.split == 'test':
        dataset.reader = _ApprovedTestReader(data_root, data.approval)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores = {p['record_index']: p['probability'] for p in data.predictions}
    metadata = []
    try:
        for local_i, (record_i, row) in enumerate(zip(indices, records)):
            tensor, _, _ = dataset[local_i]  # exact existing LazyDataset normalization; no augmentation
            result = GradCAM(model, layer)(tensor.unsqueeze(0).to(device))
            heatmap, probability = result['heatmaps'][0], float(result['probabilities'][0])
            # Read the same raw RGB frame, rather than approximating inversion of normalized pixels.
            original = dataset.reader.read([row])[0][0]
            colour = plt.get_cmap('jet')(heatmap)[..., :3]
            overlay = np.clip(.6 * original/255. + .4 * colour, 0, 1)
            label = {0: 'Real', 1: 'Fake'}.get(row.get('label'), 'unavailable')
            timestamp = row['timestamps_exact'][0]
            difference = abs(probability-scores[record_i])
            if difference > .01:
                warnings.warn('FP32 probability differs from saved prediction by >0.01; inspect checkpoint/prediction provenance.', RuntimeWarning)
            figure, axes = plt.subplots(1, 3, figsize=(12, 4))
            try:
                image_title = 'Face-cropped input RGB' if preprocessing else 'Original RGB'
                for axis, image, title in zip(axes, (original, heatmap, overlay), (image_title, 'Fake-logit Grad-CAM', 'Overlay')):
                    axis.imshow(image, **({'cmap': 'jet', 'vmin': 0, 'vmax': 1} if title == 'Fake-logit Grad-CAM' else {}))
                    axis.set_title(title); axis.axis('off')
                header = f'{config.model} | {row["source_file"]} | {float(Fraction(timestamp)):.3f}s | GT {label}\n'
                header += f'Fake p={probability:.4f} (saved={scores[record_i]:.4f}); threshold={data.frame_threshold:g}'
                if video:
                    header += f' | Top-5% contributor; video p={video["probability"]:.4f}, k={video["k"]}/{video["total_frames"]}'
                if result['zero_maps'][0]:
                    header += ' | zero positive Fake evidence map'
                figure.suptitle(header, fontsize=9)
                figure.tight_layout()
                filename = f'frame_{record_i:06d}.png'
                figure.savefig(output/filename, dpi=150)
            finally:
                plt.close(figure)
            metadata.append({'figure': filename, 'model': config.model, 'split': data.split,
                'record_index': record_i, 'candidate_id': row['id'], 'source_file': row['source_file'],
                'timestamp_exact': timestamp, 'timestamp_seconds': float(Fraction(timestamp)),
                'source_index': row['indices'][0], 'label': row.get('label'), 'saved_probability': scores[record_i],
                'gradcam_fp32_probability': probability, 'probability_absolute_difference': difference,
                'frame_threshold': data.frame_threshold, 'selection': selection, 'target_score': 'Fake logit',
                'target_layer': layer_name, 'zero_positive_map': result['zero_maps'][0],
                'top5_contributor': video is not None, 'video_probability': video['probability'] if video else '',
                'top5_k': video['k'] if video else '', 'contribution_weight': 1/video['k'] if video else ''})
            print(f'{config.model} {data.split}: saved {local_i+1}/{len(records)} Grad-CAM figures', flush=True)
        with (output/'frames.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata[0])); writer.writeheader(); writer.writerows(metadata)
        if video:
            (output/'top5_video.json').write_text(json.dumps(video, indent=2), encoding='utf-8')
            with (output/'top5_contributors.csv').open('w', newline='', encoding='utf-8') as handle:
                fields = ['record_index', 'timestamp_exact', 'probability', 'weight']
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(video['contributors'])
        (output/'provenance.json').write_text(json.dumps({'split': data.split, 'model': config.model,
            'checkpoint_sha256': data.checkpoint_sha256, 'predictions_sha256': data.predictions_sha256,
            'module_sha256': digest(__file__), 'frame_threshold': data.frame_threshold,
            'video_threshold': data.video_threshold, 'topk_fraction': .05, 'precision': 'fp32',
            'target_score': 'Fake pre-sigmoid logit', 'target_layer': layer_name,
            'input_pipeline': preprocessing,
            'warning': 'Post-hoc coarse evidence, not a manipulation mask or causal proof; no tuning using TEST.'}, indent=2), encoding='utf-8')
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error)}), encoding='utf-8')
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--index', type=Path, help='Existing candidates.sqlite; required for DEV')
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New output folder; never overwritten')
    parser.add_argument('--category', choices=['tp', 'tn', 'fp', 'fn', 'high_confidence_correct', 'high_confidence_incorrect'], default='tp')
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=4266)
    parser.add_argument('--video', help='Relative split/video.mp4; explain ALL Top-5%% contributors instead of category')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frozen-policy', type=Path)
    parser.add_argument('--approval', help='Exact final-evaluation approval phrase; TEST only')
    args = parser.parse_args(argv)
    guard_split(args.split, args.approval)
    data = load_explanation_data(args.checkpoint, args.predictions, split=args.split,
        index_path=args.index, frozen_policy_path=args.frozen_policy, approval=args.approval)
    video = top5_contributors(data, args.video) if args.video else None
    indices = [p['record_index'] for p in video['contributors']] if video else select_examples(data, args.category, args.count, args.seed)
    if not indices:
        parser.error('No examples in the requested category; nothing decoded or explained')
    save_explanations(data, args.checkpoint, args.data_root, args.output, indices,
        device=args.device, selection='top5_contributors' if video else args.category, video=video)


if __name__ == '__main__':
    main()
