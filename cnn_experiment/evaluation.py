"""Unchanged frame/video metrics, histories, DEV plots and model comparison."""
import csv
import hashlib
import json
import math
from pathlib import Path
from cnn_experiment.data import write_json


def class_loss_metrics(loss_sums, loss_counts):
    """Return whole-epoch class means; an absent class stays explicitly undefined."""
    if set(loss_sums)!={0,1} or set(loss_counts)!={0,1}:
        raise ValueError('Loss sums and counts must contain real=0 and fake=1')
    if any(not isinstance(loss_counts[label],int) or loss_counts[label]<0 for label in (0,1)):
        raise ValueError('Class loss counts must be nonnegative integers')
    if any(not math.isfinite(loss_sums[label]) or loss_sums[label]<0 for label in (0,1)):
        raise ValueError('Class loss sums must be finite and nonnegative')
    means={label:(loss_sums[label]/loss_counts[label] if loss_counts[label] else None)
           for label in (0,1)}
    balanced=(means[0]+means[1])/2 if all(means[label] is not None for label in (0,1)) else None
    return {'real_class_loss':means[0],'fake_class_loss':means[1],
            'balanced_loss':balanced,
            'real_class_loss_samples':loss_counts[0],
            'fake_class_loss_samples':loss_counts[1]}


def binary_curves(labels, scores):
    """Tie-aware ROC/PR points and areas from saved probabilities."""
    if not labels or len(labels)!=len(scores) or any(y not in (0,1) for y in labels):
        raise ValueError('Nonempty, matching binary labels/scores required')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError('Finite probabilities in [0,1] required')
    positives=sum(labels); negatives=len(labels)-positives
    ordered=sorted(zip(scores,labels),reverse=True)
    tp=fp=0
    roc_fpr=[0.0]; roc_tpr=[0.0]
    pr_recall=[0.0]; pr_precision=[1.0]
    thresholds=[]; average_precision=0.0; pr_auc=0.0; roc_auc=0.0
    for i in range(len(ordered)):
        score,label=ordered[i]
        tp+=label; fp+=1-label
        if i+1<len(ordered) and ordered[i+1][0]==score:
            continue
        thresholds.append(score)
        recall=tp/positives if positives else 0.0
        precision=tp/(tp+fp)
        if positives:
            average_precision+=(recall-pr_recall[-1])*precision
            pr_auc+=(recall-pr_recall[-1])*(precision+pr_precision[-1])/2
        pr_recall.append(recall); pr_precision.append(precision)
        if positives and negatives:
            fpr=fp/negatives; tpr=recall
            roc_auc+=(fpr-roc_fpr[-1])*(tpr+roc_tpr[-1])/2
            roc_fpr.append(fpr); roc_tpr.append(tpr)
    return {'roc_auc':roc_auc if positives and negatives else None,
            'average_precision':average_precision if positives else None,
            'pr_auc':pr_auc if positives else None,
            'roc':{'fpr':roc_fpr,'tpr':roc_tpr,'thresholds':thresholds} if positives and negatives else None,
            'pr':{'recall':pr_recall,'precision':pr_precision,'thresholds':thresholds} if positives else None}


def binary_metrics(labels,scores,threshold=0.5):
    if not labels or len(labels)!=len(scores) or any(y not in (0,1) for y in labels):
        raise ValueError('Nonempty, matching binary labels/scores required')
    if not 0 <= threshold <= 1 or any(not math.isfinite(p) or not 0<=p<=1 for p in scores):
        raise ValueError('Finite probabilities and threshold in [0,1] required')
    predicted=[int(p>=threshold) for p in scores]
    tp=sum(y==1 and p==1 for y,p in zip(labels,predicted))
    tn=sum(y==0 and p==0 for y,p in zip(labels,predicted))
    fp=sum(y==0 and p==1 for y,p in zip(labels,predicted))
    fn=sum(y==1 and p==0 for y,p in zip(labels,predicted))
    positives=sum(labels); negatives=len(labels)-positives
    curves=binary_curves(labels,scores)
    sensitivity=tp/(tp+fn) if tp+fn else 0.0
    specificity=tn/(tn+fp) if tn+fp else 0.0
    return {'roc_auc':curves['roc_auc'],'average_precision':curves['average_precision'],
            'pr_auc':curves['pr_auc'],
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'f2':5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.0,
            'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn) if tp+fn else 0.0,
            'sensitivity':sensitivity,'specificity':specificity,
            'balanced_accuracy':(sensitivity+specificity)/2,
            'false_positive_rate':fp/negatives if negatives else 0.0,
            'false_negative_rate':fn/positives if positives else 0.0,
            'accuracy':(tp+tn)/len(labels),'threshold':threshold,'n':len(labels),
            'positive':positives,'negative':negatives,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'confusion_matrix':[[tn,fp],[fn,tp]]}


def video_predictions(records, predictions, policy):
    """Aggregate all scheduled predictions, including boundary/ambiguous samples."""
    from collections import defaultdict
    groups=defaultdict(list)
    for prediction in predictions:
        row=records[prediction['record_index']]
        groups[row['source_file']].append((row,prediction['probability']))
    result=[]
    for source,items in sorted(groups.items()):
        probabilities=sorted((p for _,p in items),reverse=True)
        if policy['aggregation']=='topk':
            count=max(1,math.ceil(len(probabilities)*policy['topk_fraction']))
            score=sum(probabilities[:count])/count
        elif policy['aggregation']=='proportion':
            score=sum(p>=policy['prediction_threshold'] for p in probabilities)/len(probabilities)
        else:
            raise ValueError('Unknown aggregation')
        row=items[0][0]
        if len({r['video_label'] for r,_ in items})!=1:
            raise ValueError('Inconsistent video labels')
        result.append({'source_file':source,'source_family':row['source_family'],
                       'target':row['video_label'],'probability':score,'samples':len(items)})
    return result


def evaluate_predictions(records,predictions,policy,kind='frame'):
    if kind not in ('frame','clip'):
        raise ValueError('Expected frame or clip evaluation')
    indices=[p['record_index'] for p in predictions]
    if sorted(indices)!=list(range(len(records))):
        raise ValueError('Evaluation needs exactly one prediction for every scheduled input')
    guarded=[p for p in predictions if records[p['record_index']]['metric_eligible']]
    scored=predictions if kind=='frame' else guarded
    if not scored:
        raise ValueError('No unambiguous samples for unit metrics')
    unit=binary_metrics([records[p['record_index']]['label'] for p in scored],
                        [p['probability'] for p in scored],policy['prediction_threshold'])
    guarded_metrics=(binary_metrics([records[p['record_index']]['label'] for p in guarded],
                        [p['probability'] for p in guarded],policy['prediction_threshold']) if guarded else None)
    videos=video_predictions(records,predictions,policy)
    video=binary_metrics([v['target'] for v in videos],[v['probability'] for v in videos],policy['video_threshold'])
    return {'unit':unit,'unit_scope':'all scheduled frame timestamps' if kind=='frame' else 'eligible clips only',
            'boundary_guarded_unit':guarded_metrics,'video':video,'videos':videos,
            'scheduled_predictions':len(predictions),'metric_eligible_predictions':len(scored),
            'metric_ineligible_predictions':len(predictions)-len(scored),
            'ignored_for_unit_metrics':len(predictions)-len(scored)}


def labelled_report(report,model,kind,split):
    """Name the prediction level without changing the existing metrics schema."""
    if kind not in ('frame','clip') or split not in ('dev','test'):
        raise ValueError('Expected DEV/TEST frame or clip report')
    return report | {'model':model,'split':split,'native_level':kind,
                     'native_metrics':report['unit'],'video_metrics':report['video']}


def select_dev_policy(records,predictions,base):
    """Freeze aggregation as supplied; select its video decision threshold on DEV."""
    if {r['split'] for r in records}!={'dev'}:
        raise ValueError('Aggregation selection accepts DEV only')
    videos=video_predictions(records,predictions,base)
    labels=[v['target'] for v in videos]; scores=[v['probability'] for v in videos]
    best=None; best_rank=None
    for threshold in sorted(set([base['video_threshold']]+[i/20 for i in range(21)])):
        metrics=binary_metrics(labels,scores,threshold)
        if metrics['roc_auc'] is None: raise ValueError('DEV needs both video classes')
        rank=(metrics['roc_auc'],metrics['f1'],metrics['accuracy'])
        if best_rank is None or rank>best_rank:
            best_rank=rank
            best=dict(base,video_threshold=threshold)
    return best


def localization_metrics(records,predictions,threshold,kind):
    """Sampled-point metrics for frames; temporal union metrics for clips."""
    from collections import defaultdict
    from cnn_experiment.data import merge_periods
    if kind=='frame':
        eligible=[p for p in predictions if records[p['record_index']]['metric_eligible']]
        return {'scope':'all sampled timestamps with exact [start,end) labels; guarded scores also reported',
                'sampled_frame_metrics':binary_metrics(
                    [records[p['record_index']]['label'] for p in predictions],
                    [p['probability'] for p in predictions],threshold),
                'boundary_guarded_frame_metrics':binary_metrics(
                    [records[p['record_index']]['label'] for p in eligible],
                    [p['probability'] for p in eligible],threshold) if eligible else None}
    grouped=defaultdict(list)
    for p in predictions: grouped[records[p['record_index']]['source_file']].append(p)
    intersection=predicted=actual=0.0
    by_video=[]
    for source,items in sorted(grouped.items()):
        row=records[items[0]['record_index']]
        spans=[[records[p['record_index']]['start_s'],
                min(records[p['record_index']]['end_s'],row['decoded_duration_s'])]
               for p in items if p['probability']>=threshold]
        pred=merge_periods(spans)
        truth=merge_periods(row['fake_periods_original']) if row['modify_video'] else []
        pd=float(sum(b-a for a,b in pred)); gt=float(sum(b-a for a,b in truth))
        overlap=float(sum(max(0,min(b,d)-max(a,c)) for a,b in pred for c,d in truth))
        union=pd+gt-overlap
        by_video.append({'source_file':source,'intersection_s':overlap,'predicted_s':pd,
                         'ground_truth_s':gt,'temporal_iou':overlap/union if union else None})
        predicted+=pd; actual+=gt; intersection+=overlap
    union=predicted+actual-intersection
    return {'scope':'union of high-scoring clip intervals; micro duration metrics',
            'temporal_iou':intersection/union if union else None,
            'temporal_precision':intersection/predicted if predicted else 0.0,
            'temporal_recall':intersection/actual if actual else 0.0,
            'intersection_s':intersection,'predicted_s':predicted,'ground_truth_s':actual,
            'videos':by_video}


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + '.'))
        else:
            result[name] = json.dumps(item) if isinstance(item, list) else item
    return result


def write_history_metrics(out, history):
    """Persist the logically appended history before any figure rendering."""
    out = Path(out)
    json_tmp = out/'history.jsonl.tmp'
    json_tmp.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in history), encoding='utf-8')
    json_tmp.replace(out/'history.jsonl')
    rows = [flatten(r) for r in history]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    csv_tmp = out/'history.csv.tmp'
    with csv_tmp.open('w', newline='', encoding='utf-8') as h:
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    csv_tmp.replace(out/'history.csv')


def plot_history(out, history):
    out = Path(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = [r['epoch'] for r in history]
    for key, label in (('loss', 'TRAIN (balanced, augmented)'),):
        axes[0, 0].plot(epochs, [r['train'][key] for r in history], label=label)
    axes[0, 0].plot(epochs, [r['dev']['loss'] for r in history], label='DEV natural')
    for key, label in (('balanced_loss', 'Balanced DEV'), ('real_class_loss', 'Real DEV'), ('fake_class_loss', 'Fake DEV')):
        axes[0, 1].plot(epochs, [r['dev'][key] for r in history], label=label)
    axes[1, 0].plot(epochs, [r['train']['roc_auc'] for r in history], label='TRAIN online')
    axes[1, 0].plot(epochs, [r['dev']['roc_auc'] for r in history], label='DEV frame')
    axes[1, 1].plot(epochs, [r['dev_video']['roc_auc'] for r in history], label='DEV video Top-5%')
    for ax, title in zip(axes.flat, ('BCE loss', 'Class-aware DEV BCE', 'Frame ROC-AUC', 'Video ROC-AUC')):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=150); plt.close(fig)


def write_history(out, history):
    """Backward-compatible helper; training commits metrics separately first."""
    write_history_metrics(out, history)
    plot_history(out, history)


def prediction_artifacts(out, records, predictions, report):
    out = Path(out)
    with (out / 'best_dev_predictions.csv').open('w', newline='', encoding='utf-8') as h:
        fields = ['candidate_id', 'source_file', 'source_family', 'timestamp_exact', 'label', 'probability']
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader()
        for p in predictions:
            row = records[p['record_index']]
            writer.writerow(dict(zip(fields, (row['id'], row['source_file'], row['source_family'],
                                              row['timestamps_exact'][0], row['label'], p['probability']))))
    write_json(out / 'best_dev_metrics.json', {k: v for k, v in report.items() if k != 'videos'})
    write_json(out / 'best_dev_video_predictions.json', report['videos'])
    curves = {}
    for name, labels, scores in (
        ('frame', [p['target'] for p in predictions], [p['probability'] for p in predictions]),
        ('video', [p['target'] for p in report['videos']], [p['probability'] for p in report['videos']])):
        curves[name] = binary_curves(labels, scores)
    write_json(out / 'best_dev_curves.json', curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for level, curve in curves.items():
        axes[0].plot(curve['roc']['fpr'], curve['roc']['tpr'], label=level)
        axes[1].plot(curve['pr']['recall'], curve['pr']['precision'], label=level)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='DEV ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='DEV precision-recall')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'best_dev_roc_pr.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, level in zip(axes, ('unit', 'video')):
        cm = report[level]['confusion_matrix']; ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha='center', va='center')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
               xlabel='Predicted', ylabel='True', title=('Frame' if level == 'unit' else 'Video') + f" threshold {report[level]['threshold']}")
    fig.tight_layout(); fig.savefig(out / 'best_dev_confusion.png', dpi=150); plt.close(fig)


def compare_runs(runs, output):
    rows = []
    for name, run in runs.items():
        run = Path(run)
        summary = json.loads((run / 'training_summary.json').read_text())
        metrics = json.loads((run / 'best_dev_metrics.json').read_text())
        rows.append({'model': name, 'status': summary['status'], 'best_epoch': summary['best_epoch'],
                     'dev_video_roc_auc': metrics['video']['roc_auc'], 'dev_frame_roc_auc': metrics['unit']['roc_auc'],
                     'dev_frame_pr_auc': metrics['unit']['pr_auc'],
                     'balanced_dev_loss': summary['best_balanced_dev_loss'], 'run': str(run)})
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with (output / 'model_comparison.csv').open('w', newline='', encoding='utf-8') as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    winner = max(rows, key=lambda r: (r['dev_video_roc_auc'], -r['balanced_dev_loss']))
    write_json(output / 'comparison.json', {'selection': 'DEV video ROC-AUC; balanced DEV loss breaks ties',
        'models': rows, 'selected_cnn': winner['model'], 'lstm_trained': False,
        'historical_comparison': 'Inspect frame ROC-AUC and class losses against documented history. Historical Top-10% video scores cannot establish a Top-5% improvement without re-aggregation of saved DEV predictions.'})
    print(json.dumps(rows, indent=2)); print('Selected CNN by DEV:', winner['model'])
    return rows


def write_epoch_diagnostics(out, history, records, predictions, report):
    """Save current-epoch DEV plots even when this is not the best checkpoint."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epoch = history[-1]['epoch']
    folder = Path(out)/'epoch_diagnostics'/f'epoch_{epoch:03}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'dev_metrics.json', {k: v for k, v in report.items() if k != 'videos'})
    curves = {}
    curves['frame'] = binary_curves([records[p['record_index']]['label'] for p in predictions],
                                     [p['probability'] for p in predictions])
    curves['video'] = binary_curves([p['target'] for p in report['videos']],
                                     [p['probability'] for p in report['videos']])
    write_json(folder/'dev_curves.json', curves)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, level, title in zip(axes[:2], ('unit', 'video'), ('Frame', 'Video')):
        cm = report[level]['confusion_matrix']
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha='center', va='center')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
               xlabel='Predicted', ylabel='True', title=f'{title} DEV epoch {epoch}')
    for level, curve in curves.items():
        axes[2].plot(curve['roc']['fpr'], curve['roc']['tpr'], label=level)
    axes[2].set(title='DEV ROC', xlabel='False positive rate', ylabel='True positive rate')
    axes[2].legend(); axes[2].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'dev_confusion_roc.png', dpi=120); plt.close(fig)
    numerics = [row['train'].get('numerics', {}) for row in history]
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for key, title, ax in zip(('gradient_norm_max_finite', 'amp_scale_end', 'skipped_steps'),
                              ('Max finite gradient norm (before clipping)', 'AMP scale after epoch', 'Skipped optimizer steps'), axes):
        ax.plot(epochs, [row.get(key) for row in numerics], marker='o')
        ax.set(title=title, xlabel='Epoch'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(Path(out)/'numerical_curves.png', dpi=120); plt.close(fig)
    return [Path(out)/'training_curves.png', folder/'dev_confusion_roc.png', Path(out)/'numerical_curves.png']


def display_epoch_diagnostics(paths):
    """Render in VS Code/Jupyter; CLI runs retain the same saved artifacts."""
    try:
        from IPython import get_ipython
        from IPython.display import display, Image
    except ImportError:
        return
    if get_ipython() is not None:
        for path in paths:
            display(Image(filename=str(path)))


def comparison_table(runs, output):
    """Existing selection policy, with frame/video precision, recall, F1 and losses."""
    from cnn_experiment.data import write_json
    import csv
    output = Path(output)
    if output.exists():
        raise FileExistsError('Comparison already exists; preserve it or choose a new comparison folder.')
    rows = compare_runs(runs, output)
    for row in rows:
        run = Path(row['run'])
        metrics = json.loads((run/'best_dev_metrics.json').read_text(encoding='utf-8'))
        losses = metrics['losses']
        row.update(dev_loss=losses['loss'], real_dev_loss=losses['real_class_loss'],
                   fake_dev_loss=losses['fake_class_loss'])
        digest = hashlib.sha256()
        with (run/'best.pt').open('rb') as checkpoint:
            for chunk in iter(lambda: checkpoint.read(2**20), b''):
                digest.update(chunk)
        row['best_checkpoint_sha256'] = digest.hexdigest()
        for level, key in (('frame', 'unit'), ('video', 'video')):
            for metric in ('precision', 'recall', 'f1'):
                row[f'dev_{level}_{metric}'] = metrics[key][metric]
    with (output/'model_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Preserve compare_runs' winner and selection rule; enrich display fields only.
    saved = json.loads((output/'comparison.json').read_text(encoding='utf-8'))
    saved['models'] = rows
    write_json(output/'comparison.json', saved)
    return rows
