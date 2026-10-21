"""Deterministic family-separated DEV-only threshold selection and reporting."""
from collections import defaultdict
from pathlib import Path
from .data import stable_seed, write_json
from .evaluation import binary_metrics, dev_threshold_analysis


def family_partition(records,seed=4266):
    groups=defaultdict(list)
    for i,r in enumerate(records):
        if r['split']!='dev' or not r['source_file'].startswith('dev/') or not r['source_family'].startswith('dev/'):
            raise ValueError('Threshold calibration is DEV only')
        groups[r['source_family']].append(i)
    # Stratify by whether a family contains any visually fake scheduled input.
    strata=defaultdict(list)
    for family,ids in groups.items():strata[int(any(records[i]['label'] for i in ids))].append(family)
    calibration=[];reporting=[]
    for families in strata.values():
        families.sort(key=lambda f:stable_seed(seed,'dev_threshold_partition',f))
        for j,family in enumerate(families):
            (calibration if j%2==0 else reporting).extend(groups[family])
    calibration.sort();reporting.sort()
    if any(len({records[i]['label'] for i in ids})!=2 for ids in (calibration,reporting)):
        raise ValueError('Not enough separate DEV families with both classes; no fallback to in-sample threshold reporting')
    return calibration,reporting


def calibrated_f2(records,predictions,seed=4266):
    if sorted(p['record_index'] for p in predictions)!=list(range(len(records))):raise ValueError('Complete DEV predictions required')
    by_index={p['record_index']:p for p in predictions};cal,rep=family_partition(records,seed)
    def values(ids):return [records[i]['label'] for i in ids],[by_index[i]['probability'] for i in ids]
    sweep=dev_threshold_analysis(*values(cal));threshold=sweep['max_f2']['threshold']
    return {'scope':'DEV family-separated calibration/reporting; full DEV was used for checkpoint selection, so reporting is not an untouched final holdout',
        'selection':'maximum calibration F2; equal F2 chooses highest threshold (descending sweep)',
        'seed':seed,'threshold':threshold,'calibration':binary_metrics(*values(cal),threshold),
        'reporting':binary_metrics(*values(rep),threshold),'reporting_at_0_5':binary_metrics(*values(rep),.5),
        'calibration_sweep':sweep,'calibration_families':sorted({records[i]['source_family'] for i in cal}),
        'reporting_families':sorted({records[i]['source_family'] for i in rep}),
        'calibration_ids':[records[i]['id'] for i in cal],'reporting_ids':[records[i]['id'] for i in rep],
        'full_dev_ranking_metrics_unchanged':True,'test_used':False}


def save_calibration(out,records,predictions,seed=4266):
    report=calibrated_f2(records,predictions,seed);out=Path(out)
    write_json(out/'dev_f2_calibration.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,4));cm=report['reporting']['confusion_matrix'];ax.imshow(cm,cmap='Blues',vmin=0)
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],
        xlabel='Predicted',ylabel='True',title=f"DEV reporting families; F2 threshold {report['threshold']:.4f}")
    fig.tight_layout();fig.savefig(out/'dev_f2_reporting_confusion.png',dpi=150);plt.close(fig)
    return report
