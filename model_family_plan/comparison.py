"""Read-only DEV comparisons. No inference, training, index generation or TEST."""
import csv
from html import escape
from html.parser import HTMLParser
import io
import json
import math
import pickle
from pathlib import Path
import zipfile

from cnn_experiment.checkpoints import digest, assert_finite_state
from cnn_experiment.evaluation import binary_metrics, metrics_with_f2
from cnn_experiment.reporting import SavedRun, epoch_table, at_half
from .plan import RECIPES, configuration

# Canonical identities; no filename aliases or old/new result substitution.
REGISTRY = {
    'original_resnet50': ('resnet50','full_frame','Original full-frame ResNet50'),
    'A': ('resnet50','face_mask','Previous face-masked ResNet50'),
    'E': ('resnet50','face_mask','Previous E'),
    '1a': ('resnet50','full_frame','New 1a'), '1b': ('resnet50','face_mask','New 1b'),
    'original_efficientnet_b0': ('efficientnet_b0','full_frame','Original full-frame EfficientNet-B0'),
    'B': ('efficientnet_b0','face_mask','Previous face-masked EfficientNet-B0'),
    'C': ('efficientnet_b0','full_frame','Previous C: higher head LR'),
    'D': ('efficientnet_b0','face_mask','Previous D'),
    '2a': ('efficientnet_b0','full_frame','New 2a'), '2b': ('efficientnet_b0','face_mask','New 2b'),
    'original_resnet50_lstm': ('resnet50_lstm','full_frame_clips','Original frozen ResNet50-LSTM'),
    '3a': ('resnet50_lstm','full_frame_clips','New 3a'),
}
METRIC_NAMES = {'ROC-AUC':'roc_auc','PR-AUC':'pr_auc','Precision':'precision','Recall':'recall','F1':'f1','F2':'f2'}
PERFORMANCE = ['TRAIN Loss','DEV Loss','Balanced DEV Loss'] + [f'{level} {key}' for level in ('Frame','Clip','Video') for key in METRIC_NAMES] + ['Video Accuracy']


def previous_configs():
    from cnn_face_experiment.experiments import experiment_configurations
    from cnn_face_experiment.roc_experiments import roc_experiment_configurations
    ids={'A':'resnet50_face_crop','B':'efficientnet_b0_face_crop','C':'efficientnet_b0_head_lr_2e4',
         'D':'efficientnet_b0_face_mask_roc_d','E':'resnet50_face_mask_roc_e'}
    configs=experiment_configurations()|roc_experiment_configurations()
    return {name:configs[key].as_dict() for name,key in ids.items()}


def guarded(path):
    path=Path(path)
    if any(p.lower() in ('test','final_test') for p in path.parts):
        raise PermissionError('Only saved TRAIN/DEV experiment artifacts are accepted')
    return path


class Artifacts:
    """Bounded named members, folders or single-run ZIPs; never extract arbitrary paths."""
    def __init__(self, source):
        self.source=guarded(source);self.saved=SavedRun(self.source)
        if self.saved.single_history:raise ValueError('Use an audited historical import for a lone history')

    def read(self,name,limit=32*1024**2):
        if '/' in name or '\\' in name or name.startswith('.') or 'test' in name.lower():
            raise PermissionError('Explicit DEV/run artifact names only')
        if self.saved.archive:
            with zipfile.ZipFile(self.source) as z:
                try:
                    item=z.getinfo(self.saved.prefix+name)
                except KeyError:return None
                if item.file_size>limit:raise ValueError('Oversized saved artifact: '+name)
                return z.read(item)
        path=self.source/name
        if not path.is_file():return None
        if path.stat().st_size>limit:raise ValueError('Oversized saved artifact: '+name)
        return path.read_bytes()

    def json(self,name):
        value=self.read(name)
        return json.loads(value.decode('utf-8-sig')) if value is not None else None

    def checkpoint(self):
        import torch
        try:
            if self.saved.archive:
                data=self.read('best.pt',2*1024**3)
                if data is None:return None
                state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
            else:
                path=self.source/'best.pt'
                if not path.is_file():return None
                state=torch.load(path,map_location='cpu',weights_only=True)
        except (pickle.UnpicklingError,EOFError,RuntimeError,IndexError) as error:
            raise ValueError('Cannot safely load best.pt: '+str(error)) from None
        if not isinstance(state,dict):raise ValueError('best.pt is not an experiment checkpoint mapping')
        return state


def model_from_state(state):
    """Strict weights-only architecture reconstruction; no downloads, optimizer or RNG changes."""
    import torch
    from torch import nn
    from cnn_experiment.models import StagedCNN
    from cnn_experiment.config import Config
    from cnn_experiment.temporal import TemporalConfig, ResNetLSTM, TEMPORAL_PROTOCOL
    from cnn_experiment.data import PROTOCOL
    raw=state['config'];temporal=raw.get('model')=='resnet50_lstm'
    if state.get('protocol')!=(TEMPORAL_PROTOCOL if temporal else PROTOCOL):
        raise ValueError('Unsupported checkpoint protocol; import historical metrics without relabeling weights')
    cls=TemporalConfig if temporal else Config
    fields=cls.__dataclass_fields__
    cfg=cls(**{k:v for k,v in raw.items() if k in fields}).validate()
    assert_finite_state(state['model'],'best.model')
    with torch.random.fork_rng(devices=[]):
        if temporal:
            encoder=StagedCNN(Config(model='resnet50',dropout=cfg.dropout),'.',pretrained=False).network
            encoder.fc=nn.Identity();model=ResNetLSTM(encoder,cfg)
        else:model=StagedCNN(cfg,'.',pretrained=False)
    model.load_state_dict(state['model'],strict=True)
    return model.eval(),cfg


def _mode(pipeline, temporal=False):
    return 'full_frame_clips' if temporal and not pipeline else {'face_mask':'face_mask','face_crop':'crop_30'}.get((pipeline or {}).get('kind'),'full_frame' if not pipeline else 'unknown')


def _identity(name,cfg,pipeline,plan=None):
    model,mode,_=REGISTRY[name]
    if cfg.get('model')!=model or _mode(pipeline,model=='resnet50_lstm')!=mode:
        raise ValueError('Architecture/preprocessing differs from canonical ID '+name)
    if plan and plan.get('experiment_id') in REGISTRY and plan['experiment_id']!=name:
        raise ValueError('Recorded experiment ID differs; do not double-count or rename it')
    if name.startswith('original_') and plan and plan.get('experiment_id') not in (None,name,model):
        raise ValueError('A separately identified experiment cannot be relabelled as an original baseline')
    if name in RECIPES:
        expected=configuration(name).as_dict()
        if not plan or plan.get('experiment_id')!=name:raise ValueError('Missing exact new experiment ID')
        fields=('head_lr','backbone_lr','epochs','patience','checkpoint_selection')
        if any(cfg.get(k)!=expected[k] for k in fields):raise ValueError('New configuration identity differs')
    elif name in ('A','B','C','D','E'):
        expected=previous_configs()[name]
        if any(cfg.get(k)!=expected[k] for k in ('head_lr','backbone_lr','epochs','patience','checkpoint_selection')):
            raise ValueError('Historical recipe differs; not '+name)


def blank(name):
    model,mode,label=REGISTRY[name]
    cfg=configuration(name).as_dict() if name in RECIPES else previous_configs().get(name,{})
    row={'Experiment ID':name,'Model':model,'Description':label,'Preprocessing':mode,
         'Head/Temporal LR':cfg.get('head_lr'),'Backbone LR/Frozen':'Frozen' if model=='resnet50_lstm' else cfg.get('backbone_lr'),
         'Max Epochs':cfg.get('epochs'),'Patience':cfg.get('patience'),'Best Epoch':None,
         'Checkpoint Selection':cfg.get('checkpoint_selection'),'Classification Threshold':.5,
         'Configuration Source':'Declared recipe; not proof of a historical run' if cfg else 'Unknown historical configuration',
         'Recorded Threshold':None,'Training Status':'Pending' if name in RECIPES else 'Unknown (not attached)',
         'Checkpoint Status':'Pending' if name in RECIPES else 'Missing checkpoint',
         'Status':'Pending' if name in RECIPES else 'Pending verification','Verified':False,'Rank Eligible':False,
         'Checkpoint SHA256':None,'Source':None,'Issue':None if name in RECIPES else 'Historical completion cannot be inferred without saved artifacts'}
    row.update({key:None for key in PERFORMANCE});return row


def _fill(row,unit,video,losses,train):
    unit=at_half(unit,.5);video=at_half(video,.5)
    # Accuracy is threshold-dependent too (the legacy helper covers P/R/F only).
    if video.get('threshold',.5)!=.5:video.pop('accuracy',None)
    level='Clip' if row['Model']=='resnet50_lstm' else 'Frame'
    for column,key in METRIC_NAMES.items():
        row[level+' '+column]=unit.get(key);row['Video '+column]=video.get(key)
    if video.get('accuracy') is None and all(video.get(k) is not None for k in ('tp','tn','fp','fn')) and video.get('threshold',.5)==.5:
        n=sum(video[k] for k in ('tp','tn','fp','fn'));video['accuracy']=(video['tp']+video['tn'])/n if n else None
    row.update({'Video Accuracy':video.get('accuracy'),'TRAIN Loss':train.get('loss'),
                'DEV Loss':losses.get('loss',unit.get('loss')),
                'Balanced DEV Loss':losses.get('balanced_loss',unit.get('balanced_loss'))})


def load_row(name,source):
    """Partial histories never become completed because best.pt happens to exist."""
    row=blank(name)
    if source is None:return row,[]
    if not guarded(source).exists():
        row.update({'Source':str(source),'Issue':'Saved artifact is not attached/available','Checkpoint Status':'Missing checkpoint'})
        return row,[]
    row['Source']=str(source);calibrated=[]
    try:
        artifacts=Artifacts(source);cfg=artifacts.json('config.json') or {};summary=artifacts.json('training_summary.json') or {}
        pipeline=artifacts.json('input_pipeline.json');plan=artifacts.json('experiment_plan.json')
        _identity(name,cfg,pipeline,plan)
        rows,best,_=epoch_table(artifacts.saved)
        history=__import__('cnn_experiment.reporting',fromlist=['load_history']).load_history(artifacts.saved)
        final=summary.get('status') in ('trained','resumed','completed') and isinstance(summary.get('epochs_run'),int) and summary['epochs_run']==len(history) and bool(history)
        row.update({'Head/Temporal LR':cfg.get('head_lr'),'Backbone LR/Frozen':'Frozen' if row['Model']=='resnet50_lstm' else cfg.get('backbone_lr'),
            'Max Epochs':cfg.get('epochs'),'Patience':cfg.get('patience'),'Best Epoch':best,
            'Checkpoint Selection':cfg.get('checkpoint_selection','video_roc_auc'),'Recorded Threshold':cfg.get('threshold'),
            'Training Status':'Completed' if final else 'Partial','Checkpoint Status':'Not verified','Status':'Partial',
            'Configuration Source':'Saved config.json','Issue':None})
        selected=next((r for r in history if r['epoch']==best),{})
        metrics=artifacts.json('best_dev_metrics.json') or {}
        unit=dict(metrics.get('unit',{}));unit.update(selected.get('dev',{}))
        video=dict(metrics.get('video',{}));video.update(selected.get('dev_video',{}))
        unit.setdefault('threshold',cfg.get('threshold'));video.setdefault('threshold',cfg.get('threshold'))
        for level,values in (('Clip' if row['Model']=='resnet50_lstm' else 'Frame',unit),('Video',video)):
            threshold=values.get('threshold')
            if _number(threshold) and threshold!=.5:
                calibrated.append({'Experiment ID':name,'Level':level,'Scope':'Saved best-epoch DEV metrics at recorded threshold; separate from 0.5 comparison',
                                   'Threshold':threshold,**metrics_with_f2(values)})
        videos=artifacts.json('best_dev_video_predictions.json')
        if videos and final:
            if any(not str(v['source_file']).startswith('dev/') for v in videos):raise ValueError('Only DEV predictions are allowed')
            video=binary_metrics([v['target'] for v in videos],[v['probability'] for v in videos],.5)
        _fill(row,unit,video,metrics.get('losses',{}),selected.get('train',{}))
        for filename,level in (('dev_f2_calibration.json','Frame/Clip'),('video_dev_f2_calibration.json','Video')):
            report=artifacts.json(filename)
            if report:
                calibrated.append({'Experiment ID':name,'Level':level,'Scope':report.get('scope','saved DEV calibrated reporting'),
                    'Threshold':report.get('threshold'),**metrics_with_f2(report.get('reporting',{}))})
        state=artifacts.checkpoint()
        if state is None:
            row.update({'Checkpoint Status':'Missing checkpoint','Status':'Missing checkpoint' if final else 'Partial'})
            return row,calibrated
        _identity(name,state.get('config',{}),state.get('input_pipeline'),plan)
        if (state.get('epoch',-1)+1!=best or state.get('best_epoch')!=best or best not in [r['epoch'] for r in history]
            or state.get('config',{}).get('checkpoint_selection','video_roc_auc')!=row['Checkpoint Selection']):
            raise ValueError('best.pt epoch/selection differs from saved final summary')
        if summary.get('checkpoint_selection',row['Checkpoint Selection'])!=row['Checkpoint Selection']:
            raise ValueError('Final summary selection differs from config')
        from cnn_experiment.config import checkpoint_score
        expected_score=checkpoint_score({'unit':selected.get('dev',{}),'video':selected.get('dev_video',{})},row['Checkpoint Selection'])
        if not _number(state.get('best')) or not math.isclose(state['best'],expected_score,abs_tol=1e-8):
            raise ValueError('Checkpoint selection score differs from actual selected history epoch')
        from .runner import INDEX_SHA256
        from cnn_experiment.temporal import TEMPORAL_PROTOCOL
        if state.get('protocol')==TEMPORAL_PROTOCOL:
            manifest=artifacts.json('clip_manifest.json')
            if not manifest or manifest.get('dev_index_sha256')!=INDEX_SHA256:raise ValueError('Original DEV clip schedule identity unavailable')
            import hashlib
            if state.get('index_sha256')!=hashlib.sha256(artifacts.read('clip_manifest.json')).hexdigest():raise ValueError('Clip manifest hash differs')
        elif state.get('index_sha256')!=INDEX_SHA256:raise ValueError('Original DEV frame schedule identity differs')
        if cfg.get('topk_fraction')!=.05:raise ValueError('Comparison requires original Top-5% mean pooling')
        model,_=model_from_state(state);del model
        if artifacts.saved.archive:
            import hashlib
            row['Checkpoint SHA256']=hashlib.sha256(artifacts.read('best.pt',2*1024**3)).hexdigest()
        else:row['Checkpoint SHA256']=digest(artifacts.source/'best.pt')
        row['Checkpoint Status']='Verified best.pt';row['Verified']=True
        row['Rank Eligible']=final and _number(row['Video ROC-AUC'])
        row['Status']='Completed' if final else 'Partial'
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,EOFError,zipfile.BadZipFile) as error:
        row.update({'Checkpoint Status':'Invalid/unverified','Status':'Invalid/unverified','Issue':str(error),'Verified':False,'Rank Eligible':False})
    for key,value in row.items():
        if isinstance(value,float) and not math.isfinite(value):row[key]=None
    return row,calibrated


def collect_sources(namespace,overrides=None):
    """Refresh references each time; never search/guess unrelated historical folders."""
    sources={}
    for model in ('resnet50','efficientnet_b0'):
        value=namespace.get('BASELINE_RUNS',{}).get(model)
        if value is None:value=namespace.get('BASELINE_ZIPS',{}).get(model)
        sources['original_'+model]=value
    sources['original_resnet50_lstm']=namespace.get('LSTM_REPORT_RUN')
    previous={'A':'resnet50_face_crop','B':'efficientnet_b0_face_crop','C':'efficientnet_b0_head_lr_2e4',
              'D':'efficientnet_b0_face_mask_roc_d','E':'resnet50_face_mask_roc_e'}
    for name,key in previous.items():
        value=None
        groups=('RUNS','SAVED_EXPERIMENT_RUNS','RUN_PATHS') if name in ('A','B','C') else ('ROC_RUNS','ROC_SAVED_RUNS','ROC_RUN_PATHS')
        for group in groups:
            candidate=namespace.get(group,{}).get(key)
            if candidate is not None and (group not in ('RUN_PATHS','ROC_RUN_PATHS') or Path(candidate).exists()):
                value=candidate;break
        sources[name]=value
    for name in RECIPES:
        value=namespace.get('NEW_RUNS',{}).get(name)
        if value is None:
            path=namespace.get('NEW_OUTPUTS',{}).get(name)
            if path is not None and Path(path).exists():value=path
        sources[name]=value
    sources.update(overrides or {});return sources


def _number(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def build_comparison(sources,imports=(),*,new_only=False):
    unknown=set(sources)-set(REGISTRY)
    if unknown:raise ValueError('Unknown IDs; do not alias runs: '+str(sorted(unknown)))
    imported={}
    for source in imports:
        data=json.loads(guarded(source).read_text(encoding='utf-8'))
        if data.get('schema')!='audited_dev_import_v1':raise ValueError('Use audited historical imports')
        for record in data['records']:
            key=record['experiment_id']
            if key in imported:raise ValueError('Duplicate historical import for '+key)
            imported[key]=record
    rows=[];calibrated=[];seen={};seen_weights={}
    for name in REGISTRY:
        if new_only and name not in RECIPES:continue
        source=sources.get(name)
        if source is not None:
            resolved=str(guarded(source).resolve())
            if resolved in seen:raise ValueError('Same artifact assigned to '+seen[resolved]+' and '+name)
            seen[resolved]=name
        row,extra=load_row(name,source)
        identity=row.get('Checkpoint SHA256')
        if identity:
            if identity in seen_weights:raise ValueError('Same best.pt assigned to '+seen_weights[identity]+' and '+name)
            seen_weights[identity]=name
        if source is None and name in imported:
            r=imported[name];row.update(r['row']);row.update({'Verified':False,'Rank Eligible':False,
                'Checkpoint Status':'Missing checkpoint','Status':'Missing checkpoint','Source':r['evidence']})
        rows.append(row);calibrated.extend(extra)
    return {'rows':rows,'ranking':rank(rows),'calibrated':calibrated,'test_used':False}


def rank(rows):
    complete=[r for r in rows if r['Training Status']=='Completed' and _number(r.get('Video ROC-AUC'))]
    eligible=[r for r in complete if r.get('Rank Eligible') and r.get('Verified')]
    key=lambda r:(r['Video ROC-AUC'],r.get('Video PR-AUC') if _number(r.get('Video PR-AUC')) else -1)
    ordered=sorted(eligible,key=key,reverse=True)
    # Preserve visibility of a higher reported result with lost weights; never silently substitute it.
    reported=max(complete,key=key) if complete else None
    winner=max(eligible,key=key) if eligible else None
    blocked=reported is not None and not reported.get('Rank Eligible') and (winner is None or key(reported)>=key(winner))
    top=[r for r in complete if r['Video ROC-AUC']==max(x['Video ROC-AUC'] for x in complete)] if complete else []
    unresolved=len(top)>1 and any(not _number(r.get('Video PR-AUC')) for r in top)
    leaders={}
    for metric in ('Video ROC-AUC','Video PR-AUC','Video F1','Video F2'):
        valid=[r for r in eligible if _number(r.get(metric))]
        leaders[metric]=[r['Experiment ID'] for r in valid if r[metric]==max(x[metric] for x in valid)] if valid else []
    return {'ordered_verified_ids':[r['Experiment ID'] for r in ordered],'leaders':leaders,
        'verified_winner':winner['Experiment ID'] if winner else None,
        'highest_reported_completed':reported['Experiment ID'] if reported else None,
        'gradcam_winner':None if blocked or unresolved else winner['Experiment ID'] if winner else None,
        'gradcam_blocked_reason':('Highest video ROC-AUC is tied but saved video PR-AUC is missing; tie-break cannot be invented.') if unresolved else ('Highest reported completed model '+reported['Experiment ID']+' lacks a verified checkpoint; no substitution.') if blocked else 'No completed verified checkpoint available.' if winner is None else None,
        'rule':'Full fixed DEV video ROC-AUC; video PR-AUC breaks ties. Threshold-0.5 F1/F2 leaders separately.'}


def comparison_html(rows,title):
    best={}
    for metric in PERFORMANCE:
        valid=[r[metric] for r in rows if r.get('Rank Eligible') and _number(r.get(metric))]
        if valid:best[metric]=(min if 'Loss' in metric else max)(valid)
    columns=list(rows[0]) if rows else []
    html=['<div style="overflow:auto;max-width:100%;max-height:680px"><table style="white-space:nowrap;border-collapse:collapse"><caption>'+escape(title)+'</caption><tr>']
    html += ['<th style="position:sticky;top:0;background:#e9eef5;padding:6px">'+escape(c)+'</th>' for c in columns]
    for row in rows:
        html.append('</tr><tr>')
        for col in columns:
            v=row.get(col)
            rendered='N/A' if v is None else (f'{v:.4e}' if 'LR' in col and _number(v) else f'{v:.4f}' if _number(v) and col in PERFORMANCE else str(v))
            highlight=row.get('Rank Eligible') and col in best and v==best[col]
            html.append('<td style="padding:6px;border:1px solid #ddd;'+('background:#fff0b3;font-weight:bold' if highlight else '')+'">'+escape(rendered)+'</td>')
    return ''.join(html)+'</tr></table></div>'


def show_comparison(sources,output,imports=(),*,new_only=False):
    """Re-run this read-only cell as runs finish. Save to a NEW snapshot directory."""
    from IPython.display import HTML,display
    result=build_comparison(sources,imports,new_only=new_only);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    title='A. New Experiment Comparison' if new_only else 'B. Overall Master Comparison'
    html=comparison_html(result['rows'],title+' | yellow: best verified completed values only')
    (out/'comparison.html').write_text(html,encoding='utf-8')
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    display(HTML(html));print(json.dumps(result['ranking'],indent=2))
    if result['calibrated']:
        from cnn_experiment.reporting import table_html
        display(HTML(table_html(result['calibrated'],title='SEPARATE DEV calibrated-threshold results; not full-DEV 0.5 ranking')))
    return result


class _Table(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=[];self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.cell=''
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
        if tag=='tr' and self.row:self.rows.append(self.row)


def import_notebook_table(notebook,cell_index,output_index,mapping,output,*,threshold=.5,selection='frame_pr_auc',completed_log_cells=()):
    """Explicit auditable import of an EXISTING rendered comparison; never infer old best epoch.

    Indices are zero-based. mapping maps exact saved model strings to canonical IDs.
    Completion needs the model's saved `| completed |` log, not a best-row guess.
    Rounded display values remain rounded; no hidden precision or epoch-10 losses added.
    """
    path=guarded(notebook);nb=json.loads(path.read_text(encoding='utf-8'))
    html=nb['cells'][cell_index]['outputs'][output_index]['data']['text/html']
    html=''.join(html) if isinstance(html,list) else html
    parser=_Table();parser.feed(html)
    header,*values=parser.rows;records=[]
    logs='\n'.join(''.join(o.get('text',[])) for i in completed_log_cells for o in nb['cells'][i].get('outputs',[]))
    for values in values:
        saved=dict(zip(header,values));model=saved.get('model')
        if model not in mapping:continue
        name=mapping[model];row=blank(name)
        if REGISTRY[name][0]!=model:raise ValueError('Do not relabel a different model')
        model_logs={'resnet50':('ResNet50',),'efficientnet_b0':('EfficientNet-B0',),
                    'resnet50_lstm':('resnet50_lstm','ResNet50-LSTM','ResNet50–LSTM')}[model]
        complete=any(model_log+' | completed |' in logs for model_log in model_logs)
        if not complete:raise ValueError('No saved completion log for '+model)
        def get(key):
            value=saved.get(key)
            return float(value) if value not in (None,'','N/A','None') else None
        prefix='dev_clip_' if model=='resnet50_lstm' else 'dev_frame_'
        unit={key:get(prefix+key) for key in METRIC_NAMES.values()};unit['threshold']=threshold
        video={key:get('dev_video_'+key) for key in METRIC_NAMES.values()};video['threshold']=threshold
        _fill(row,unit,video,{'loss':get('dev_loss'),'balanced_loss':get('balanced_dev_loss')},{})
        row.update({'Best Epoch':int(saved['best_epoch']),'Training Status':'Completed','Checkpoint Selection':selection,'Recorded Threshold':threshold})
        records.append({'experiment_id':name,'row':row,'evidence':{'file':path.name,'sha256':digest(path),
            'cell_index':cell_index,'cell_id':nb['cells'][cell_index].get('id'),'output_index':output_index,
            'completion_log_cells':list(completed_log_cells),'raw_saved_row':saved,
            'precision':'saved rendered table precision; derived F2 uses these saved P/R values'}})
    if len(records)!=len(mapping):raise ValueError('Every declared model must occur exactly once')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump({'schema':'audited_dev_import_v1','records':records},f,indent=2,allow_nan=False)
    return out
