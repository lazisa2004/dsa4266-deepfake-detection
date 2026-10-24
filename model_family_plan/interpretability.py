"""Separately executed DEV-only winner attribution. No changes to training paths."""
import csv
import contextlib
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from cnn_experiment.checkpoints import digest
from cnn_experiment.gradcam import (ExplanationData, load_explanation_data, select_examples, target_layer)
from cnn_experiment.data import LazyDataset
from .comparison import Artifacts, build_comparison, guarded, model_from_state, REGISTRY


def attribution(model, images, *, temporal=False):
    """Fake-logit gradient through convolution AND temporal classifier, with frozen weights.

    One frame or one 16-frame clip at a time. For temporal models ONLY this
    attribution forward bypasses extract_features' no_grad; the original training
    model implementation, parameter flags and state are never edited. Gradients
    flow from the final temporal logit through all 16 ordered encoder features.
    """
    if temporal:
        if tuple(images.shape)!=(1,16,3,224,224):raise ValueError('One original 16-frame clip required')
        layer=model.encoder.layer4[-1];layer_name='encoder.layer4[-1] -> original LSTM/classifier'
    else:
        if tuple(images.shape)!=(1,3,224,224):raise ValueError('One original CNN input required')
        layer,layer_name=target_layer(model,model.name)
    flags=[(m,m.training) for m in model.modules()];captures=[]
    handle=layer.register_forward_hook(lambda m,i,o:captures.append(o))
    try:
        model.eval()
        with torch.inference_mode(False),torch.enable_grad(),torch.autocast(images.device.type,enabled=False):
            x=images.detach().clone().float().requires_grad_(True)
            if temporal:
                features=model.encoder(x.flatten(0,1))
                if tuple(features.shape)!=(16,2048):raise ValueError('Original pooled encoder features required')
                logits=model.classify_features(features.reshape(1,16,2048)).reshape(-1)
            else:logits=model(x).reshape(-1)
            if len(captures)!=1 or captures[0].ndim!=4 or logits.numel()!=1:
                raise ValueError('Expected one convolutional output and one Fake logit')
            activation=captures[0];gradient,=torch.autograd.grad(logits.sum(),activation)
            if not all(bool(torch.isfinite(v).all()) for v in (logits,activation,gradient)):
                raise FloatingPointError('Non-finite attribution')
            weights=gradient.mean((2,3),keepdim=True)
            maps=F.interpolate((activation*weights).sum(1,keepdim=True).relu(),size=(224,224),mode='bilinear',align_corners=False)[:,0]
            # A single shared clip scale preserves relative frame strength, unlike independent frame normalization.
            maximum=maps.max();maps=maps/maximum.clamp_min(torch.finfo(maps.dtype).tiny)
            return {'heatmaps':maps.detach().cpu().numpy(),'probability':float(logits.detach().sigmoid()[0]),
                    'gradient_l1_per_frame':gradient.detach().abs().flatten(1).sum(1).cpu().tolist(),
                    'zero_positive_maps':(maps.flatten(1).amax(1)==0).cpu().tolist(),
                    'target_layer':layer_name,'target_score':'Fake pre-sigmoid logit',
                    'normalization':'one common positive maximum across the complete clip' if temporal else 'positive maximum of this frame'}
    finally:
        handle.remove();captures.clear()
        for module,training in flags:module.training=training


def _materialize(artifacts,out):
    if not artifacts.saved.archive:return artifacts.source
    target=out/'read_only_source_copy';target.mkdir()
    names=('best.pt','best_dev_predictions.csv','clip_manifest.json','input_pipeline.json','face_detector.onnx','face_landmarker.task','face_boxes.sqlite')
    for name in names:
        data=artifacts.read(name,2*1024**3 if name=='best.pt' else 128*1024**2)
        if data is not None:
            with (target/name).open('xb') as f:f.write(data)
    return target


def _mask_pipeline(source,out,identity):
    from cnn_face_experiment.preprocessing import FaceMaskPipeline,MaskConfig
    from cnn_face_experiment.crops import ensure_detector
    from cnn_face_experiment.preprocessing import ensure_landmarker
    detector=source/'face_detector.onnx';landmarks=source/'face_landmarker.task'
    ensure_detector(detector,download=False);ensure_landmarker(landmarks,download=False)
    params=dict(identity['mask']);params['neutral_rgb']=tuple(params['neutral_rgb'])
    # Write only a NEW interpretation cache, never open the old training cache for writing.
    cache=out/'interpretation_mask_cache.sqlite'
    if (source/'face_boxes.sqlite').is_file():shutil.copy2(source/'face_boxes.sqlite',cache)
    pipeline=FaceMaskPipeline(detector,landmarks,cache,
        identity['index_sha256'],identity['pool_sha256'],MaskConfig(**params))
    if pipeline.identity!=identity:raise ValueError('Cannot reproduce the checkpoint face-mask policy exactly')
    return pipeline


def _clip_data(source,checkpoint,index_path):
    from cnn_experiment.temporal import ClipIndex
    from .runner import INDEX_SHA256
    if digest(index_path)!=INDEX_SHA256:raise ValueError('Original fixed DEV index required')
    manifest=source/'clip_manifest.json'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if digest(manifest)!=state['index_sha256']:raise ValueError('Original clip manifest mismatch')
    if json.loads(manifest.read_text())['dev_index_sha256']!=INDEX_SHA256:raise ValueError('Clip DEV index mismatch')
    schedule=ClipIndex(manifest);records=schedule.records('dev','clip');schedule.close()
    by_key={(r['source_file'],Fraction(r['timestamps_exact'][0])):i for i,r in enumerate(records)}
    seen=set();predictions=[]
    with (source/'best_dev_predictions.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key=(row['source_file'],Fraction(row['timestamp_exact']))
            if not row['source_file'].startswith('dev/') or key not in by_key:raise ValueError('Prediction outside original DEV clips')
            i=by_key[key];r=records[i];p=float(row['probability'])
            if i in seen or r['id']!=int(row['candidate_id']) or r['label']!=int(row['label']) or not np.isfinite(p) or not 0<=p<=1:
                raise ValueError('Duplicate/invalid DEV clip prediction')
            seen.add(i);predictions.append({'record_index':i,'target':r['label'],'probability':p})
    if seen!=set(range(len(records))):raise ValueError('Incomplete fixed DEV clip predictions')
    predictions.sort(key=lambda p:p['record_index'])
    return ExplanationData('dev',records,predictions,.5,.5,digest(checkpoint),digest(source/'best_dev_predictions.csv'))


def explain_master(sources,index_path,data_root,output,imports=(),*,device='cpu',count_per_category=1,seed=4266):
    """Re-check overall leader, load its actual best.pt and explain only saved DEV samples.

    Missing/unverified leading weights cause an explicit stop; no fallback model.
    Output must be new and outside the source runs. This function is never called
    by the training wrappers or comparison builder.
    """
    guarded(index_path);guarded(data_root);out=guarded(output)
    for source in sources.values():
        if source is not None:
            p=Path(source).resolve()
            if p.is_dir() and (p==out.resolve() or p in out.resolve().parents):raise ValueError('Interpretability must not write inside an existing run')
    if not 1<=count_per_category<=2:raise ValueError('Bounded explanation: 1 or 2 examples per category')
    comparison=build_comparison(sources,imports);ranking=comparison['ranking'];winner=ranking['gradcam_winner']
    if winner is None:raise RuntimeError(ranking['gradcam_blocked_reason'])
    out.mkdir(parents=True,exist_ok=False)
    stack=contextlib.ExitStack();dataset=None;metadata=[];model=None
    try:
        artifacts=Artifacts(sources[winner])
        work=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='dsa4266_explain_'))) if artifacts.saved.archive else out
        source=_materialize(artifacts,work);checkpoint=source/'best.pt'
        original_hash=digest(checkpoint);state=artifacts.checkpoint()
        if state is None:raise RuntimeError('Selected winner checkpoint disappeared; no substitute model is used')
        model,cfg=model_from_state(state)
        temporal=cfg.model=='resnet50_lstm';identity=state.get('input_pipeline');del state
        model=model.to(device=device,dtype=torch.float32).eval()
        data=_clip_data(source,checkpoint,index_path) if temporal else load_explanation_data(checkpoint,source/'best_dev_predictions.csv',index_path=index_path,split='dev')
        # Main confusion categories always use the predeclared 0.5 threshold.
        data.frame_threshold=data.video_threshold=.5
        categories={cat:select_examples(data,cat,count_per_category,seed) for cat in ('tp','tn','fp','fn')}
        indices=list(dict.fromkeys(i for ids in categories.values() for i in ids))
        if not indices:raise ValueError('No eligible DEV examples')
        records=[data.records[i] for i in indices]
        if any(r['split']!='dev' for r in records):raise PermissionError('DEV only')
        pipeline=_mask_pipeline(source,out,identity) if identity else None
        if identity and identity.get('kind')!='face_mask':raise ValueError('Unsupported saved preprocessing; no substitution')
        dataset=(pipeline.dataset(records,data_root,cfg,training=False,cache_mib=4) if pipeline else
                 LazyDataset(records,data_root,flip_probability=0,cache_mib=4))
        probabilities={p['record_index']:p['probability'] for p in data.predictions}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for local,(record_i,row) in enumerate(zip(indices,records)):
            tensor,_,_=dataset[local]
            result=attribution(model,tensor.unsqueeze(0).to(device),temporal=temporal)
            probability=result['probability'];difference=abs(probability-probabilities[record_i])
            if difference>.01:raise ValueError('FP32 attribution differs from saved prediction by >0.01; inspect checkpoint/preprocessing provenance')
            images=dataset.reader.read([row])[0]
            category=next(cat for cat,ids in categories.items() if record_i in ids)
            for start in range(0,len(images),4):
                size=min(4,len(images)-start);fig,axes=plt.subplots(size,3,figsize=(12,3.7*size),squeeze=False)
                filename=f'{winner}_{category}_{record_i:06d}_frames_{start:02d}.png'
                try:
                    for j in range(size):
                        frame=start+j;image=images[frame];heatmap=result['heatmaps'][frame]
                        colour=plt.get_cmap('jet')(heatmap)[...,:3];overlay=np.clip(.6*image/255.+.4*colour,0,1)
                        for ax,pixels,title in zip(axes[j],(image,heatmap,overlay),(f'Actual {REGISTRY[winner][1]} input','Fake-logit Grad-CAM','Transparent overlay')):
                            ax.imshow(pixels,**({'cmap':'jet','vmin':0,'vmax':1} if title=='Fake-logit Grad-CAM' else {}));ax.set_title(title);ax.axis('off')
                        stamp=row['timestamps_exact'][frame]
                        axes[j,0].set_title(f'Actual {REGISTRY[winner][1]} input | {float(Fraction(stamp)):.3f}s')
                        if result['zero_positive_maps'][frame]:axes[j,1].set_title('Zero positive Fake evidence')
                        metadata.append({'figure':filename,'experiment_id':winner,'model':cfg.model,'split':'dev',
                            'source_file':row['source_file'],'candidate_id':row['id'],'level':'clip' if temporal else 'frame',
                            'frame_in_clip':frame if temporal else None,'timestamp_exact':stamp,'timestamp_seconds':float(Fraction(stamp)),
                            'ground_truth_level':'clip' if temporal else 'frame','ground_truth':row['label'],
                            'video_ground_truth':row['video_label'],'predicted_fake':probability>=.5,'fake_probability':probability,
                            'saved_probability':probabilities[record_i],'threshold':.5,'category':category,
                            'preprocessing':REGISTRY[winner][1],'gradient_l1':result['gradient_l1_per_frame'][frame],
                            'zero_positive_map':result['zero_positive_maps'][frame],'target_layer':result['target_layer']})
                    label='Fake' if row['label'] else 'Real';prediction='Fake' if probability>=.5 else 'Real'
                    first,last=row['timestamps_exact'][start],row['timestamps_exact'][start+size-1]
                    fig.suptitle(f'{winner} | {cfg.model} | {row["source_file"]} | {first} to {last}s\n'
                        f'{"CLIP" if temporal else "FRAME"} GT {label} | predicted {prediction} | Fake p={probability:.4f} | {category.upper()} | threshold 0.5',fontsize=10)
                    fig.tight_layout();fig.savefig(out/filename,dpi=130)
                finally:plt.close(fig)
        with (out/'examples.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(metadata[0]));writer.writeheader();writer.writerows(metadata)
        (out/'provenance.json').write_text(json.dumps({'experiment_id':winner,'checkpoint_sha256':original_hash,
            'predictions_sha256':data.predictions_sha256,'ranking':ranking,'categories':categories,
            'input_pipeline':identity,'split':'dev','threshold':.5,'target_score':'Fake pre-sigmoid logit',
            'temporal_attribution':temporal,'temporal_normalization':'common maximum across 16 frame maps' if temporal else None,
            'weights_unchanged':digest(checkpoint)==original_hash,'warning':'Coarse post-hoc evidence; not a manipulation mask or causal proof. No TEST tuning.'},indent=2),encoding='utf-8')
        if digest(checkpoint)!=original_hash:raise RuntimeError('Checkpoint changed during read-only interpretation')
        return out
    except Exception as error:
        (out/'failure.json').write_text(json.dumps({'error':type(error).__name__,'message':str(error)}),encoding='utf-8');raise
    finally:
        if dataset is not None and hasattr(dataset.reader,'boxes'):dataset.reader.boxes.close()
        del model
        stack.close()
