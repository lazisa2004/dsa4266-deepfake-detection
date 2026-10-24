"""Isolated new runs; each preflight/training call accepts exactly one new ID."""
from dataclasses import asdict
import json
from pathlib import Path
import zipfile

from .plan import RECIPES, REVIEW_ORDER, configuration, declared_plan

INDEX_SHA256 = 'ac3cd8c440f0bb1e53cf955bcc6a2596e9ee1bc40e508f254c82e10bde13cfcb'
POOL_SHA256 = '8bcb5c9845f92b43e36213f22e410bcecedf983f7db87250f02a9968868eea92'


def new_session(base='/kaggle/working/model_family_plan'):
    from cnn_experiment.workflow import create_session
    from cnn_experiment.data import write_json
    session = create_session(base)
    write_json(session/'declared_plan.json', declared_plan())
    return session


def output_path(session, experiment_id):
    configuration(experiment_id)
    return Path(session)/experiment_id/'run'


def require_first_result(experiment_id, completed):
    configuration(experiment_id)
    if experiment_id != '1a' and completed.get('1a') is None:
        raise RuntimeError('Complete and review NEW 1a first, or explicitly attach its saved run. Nothing starts automatically.')
    if completed.get(experiment_id) is not None:
        raise FileExistsError('This new ID already has a completed result; replay it rather than retraining')


def _new_output(experiment_id, output):
    output = Path(output).resolve()
    if output.name != 'run' or output.parent.name != experiment_id:
        raise ValueError('Use output_path(session, exact_new_id); historical directories are not accepted')
    if output.exists() or any((output.parent/name).exists() for name in ('experiment_plans.json','git_provenance.json')):
        raise FileExistsError('Existing run/plans preserved; use a new session for this ID')
    return output


def _cnn_inputs(index_path, pool_path, data_root):
    from cnn_experiment.checkpoints import digest
    from cnn_experiment.pool import FixedPool, DevSchedule
    if digest(index_path) != INDEX_SHA256 or digest(pool_path) != POOL_SHA256:
        raise ValueError('Use the original index/fixed TRAIN pool; no replacement is built')
    pool = FixedPool(pool_path, data_root)
    dev = DevSchedule(index_path, data_root)
    try:
        return {'index_sha256': INDEX_SHA256, 'pool_sha256': pool.sha256,
                'dev': dev.report, 'train': pool.report(), 'test_used': False}
    finally:
        dev.close()


def _check_mask(pipeline):
    from cnn_face_experiment.preprocessing import MaskConfig
    from cnn_face_experiment.crops import CropConfig
    if pipeline is None or pipeline.identity.get('kind') != 'face_mask':
        raise ValueError('This ID requires the original face_mask pipeline explicitly')
    expected = {'mask': asdict(MaskConfig()), 'crop': asdict(CropConfig()),
                'index_sha256': INDEX_SHA256, 'pool_sha256': POOL_SHA256}
    expected = json.loads(json.dumps(expected))
    if any(pipeline.identity.get(key) != value for key,value in expected.items()):
        raise ValueError('Mask parameters/index/pool differ from the preserved policy')


def prepare_mask(experiment_id, output, *, detector=None, landmarker=None):
    if RECIPES.get(experiment_id, {}).get('input') != 'face_mask':
        raise ValueError('Only optional 1b/2b use a face mask')
    output = _new_output(experiment_id, output)
    from cnn_face_experiment.crops import ensure_detector
    from cnn_face_experiment.preprocessing import ensure_landmark_runtime, ensure_landmarker, make_preprocessing
    ensure_landmark_runtime(install=True)
    assets = output.parent/'assets'
    detector = Path(detector) if detector else assets/'face_detection_yunet_2023mar.onnx'
    landmarker = Path(landmarker) if landmarker else assets/'face_landmarker_float16_v1.task'
    ensure_detector(detector, download=True); ensure_landmarker(landmarker, download=True)
    pipeline = make_preprocessing('face_mask', detector, output.parent/'mask_metadata.sqlite',
                                  INDEX_SHA256, POOL_SHA256, landmark_weights=landmarker)
    _check_mask(pipeline)
    return pipeline


def validate_original_encoder(checkpoint, expected_sha256=None):
    """Original trained full-frame CNN, never ImageNet/1a/F/masked substitution."""
    import torch
    from cnn_experiment.checkpoints import digest
    from cnn_experiment.data import PROTOCOL
    if checkpoint is None:
        raise FileNotFoundError('Optional 3a needs an explicit ORIGINAL trained full-frame ResNet50 best.pt')
    checkpoint = Path(checkpoint)
    if checkpoint.name != 'best.pt' or not checkpoint.is_file():
        raise FileNotFoundError('Attach the original trained ResNet50 best.pt; no automatic substitute')
    sha = digest(checkpoint)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError('Original encoder SHA256 differs from its recorded identity')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = state.get('config', {})
    if (state.get('protocol') != PROTOCOL or cfg.get('model') != 'resnet50' or
        cfg.get('checkpoint_selection', 'video_roc_auc') not in ('frame_pr_auc','video_roc_auc') or
        state.get('input_pipeline') is not None or not state.get('history') or
        state.get('epoch', -1) < 0 or state.get('best_epoch') != state['epoch']+1):
        raise ValueError('3a requires the original selected full-frame PR/video-ROC ResNet50, not a new/masked model')
    return {'checkpoint_sha256': sha, 'best_epoch': state['best_epoch'],
            'selection': cfg.get('checkpoint_selection', 'video_roc_auc'), 'frozen_encoder': True}


def resolve_clips(path=None):
    if path is None:
        from cnn_experiment.workflow import input_files, unique_path
        path = unique_path(input_files('clip_manifest.json'), 'ORIGINAL_CLIP_MANIFEST')
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Reattach/restore the original clip_manifest.json; no automatic timeline rebuild')
    return path


def preflight(experiment_id, index_path, pool_path, data_root, weights_root, output, *,
              mask_pipeline=None, clip_manifest=None, original_cnn_best=None, original_cnn_sha256=None):
    from cnn_experiment.data import write_json
    from cnn_experiment.checkpoints import digest
    from cnn_experiment.workflow import diagnostic_stage
    cfg = configuration(experiment_id); output = _new_output(experiment_id, output)
    folder = output.parent/'preflight'; folder.mkdir(parents=True, exist_ok=False)
    mode = RECIPES[experiment_id]['input']
    with diagnostic_stage(folder, 'preflight'):
        if experiment_id == '3a':
            if mask_pipeline is not None: raise ValueError('3a uses original full-frame clips only')
            from cnn_experiment.temporal import ClipIndex, ClipPool, preflight_temporal
            from cnn_experiment.pool import FixedPoolSampler
            if digest(index_path) != INDEX_SHA256: raise ValueError('Original DEV index required')
            clip_manifest = resolve_clips(clip_manifest)
            raw = json.loads(clip_manifest.read_text(encoding='utf-8'))
            if raw.get('dev_index_sha256') != INDEX_SHA256:
                raise ValueError('Clip manifest is not linked to the exact original full DEV schedule')
            index = ClipIndex(clip_manifest, data_root)
            try:
                sampler = FixedPoolSampler(ClipPool(index), cfg.clip_per_class, cfg.seed)
                sampling = [sampler.select(epoch)[1] for epoch in range(cfg.epochs)]
                inputs = {'index_sha256': INDEX_SHA256, 'clip_sha256': digest(clip_manifest), 'schedule': index.report}
            finally: index.close()
            encoder = validate_original_encoder(original_cnn_best, original_cnn_sha256)
            smoke = preflight_temporal(cfg, clip_manifest, data_root, original_cnn_best, folder/'temporal_smoke.json')
            if smoke.get('optimizer_steps') != 0: raise ValueError('Preflight must make zero updates')
            pipeline_identity = None
        else:
            from cnn_experiment.workflow import verify_epoch_sampling, diagnose_training_start, probe_gpu
            inputs = _cnn_inputs(index_path, pool_path, data_root)
            sampling = verify_epoch_sampling(index_path, data_root, cfg, pool_path=pool_path)
            encoder = None
            if mode == 'face_mask':
                _check_mask(mask_pipeline)
                from cnn_experiment.pool import DevSchedule
                from cnn_face_experiment.experiments import bounded_preflight
                dev = DevSchedule(index_path, data_root)
                try: records = dev.records('dev', 'frame')[:4]
                finally: dev.close()
                smoke = bounded_preflight({experiment_id:cfg}, records, data_root, weights_root,
                    mask_pipeline, folder/'masked_smoke', face_inputs={experiment_id:True})
                if smoke[experiment_id]['optimizer_steps'] != 0: raise ValueError('Preflight must make zero updates')
                pipeline_identity = mask_pipeline.identity
            else:
                if mask_pipeline is not None: raise ValueError('Full-frame ID cannot receive a face pipeline')
                loading = diagnose_training_start(cfg,index_path,data_root,weights_root,folder/'loader_smoke',max_batches=1,pool_path=pool_path)
                smoke = probe_gpu(index_path,data_root,weights_root,cfg,pool_path=pool_path)
                smoke['loading_diagnostics'] = str(loading); pipeline_identity = None
                if smoke['optimizer_steps'] != 0: raise ValueError('Preflight must make zero updates')
        proof = {'experiment_id':experiment_id,'config':cfg.as_dict(),'input':mode,'inputs':inputs,
                 'output':str(output),'data_root':str(Path(data_root).resolve()),
                 'weights_root':str(Path(weights_root).resolve()),'input_pipeline':pipeline_identity,
                 'clip_manifest':str(Path(clip_manifest).resolve()) if experiment_id=='3a' else None,
                 'encoder':encoder,'sampling':sampling,'smoke':smoke,'optimizer_steps':0,'test_used':False}
        write_json(folder/'preflight.json',proof)
        return proof


def _saved_identity(experiment_id, source):
    from cnn_experiment.reporting import SavedRun
    saved = SavedRun(source)
    if saved.single_history: raise ValueError('New run identity needs config and experiment_plan.json, not a lone history')
    if saved.archive:
        with zipfile.ZipFile(source) as z:
            name = saved.prefix+'experiment_plan.json'
            if z.getinfo(name).file_size > 1024**2: raise ValueError('Oversized plan')
            plan = json.loads(z.read(name))
    else: plan = json.loads((Path(source)/'experiment_plan.json').read_text(encoding='utf-8'))
    cfg = saved.json('config.json') or {}; expected = configuration(experiment_id)
    if (plan.get('experiment_id') != experiment_id or plan.get('preprocessing_mode') != RECIPES[experiment_id]['input'] or
        any(cfg.get(key) != expected.as_dict()[key] for key in ('model','head_lr','backbone_lr','epochs','patience','checkpoint_selection'))):
        raise ValueError('Saved identity/config differs; never rename historical C/D/E/F results')
    pipeline = saved.json('input_pipeline.json')
    if RECIPES[experiment_id]['input'] == 'face_mask':
        if not pipeline or pipeline.get('kind') != 'face_mask': raise ValueError('Masked run identity missing')
    elif pipeline is not None: raise ValueError('Full-frame/clip identity cannot use a mask')
    return saved


def train_planned(experiment_id,index_path,pool_path,data_root,weights_root,output,*,proof,
                  mask_pipeline=None,original_cnn_best=None,resume_from=None,provenance=None):
    from cnn_experiment.training import train
    from cnn_experiment.data import write_json
    from cnn_experiment.checkpoints import digest
    cfg = configuration(experiment_id); output = _new_output(experiment_id,output)
    mode = RECIPES[experiment_id]['input']
    if (proof.get('experiment_id') != experiment_id or proof.get('config') != cfg.as_dict() or
        proof.get('output') != str(output) or proof.get('optimizer_steps') != 0 or
        proof.get('data_root') != str(Path(data_root).resolve()) or
        proof.get('weights_root') != str(Path(weights_root).resolve()) or proof.get('input') != mode):
        raise ValueError('Complete this exact ID/config/output preflight first')
    if mode == 'face_mask': _check_mask(mask_pipeline)
    elif mask_pipeline is not None: raise ValueError('Full-frame IDs cannot use a face pipeline')
    if proof.get('input_pipeline') != (mask_pipeline.identity if mask_pipeline else None):
        raise ValueError('Preprocessing policy differs from preflight')
    if experiment_id == '3a':
        manifest = Path(proof['clip_manifest'])
        if digest(manifest) != proof['inputs']['clip_sha256']: raise ValueError('Clip manifest changed')
        if validate_original_encoder(original_cnn_best) != proof['encoder']: raise ValueError('Original encoder changed')
    elif _cnn_inputs(index_path,pool_path,data_root) != proof['inputs']:
        raise ValueError('Input metadata changed after preflight')
    if resume_from is not None: _saved_identity(experiment_id,Path(resume_from).parent)
    plan = declared_plan()[experiment_id] | {'preprocessing_mode':mode,'input_pipeline':proof['input_pipeline'],
        'inputs':proof['inputs'],'original_encoder':proof['encoder'],'source':provenance}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.parent/'experiment_plans.json',{cfg.model:plan})
    write_json(output.parent/'git_provenance.json',provenance)
    if experiment_id == '3a':
        from cnn_experiment.temporal import train_temporal
        result = train_temporal(cfg,manifest,data_root,original_cnn_best,output,resume_from)
    else:
        result = train(cfg,index_path,data_root,weights_root,output,resume_from,pool_path=pool_path,input_pipeline=mask_pipeline)
    display_result(experiment_id,result)
    print('Independent checkpoint folder:',result,'| ZIP folder:',Path(result).parent/'downloads',flush=True)
    return result


def display_result(experiment_id,source):
    from cnn_experiment.reporting import display_completed_run
    _saved_identity(experiment_id,source)
    spec = RECIPES[experiment_id]
    print(experiment_id,'|',spec['model'],'|',spec['input'],'| selected by',declared_plan()[experiment_id]['checkpoint_monitor'])
    return display_completed_run(source,name=experiment_id+' '+spec['model']+' '+spec['input'],
                                 kind='clip' if experiment_id=='3a' else 'frame')


def comparison_rows(runs):
    from cnn_experiment.reporting import epoch_table
    rows = []
    for name in REVIEW_ORDER:
        row = {'ID':name,'Model':RECIPES[name]['model'],'Input':RECIPES[name]['input'],
               'Level':'Clip' if name=='3a' else 'Frame','Selection':declared_plan()[name]['checkpoint_monitor'],
               'Status':'Pending','Best Epoch':None}
        keys = ['PR-AUC','ROC-AUC','Precision','Recall','F1','F2']
        row.update({level+' '+key:None for level in ('Frame/Clip','Video') for key in keys})
        row.update({'DEV Loss':None,'Balanced DEV Loss':None})
        if runs.get(name) is not None:
            _saved_identity(name,runs[name]);table,best,_=epoch_table(runs[name],kind=row['Level'].lower())
            selected=next((r for r in table if r['Epoch']==best),{})
            row.update(Status='Completed',**{'Best Epoch':best})
            for key in keys: row['Frame/Clip '+key]=selected.get(row['Level']+' '+key);row['Video '+key]=selected.get('Video '+key)
            for key in ('DEV Loss','Balanced DEV Loss'):row[key]=selected.get(key)
        rows.append(row)
    return rows
