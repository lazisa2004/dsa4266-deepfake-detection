"""New IDs, preserved old recipes and synthetic engine checks; no real training."""
import ast
import contextlib
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn

from cnn_experiment.config import experiment_config, checkpoint_score, selection_label
from cnn_experiment.temporal import TemporalConfig, ResNetLSTM
from cnn_experiment.reporting import epoch_table, GRAPH_ORDER
from model_family_plan.plan import RECIPES, REVIEW_ORDER, configuration, declared_plan, duplicate_audit
from model_family_plan import runner


class ModelFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_recipes_and_only_approved_config_changes(self):
        expected = {'1a':('resnet50','full_frame',1e-4,4.5e-6),
                    '1b':('resnet50','face_mask',1e-4,4.5e-6),
                    '2a':('efficientnet_b0','full_frame',2e-4,1.5e-5),
                    '2b':('efficientnet_b0','face_mask',2e-4,1.5e-5),
                    '3a':('resnet50_lstm','full_frame_clips',3e-5,None)}
        self.assertEqual(set(RECIPES),set(expected))
        for name, values in expected.items():
            recipe, cfg = RECIPES[name], configuration(name)
            self.assertEqual(tuple(recipe[k] for k in ('model','input','head_lr','backbone_lr')),values)
            base = TemporalConfig() if name=='3a' else experiment_config(cfg.model,'fixed20k_lr15e6')
            allowed = {'head_lr','epochs','checkpoint_selection'} if name=='3a' else {'head_lr','backbone_lr','epochs','checkpoint_selection'}
            changed = {k for k,v in cfg.as_dict().items() if v!=base.as_dict()[k]}
            self.assertLessEqual(changed,allowed)
            self.assertEqual((cfg.epochs,cfg.patience,cfg.threshold,cfg.topk_fraction,cfg.seed),(15,3,.5,.05,4266))
            self.assertEqual(cfg.checkpoint_selection,'clip_roc_auc' if name=='3a' else 'frame_roc_auc')
            self.assertEqual(cfg.head_lr,values[2])
            if name!='3a':
                self.assertEqual((cfg.backbone_lr,cfg.warmup_epochs,cfg.batch_size,cfg.dropout),(values[3],2,32,.3))
            else:
                self.assertEqual((cfg.batch_size,cfg.clip_per_class,cfg.warmup_epochs,cfg.hidden_size,cfg.lstm_layers),(8,1500,0,128,1))
                self.assertFalse(cfg.finetune_final_stage)
        self.assertEqual(TemporalConfig().checkpoint_selection,'clip_pr_auc')
        self.assertEqual(declared_plan()['3a']['scheduler_monitor'],'DEV clip PR-AUC (original)')
        for name in ('1a','1b','2a','2b'):
            self.assertIn('video ROC-AUC',declared_plan()[name]['scheduler_monitor'])

    def test_historical_c_duplicate_distinguished_and_other_recipes_preserved(self):
        rows=duplicate_audit()
        self.assertFalse(any(r['identical_configuration'] for r in rows))
        c=next(r for r in rows if r['new_id']=='2a' and r['historical_id']=='efficientnet_b0_head_lr_2e4')
        self.assertEqual(c['differences'],{'epochs':{'historical':10,'new':15},
            'checkpoint_selection':{'historical':'frame_pr_auc','new':'frame_roc_auc'}})
        from cnn_face_experiment.roc_experiments import roc_experiment_configurations
        self.assertTrue(all((c.epochs,c.patience)==(15,5) for c in roc_experiment_configurations().values()))
        from experiment_f.runner import experiment_f_config
        self.assertEqual((experiment_f_config().head_lr,experiment_f_config().patience),(5e-5,5))

    def test_order_guards_independent_session_and_no_historical_outputs(self):
        self.assertEqual(REVIEW_ORDER,('1a','1b','3a','2a','2b'))
        runner.require_first_result('1a',{})
        for name in REVIEW_ORDER[1:]:
            with self.assertRaises(RuntimeError):runner.require_first_result(name,{})
            runner.require_first_result(name,{'1a':'saved'})
        with self.assertRaises(FileExistsError):runner.require_first_result('1a',{'1a':'saved'})
        with tempfile.TemporaryDirectory() as folder:
            sessions=[runner.new_session(folder) for _ in range(2)]
            self.assertNotEqual(*sessions)
            paths=[runner.output_path(s,n) for s in sessions for n in RECIPES]
            self.assertEqual(len(set(paths)),10)
            for path in paths: self.assertEqual(runner._new_output(path.parent.name,path),path.resolve())
            paths[0].mkdir(parents=True)
            with self.assertRaises(FileExistsError):runner._new_output('1a',paths[0])
            with self.assertRaises(ValueError):runner._new_output('2a',Path(folder)/'historical_C')

    def test_clip_roc_score_not_pr_video_and_old_selectors_unchanged(self):
        report={'unit':{'roc_auc':.85,'pr_auc':.2},'video':{'roc_auc':.6}}
        self.assertEqual(checkpoint_score(report,'clip_roc_auc'),.85)
        self.assertEqual(checkpoint_score(report,'clip_pr_auc'),.2)
        self.assertEqual(checkpoint_score(report,'video_roc_auc'),.6)
        self.assertEqual(selection_label('clip_roc_auc'),'DEV clip ROC-AUC')
        for value in (None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):checkpoint_score({'unit':{'roc_auc':value}},'clip_roc_auc')

    def test_unchanged_mask_policy_and_full_frame_rejects_mask_creation(self):
        from cnn_face_experiment.preprocessing import MaskConfig
        from cnn_face_experiment.crops import CropConfig
        identity=json.loads(json.dumps({'kind':'face_mask','mask':asdict(MaskConfig()),'crop':asdict(CropConfig()),
            'index_sha256':runner.INDEX_SHA256,'pool_sha256':runner.POOL_SHA256}))
        runner._check_mask(SimpleNamespace(identity=identity))
        identity['mask']['feather_pixels']=999
        with self.assertRaises(ValueError):runner._check_mask(SimpleNamespace(identity=identity))
        for name in ('1a','2a','3a'):
            with self.assertRaises(ValueError):runner.prepare_mask(name,'unused')

    def test_original_encoder_identity_rejects_new_masked_or_wrong_hash(self):
        from cnn_experiment.data import PROTOCOL
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'best.pt'
            state={'protocol':PROTOCOL,'config':{'model':'resnet50','checkpoint_selection':'frame_pr_auc'},
                   'epoch':8,'best_epoch':9,'history':[{'epoch':9}]}
            torch.save(state,path)
            checked=runner.validate_original_encoder(path)
            self.assertEqual(checked['best_epoch'],9);self.assertTrue(checked['frozen_encoder'])
            with self.assertRaises(ValueError):runner.validate_original_encoder(path,'wrong')
            for changes in ({'config':{'model':'resnet50','checkpoint_selection':'frame_roc_auc'}},
                            {'input_pipeline':{'kind':'face_mask'}},{'history':[]}):
                torch.save(state|changes,path)
                with self.assertRaises(ValueError):runner.validate_original_encoder(path)
        with self.assertRaises(FileNotFoundError):runner.validate_original_encoder(None)

    def test_frozen_temporal_parameters_and_architecture_unchanged(self):
        class Encoder(nn.Module):
            def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(1))
            def forward(self,x):return x.mean((1,2,3))[:,None].expand(-1,2048)*self.weight
        old=ResNetLSTM(Encoder(),TemporalConfig());new=ResNetLSTM(Encoder(),configuration('3a'))
        self.assertEqual({k:tuple(v.shape) for k,v in old.state_dict().items()},
                         {k:tuple(v.shape) for k,v in new.state_dict().items()})
        new.train();self.assertFalse(new.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in new.encoder.parameters()))
        new.load_state_dict(old.state_dict(),strict=True)

    def test_wrapper_passes_original_loader_no_mask_and_fresh_imagenet_weights(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner,'_cnn_inputs',return_value={'fixed':'original'}))
            stack.enter_context(patch.object(runner,'display_result'))
            train=stack.enter_context(patch('cnn_experiment.training.train',side_effect=lambda *a,**k:a[4]))
            for name in ('1a','2a'):
                out=runner.output_path(folder,name).resolve();cfg=configuration(name)
                proof={'experiment_id':name,'config':cfg.as_dict(),'output':str(out),'optimizer_steps':0,
                       'data_root':str(Path('videos').resolve()),'weights_root':str(Path('imagenet').resolve()),
                       'input':RECIPES[name]['input'],'input_pipeline':None,'inputs':{'fixed':'original'},'encoder':None}
                runner.train_planned(name,'index','pool','videos','imagenet',out,proof=proof)
                call=train.call_args
                self.assertEqual(call.args[3],'imagenet');self.assertIsNone(call.args[5])
                self.assertIsNone(call.kwargs['input_pipeline']);self.assertEqual(call.kwargs['pool_path'],'pool')
                plan=json.loads((out.parent/'experiment_plans.json').read_text())[cfg.model]
                self.assertEqual((plan['experiment_id'],plan['preprocessing_mode']),(name,'full_frame'))
                self.assertEqual(plan['initialization'],'independent ImageNet')
                with self.assertRaises(FileExistsError):
                    runner.train_planned(name,'index','pool','videos','imagenet',out,proof=proof)
            self.assertEqual(train.call_count,2)

    def test_saved_run_identity_replay_pending_and_no_inference(self):
        from tests.test_reporting import fixture
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'run';fixture(run)
            (run/'config.json').write_text(json.dumps(configuration('1a').as_dict()))
            plan={'experiment_id':'1a','preprocessing_mode':'full_frame'}
            (run/'experiment_plan.json').write_text(json.dumps(plan))
            runner._saved_identity('1a',run)
            with self.assertRaises(ValueError):runner._saved_identity('2a',run)
            rows=runner.comparison_rows({'1a':run})
            self.assertEqual(rows[0]['Status'],'Completed');self.assertEqual(rows[0]['Best Epoch'],2)
            self.assertEqual(sum(r['Status']=='Pending' for r in rows),4)
            self.assertEqual(next(r['Level'] for r in rows if r['ID']=='3a'),'Clip')
            self.assertIsNotNone(rows[0]['Video F2'])
            archive=Path(folder)/'saved.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for p in run.iterdir():z.write(p,'saved/'+p.name)
            runner._saved_identity('1a',archive)

    def test_notebook_has_independent_cells_no_training_chain_or_test_call(self):
        path=Path(__file__).resolve().parents[1]/'notebooks/experiment1_face_cnn.ipynb'
        if not path.is_file():self.skipTest('Local-only notebook')
        nb=json.loads(path.read_text(encoding='utf-8'));cells=nb['cells']
        self.assertGreaterEqual(len(cells),50)
        added=[c for c in cells[:50] if c['id'].startswith('family-')]
        self.assertEqual(len(added),19)
        for name in RECIPES:
            cell=next(c for c in added if c['id']==f'family-{name}-train')
            code=''.join(cell['source']);tree=ast.parse(code)
            self.assertEqual(code.count('train_planned('),1)
            self.assertIn(repr(name),code)
            self.assertFalse(any(isinstance(n,(ast.For,ast.While,ast.AsyncFor)) for n in ast.walk(tree)))
            self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
        for cell in added:
            if cell['cell_type']=='code':
                tree=ast.parse(''.join(cell['source']))
                self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
                                     n.func.id in ('evaluate_test','run_test','train') for n in ast.walk(tree)))

    def test_mocked_temporal_loop_roc_best_patience_three_pr_scheduler_and_zip(self):
        from cnn_experiment import training
        from cnn_experiment.evaluation import binary_metrics,class_loss_metrics
        from tests.test_temporal import manifest_fixture
        with tempfile.TemporaryDirectory() as folder,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);manifest=root/'clips.json';index=manifest_fixture(manifest)
            cfg=replace(configuration('3a'),hidden_size=4,clip_per_class=1,amp=False,heartbeat_seconds=0)
            records=index.records('dev','clip');labels=[r['label'] for r in records]
            rng=np.random.default_rng(4266)
            choices=[(scores,binary_metrics(labels,scores)) for scores in rng.uniform(.01,.99,(256,len(labels))).tolist()]
            low,high=next((a,b) for a in choices for b in choices
                          if b[1]['roc_auc']>a[1]['roc_auc']+.08 and b[1]['pr_auc']<a[1]['pr_auc']-.02)
            poor=min(choices,key=lambda a:a[1]['roc_auc'])
            scheduled=[low,high]+[poor]*13
            def epoch(*args,**kw):
                scores,metrics=scheduled[kw['epoch']];metrics=copy.deepcopy(metrics)
                counts={y:labels.count(y) for y in (0,1)}
                metrics.update(class_loss_metrics({y:counts[y]*.2 for y in (0,1)},counts));metrics['loss']=.2
                return metrics,[{'record_index':i,'target':labels[i],'probability':p} for i,p in enumerate(scores)],.01
            stack.enter_context(patch('cnn_experiment.temporal.ClipIndex',return_value=index))
            stack.enter_context(patch('cnn_experiment.temporal.trained_resnet',side_effect=lambda c,p:(ResNetLSTM(nn.Identity(),c),{'original':'synthetic'})))
            stack.enter_context(patch.object(nn.Module,'cuda',lambda self:self))
            stack.enter_context(patch.object(training,'make_loader',return_value=object()))
            stack.enter_context(patch.object(training,'run_epoch',side_effect=epoch))
            boundary=root/'interrupted_boundary';boundary.mkdir()
            original_commit=training.commit_epoch
            def commit(*args,**kwargs):
                result=original_commit(*args,**kwargs)
                if args[2]['epoch']==2:
                    for name in ('best.pt','last.pt'):shutil.copy2(Path(args[0])/name,boundary/name)
                return result
            stack.enter_context(patch.object(training,'commit_epoch',side_effect=commit))
            monitored=[];original_step=torch.optim.lr_scheduler.ReduceLROnPlateau.step
            def scheduler_step(self,metric,*args,**kwargs):
                monitored.append(metric);return original_step(self,metric,*args,**kwargs)
            stack.enter_context(patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau,'step',scheduler_step))
            for name,value in (('get_device_name','mock GPU'),('reset_peak_memory_stats',None),('max_memory_allocated',0),('empty_cache',None)):
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            out=root/'run';out.mkdir()
            with training.RunProgress(out,cfg.model,0) as progress:
                training._train(cfg,manifest,root,root/'original_best.pt',out,None,progress,None)
            last=torch.load(out/'last.pt',weights_only=True);best=torch.load(out/'best.pt',weights_only=True)
            self.assertEqual((last['epoch']+1,last['stale'],last['best_epoch']),(5,3,2))
            self.assertEqual(best['epoch']+1,2);self.assertEqual(len(last['history']),5)
            from cnn_experiment.checkpoints import load_resume_bundle, restore_training_state, digest
            from cnn_experiment.pool import FixedPoolSampler
            from cnn_experiment.temporal import ClipPool, TEMPORAL_PROTOCOL
            with self.assertRaisesRegex(ValueError,'epoch cap or early stopping'):
                load_resume_bundle(out/'last.pt',cfg,digest(manifest),protocol=TEMPORAL_PROTOCOL)
            bundle=load_resume_bundle(boundary/'last.pt',cfg,digest(manifest),protocol=TEMPORAL_PROTOCOL)
            restored_model=ResNetLSTM(nn.Identity(),cfg)
            restored_optimizer=torch.optim.AdamW([{'name':'temporal_head',
                'params':[p for p in restored_model.parameters() if p.requires_grad],
                'lr':cfg.head_lr}],weight_decay=cfg.weight_decay)
            restored_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer,mode='max',factor=.5,patience=cfg.scheduler_patience)
            restored_scaler=torch.amp.GradScaler('cpu',enabled=False)
            restored_sampler=FixedPoolSampler(ClipPool(index),cfg.clip_per_class,cfg.seed)
            restore_training_state(bundle,restored_model,restored_optimizer,restored_scheduler,restored_scaler,restored_sampler)
            self.assertEqual(bundle['report']['checkpoint_selection'],'clip_roc_auc')
            self.assertEqual(restored_scheduler.state_dict(),bundle['state']['scheduler'])
            self.assertEqual(restored_sampler.state_dict(),bundle['state']['sampler'])
            self.assertTrue(all(torch.equal(v,restored_model.state_dict()[k]) for k,v in bundle['state']['model'].items()))
            self.assertLess(high[1]['pr_auc'],low[1]['pr_auc'])
            np.testing.assert_allclose(monitored,[p[1]['pr_auc'] for p in scheduled[:5]])
            rows,selected,kind=epoch_table(out)
            self.assertEqual((len(rows),len(rows[0]),selected,kind),(5,19,2,'clip'))
            self.assertTrue(all(r['Video F2'] is not None and r['Clip F2'] is not None for r in rows))
            html=(out/'epoch_metrics.html').read_text();self.assertEqual(html.count('background:#fff0b3'),1)
            self.assertIn('DEV clip ROC-AUC',html)
            archives=list((root/'downloads').glob('*.zip'))
            self.assertTrue(any('final' not in p.name for p in archives))
            archive=next(p for p in archives if 'final' in p.name)
            with zipfile.ZipFile(archive) as z:
                self.assertTrue(set(('best.pt','last.pt','history.csv','epoch_metrics.csv','epoch_metrics.json',*GRAPH_ORDER))<=set(z.namelist()))
                self.assertEqual(len(json.loads(z.read('epoch_metrics.json'))['rows']),5)


if __name__=='__main__':unittest.main()
