"""D/E fixtures: mocked epoch computation, no CNN training or held-out data."""
import ast
import contextlib
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from cnn_experiment.config import checkpoint_score, selection_label
from cnn_experiment.evaluation import binary_metrics, class_loss_metrics, compare_runs, plot_history
from cnn_experiment.reporting import GRAPH_ORDER, epoch_table, export_epoch_report, table_html
from cnn_experiment.pool import FixedPoolSampler
from cnn_experiment.backup import restore_local_zip
from cnn_face_experiment.experiments import experiment_configurations, output_paths
from cnn_face_experiment.roc_experiments import (
    ROC_EXPERIMENTS, roc_experiment_configurations, roc_output_paths,
    train_roc_experiment, validate_roc_pipeline, display_roc_completed_run)
from tests import test_fixed_pool as pool_fixtures
from tests import test_reporting as reporting_fixtures
from tests import test_training_state as resume_fixtures


class RocExperimentTests(unittest.TestCase):
    def test_exact_two_configs_only_approved_fields_differ_and_old_recipes_unchanged(self):
        old = experiment_configurations()
        self.assertEqual(len(old), 3)
        for cfg in old.values():
            self.assertEqual((cfg.epochs, cfg.patience, cfg.checkpoint_selection), (10, 3, 'frame_pr_auc'))
        new = roc_experiment_configurations()
        self.assertEqual(len(new), 2)
        for name, recipe in ROC_EXPERIMENTS.items():
            cfg, base = new[name], old[recipe['baseline']]
            allowed = {'epochs','patience','checkpoint_selection',
                       'backbone_lr' if recipe['letter']=='D' else 'head_lr'}
            self.assertEqual({k for k,v in cfg.as_dict().items() if v!=base.as_dict()[k]}, allowed)
            self.assertEqual((cfg.head_lr,cfg.backbone_lr,cfg.epochs,cfg.patience),
                             (recipe['head_lr'],recipe['backbone_lr'],15,5))
            self.assertEqual(cfg.checkpoint_selection,'frame_roc_auc')
            self.assertEqual((cfg.threshold,cfg.topk_fraction,cfg.seed,cfg.train_per_class),(.5,.05,4266,4000))
        self.assertFalse(set(output_paths('session').values()) & set(roc_output_paths('session').values()))

    def test_score_uses_frame_roc_not_pr_or_video_and_rejects_undefined(self):
        a={'unit':{'roc_auc':.9,'pr_auc':.2},'video':{'roc_auc':.3}}
        b={'unit':{'roc_auc':.7,'pr_auc':.8},'video':{'roc_auc':.95}}
        self.assertGreater(checkpoint_score(a,'frame_roc_auc'),checkpoint_score(b,'frame_roc_auc'))
        for old in ('frame_pr_auc','video_roc_auc'):
            self.assertLess(checkpoint_score(a,old),checkpoint_score(b,old))
        self.assertEqual(selection_label('frame_roc_auc'),'DEV frame ROC-AUC')
        for invalid in (None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                checkpoint_score({'unit':{'roc_auc':invalid}},'frame_roc_auc')

    def test_independent_imagenet_calls_and_saved_selection_plan_no_old_outputs(self):
        pipeline=SimpleNamespace(identity={'kind':'face_mask','index_sha256':'index','pool_sha256':'pool'})
        with tempfile.TemporaryDirectory() as temp, \
             patch('cnn_face_experiment.roc_experiments.existing_pool',return_value='pool'), \
             patch('cnn_face_experiment.roc_experiments.digest',side_effect=lambda p:'index' if str(p)=='index' else 'pool'), \
             patch('cnn_experiment.training.train',side_effect=lambda *a,**k:a[4]) as train:
            for name,cfg in roc_experiment_configurations().items():
                out=roc_output_paths(temp)[name]
                train_roc_experiment(name,'index','pool','videos','original_imagenet',out,pipeline)
                plan=json.loads((out.parent/'experiment_plans.json').read_text())[cfg.model]
                self.assertEqual(plan['checkpoint_selection'],'frame_roc_auc')
                self.assertEqual(plan['scheduler_monitor'],'DEV video ROC-AUC (unchanged)')
            self.assertEqual(train.call_count,2)
            for call in train.call_args_list:
                self.assertEqual(call.args[3],'original_imagenet');self.assertIsNone(call.args[5])
                self.assertIs(call.kwargs['input_pipeline'],pipeline)
            existing=Path(temp)/'existing';existing.mkdir();(existing/'history.csv').write_bytes(b'KEEP')
            with self.assertRaises(FileExistsError):
                train_roc_experiment(next(iter(ROC_EXPERIMENTS)),'index','pool','videos','weights',existing,pipeline)
            self.assertEqual((existing/'history.csv').read_bytes(),b'KEEP')
        with self.assertRaises(ValueError): validate_roc_pipeline(SimpleNamespace(identity={'kind':'face_crop'}))

    def test_resume_preserves_roc_state_and_rejects_switching_a_previous_selector(self):
        from cnn_experiment.checkpoints import load_resume_bundle
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,_,_=resume_fixtures.ResumeTests().fixture(root)
            config=replace(config,checkpoint_selection='frame_roc_auc')
            for name in ('last.pt','best.pt'):
                state=torch.load(root/name,weights_only=True)
                state['config']['checkpoint_selection']='frame_roc_auc';state['best']=.8
                for row in state['history']:
                    row['dev_video']={'roc_auc':.6};row['dev']['roc_auc']=.8 if row['epoch']==2 else .7
                torch.save(state,root/name)
            bundle=load_resume_bundle(root/'last.pt',config,'fixture-index')
            self.assertEqual(bundle['report']['checkpoint_selection'],'frame_roc_auc')
            self.assertEqual(bundle['report']['best_selection_score'],.8)
            self.assertEqual(bundle['report']['best_dev_video_roc_auc'],.6)
            with self.assertRaisesRegex(ValueError,'checkpoint_selection'):
                load_resume_bundle(root/'last.pt',replace(config,checkpoint_selection='frame_pr_auc'),'fixture-index')

    def test_fifteen_epoch_rotation_identical_balance_and_no_within_epoch_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            pool=pool_fixtures.pool_fixture(Path(temp)/'pool.json')
            left,right=FixedPoolSampler(pool),FixedPoolSampler(pool)
            for epoch in range(15):
                ids,report=left.select(epoch);other,_=right.select(epoch)
                self.assertEqual(ids,other);self.assertEqual(len(set(ids)),8000)
                self.assertEqual(sum(pool.frames[i]['label'] for i in ids),4000)
                self.assertEqual(report['duplicate_rate'],0)

    def test_table_columns_yellow_best_history_and_report_zip_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';reporting_fixtures.fixture(run)
            config=json.loads((run/'config.json').read_text());config['checkpoint_selection']='frame_roc_auc'
            (run/'config.json').write_text(json.dumps(config))
            summary=json.loads((run/'training_summary.json').read_text());summary['checkpoint_selection']='frame_roc_auc'
            (run/'training_summary.json').write_text(json.dumps(summary))
            baseline,selected,_=epoch_table(run)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
            export_epoch_report(run,run/'epoch_report',checkpoint_selection='frame_roc_auc')
            exported=json.loads((run/'epoch_report/epoch_metrics.json').read_text())
            self.assertEqual(exported['rows'],baseline);self.assertEqual(exported['selected_epoch'],selected)
            self.assertEqual(len(baseline[0]),19);self.assertEqual(len(baseline),3)
            html=(run/'epoch_report/epoch_metrics.html').read_text()
            self.assertEqual(html.count('background:#fff0b3'),1)
            self.assertIn('DEV frame ROC-AUC',html)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()})
            archive=root/'saved.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for p in run.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(run))
            with patch('cnn_experiment.reporting.display_completed_run',return_value=baseline) as display:
                self.assertEqual(display_roc_completed_run(archive,name='D'),baseline)
                self.assertEqual(display.call_args.kwargs['kind'],'frame')

    def test_roc_comparison_does_not_accidentally_rank_by_pr(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runs={}
            for name,roc,pr in (('D',.9,.2),('E',.7,.8)):
                run=root/name;run.mkdir();runs[name]=run
                (run/'training_summary.json').write_text(json.dumps({'status':'trained','best_epoch':1,
                    'checkpoint_selection':'frame_roc_auc','best_balanced_dev_loss':.4}))
                (run/'best_dev_metrics.json').write_text(json.dumps({'unit':{'roc_auc':roc,'pr_auc':pr},'video':{'roc_auc':1-roc}}))
            compare_runs(runs,root/'comparison',selection_metric='frame_roc_auc')
            self.assertEqual(json.loads((root/'comparison/comparison.json').read_text())['selected_cnn'],'D')

    def test_six_panel_graphs_keep_legacy_layout_and_add_all_metrics_and_best_marker(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);reporting_fixtures.fixture(root)
            rows=[json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
            for row in rows:row['train']['roc_auc']=.8
            with patch('matplotlib.pyplot.close'):
                plot_history(root,rows,best_epoch=2,selection_metric='frame_roc_auc')
                fig=plt.gcf();self.assertEqual(len(fig.axes),6)
                for ax in fig.axes:
                    markers=[line for line in ax.lines if line.get_label()=='Selected epoch 2']
                    self.assertEqual(len(markers),1);np.testing.assert_array_equal(markers[0].get_xdata(),[2,2])
                self.assertIn('DEV video PR-AUC',[line.get_label() for line in fig.axes[2].lines])
                self.assertIn('DEV Video F2 at 0.5',[line.get_label() for line in fig.axes[5].lines])
                self.assertIn('DEV frame ROC-AUC',fig._suptitle.get_text())
            plt.close('all')

    def test_notebook_additions_do_not_replace_original_cells_or_comparison(self):
        path=Path(__file__).resolve().parents[1]/'notebooks/experiment1_face_cnn.ipynb'
        if not path.is_file(): self.skipTest('Notebook intentionally local-only')
        n=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual([c['id'] for c in n['cells'][:23]],['face-cnn-%02d'%i for i in range(23)])
        added='\n'.join(''.join(c['source']) for c in n['cells'][23:])
        self.assertEqual(added.count('= train_roc_experiment('),2)
        self.assertNotIn('RUNS[roc_name]',added.replace('ROC_RUNS[roc_name]',''))
        self.assertIn('ROC_RUNS = {}',added)
        self.assertEqual(GRAPH_ORDER,('training_curves.png','best_dev_confusion.png','best_dev_roc_pr.png','dev_f2_reporting_confusion.png'))

    def test_mocked_epoch_loop_selects_roc_stops_after_five_and_backups_full_table(self):
        from cnn_experiment import training
        class Model(torch.nn.Module):
            def __init__(self,*a,**k):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.phase='head'
            def cuda(self):return self
            def set_phase(self,active):self.phase='final_stage' if active else 'head'
            def trainable_report(self):return {'trainable_parameters':1}
        records=[{'id':i,'split':'dev','kind':'frame','source_file':f'dev/{i}.mp4',
                  'source_family':f'dev/{i//4}','video_label':int(i%4>=2),'label':int(i%4==3),
                  'metric_eligible':True,'timestamps_exact':['1']} for i in range(32)]
        labels=[r['label'] for r in records]
        rng=np.random.default_rng(4266)
        choices=[(scores,binary_metrics(labels,scores)) for scores in rng.uniform(.01,.99,(256,32)).tolist()]
        low,high=next((a,b) for a in choices for b in choices
                      if b[1]['roc_auc']>a[1]['roc_auc']+.08 and b[1]['pr_auc']<a[1]['pr_auc']-.03)
        poor=min(choices,key=lambda x:x[1]['roc_auc'])
        scheduled=[poor,low,high]+[poor]*12
        def epoch(model,loader,cfg,optimizer=None,scaler=None,**kw):
            scores,metrics=scheduled[kw['epoch']];metrics=copy.deepcopy(metrics)
            metrics.update(class_loss_metrics({0:3.,1:2.},{0:24,1:8}));metrics['loss']=5/32
            predictions=[{'record_index':i,'target':labels[i],'probability':p} for i,p in enumerate(scores)]
            return metrics,predictions,.01
        class Dev:
            def records(self,*a):return records
            def close(self):pass
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);pool=pool_fixtures.pool_fixture(root/'pool.json')
            index=root/'index.sqlite';index.write_bytes(b'synthetic index');out=root/'run';out.mkdir()
            cfg=replace(next(iter(roc_experiment_configurations().values())),amp=False,heartbeat_seconds=0)
            stack.enter_context(patch.object(training,'cnn_inputs',return_value=(Dev(),FixedPoolSampler(pool),pool)))
            stack.enter_context(patch.object(training,'make_loader',return_value=object()))
            stack.enter_context(patch.object(training,'StagedCNN',Model))
            stack.enter_context(patch.object(training,'make_optimizer',side_effect=lambda m,c:torch.optim.AdamW([{'params':m.parameters(),'name':'head','lr':c.head_lr}])))
            stack.enter_context(patch.object(training,'run_epoch',side_effect=epoch))
            for name,value in (('get_device_name','mock GPU'),('reset_peak_memory_stats',None),('max_memory_allocated',0),('empty_cache',None)):
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            with training.RunProgress(out,cfg.model,0) as progress:
                training._train(cfg,index,root,'.',out,None,progress,None,pool.path)
            last=torch.load(out/'last.pt',weights_only=True);best=torch.load(out/'best.pt',weights_only=True)
            self.assertEqual((last['epoch']+1,last['stale'],last['best_epoch']),(8,5,3))
            self.assertEqual(best['epoch']+1,3)
            self.assertLess(last['history'][2]['dev']['pr_auc'],last['history'][1]['dev']['pr_auc'])
            self.assertEqual(len(last['history']),8)
            rows,selected,_=epoch_table(out);self.assertEqual((len(rows),selected),(8,3))
            archive=next((root/'downloads').glob('*final*.zip'))
            with zipfile.ZipFile(archive) as z:
                for name in ('epoch_metrics.csv','epoch_metrics.json','epoch_metrics.html',*GRAPH_ORDER):
                    self.assertIn(name,z.namelist())
                table=json.loads(z.read('epoch_metrics.json'))
                self.assertEqual((len(table['rows']),table['selected_epoch']),(8,3))
            restored=restore_local_zip(archive,root/'restored')
            self.assertEqual(epoch_table(restored),epoch_table(out))


if __name__=='__main__':unittest.main()
