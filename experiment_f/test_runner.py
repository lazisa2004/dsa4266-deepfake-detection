"""Synthetic CPU checks for F. No dataset decoding, model training or live state."""
import contextlib
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from . import runner as f


def input_report():
    return {'experiment': f.EXPERIMENT_ID, 'input': 'full_frame',
            'index_sha256': f.EXPECTED_INDEX_SHA256, 'pool_sha256': f.EXPECTED_POOL_SHA256,
            'data_root': 'synthetic', 'dev': {'dev_videos': 500, 'dev_frames': 41746,
            'dev_real': 40115, 'dev_fake': 1631, 'test_used': False}}


def proof(output):
    return {'experiment': f.EXPERIMENT_ID, 'input': 'full_frame',
            'config': f.experiment_f_config().as_dict(), 'inputs': input_report(),
            'output': str(Path(output).resolve()), 'weights_root': str(Path('imagenet').resolve()),
            'optimizer_steps': 0}


def saved_fixture(path):
    from tests.test_reporting import fixture
    fixture(path)
    (path/'config.json').write_text(json.dumps(f.experiment_f_config().as_dict()))
    summary = json.loads((path/'training_summary.json').read_text())
    summary['checkpoint_selection'] = 'frame_roc_auc'
    (path/'training_summary.json').write_text(json.dumps(summary))
    (path/'experiment_plan.json').write_text(json.dumps({
        'experiment': f.EXPERIMENT_ID, 'preprocessing_mode': 'full_frame', 'input_pipeline': None}))
    (path/'last.pt').write_bytes(b'synthetic checkpoint, no model execution')


class ExperimentFTests(unittest.TestCase):
    def test_exact_config_changes_and_original_recipe_unchanged(self):
        from cnn_experiment.config import experiment_config
        base = experiment_config('resnet50', 'fixed20k_lr15e6')
        original = copy.deepcopy(base.as_dict()); cfg = f.experiment_f_config()
        self.assertEqual({k for k,v in cfg.as_dict().items() if v != original[k]},
                         {'head_lr','epochs','patience','checkpoint_selection'})
        self.assertEqual((cfg.model,cfg.head_lr,cfg.backbone_lr,cfg.epochs,cfg.patience),
                         ('resnet50',5e-5,1.5e-5,15,5))
        self.assertEqual((cfg.warmup_epochs,cfg.batch_size,cfg.dropout), (2,32,.3))
        self.assertEqual((cfg.seed,cfg.train_per_class,cfg.threshold,cfg.topk_fraction),(4266,4000,.5,.05))
        self.assertTrue(cfg.finetune_final_stage);self.assertEqual(cfg.checkpoint_selection,'frame_roc_auc')
        self.assertEqual(base.as_dict(),original)

    def test_import_has_no_face_detector_dependency_or_runtime_work(self):
        root = Path(__file__).resolve().parents[1]
        code = ('import sys;sys.path.insert(0,'+repr(str(root))+');import experiment_f.runner;'
                'assert not any(n=="cv2" or n=="mediapipe" or n.startswith("cnn_face_experiment") for n in sys.modules);'
                'assert "cnn_experiment.training" not in sys.modules;print("pure configuration import passed")')
        result = subprocess.run([sys.executable,'-B','-I','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_full_frame_loader_pixels_normalization_and_train_flip_match_original(self):
        from cnn_experiment.data import FrameReader, make_loader
        from cnn_experiment.config import experiment_config
        image = np.random.default_rng(4266).integers(0,256,(224,224,3),dtype=np.uint8)
        row = {'id':5,'split':'dev','kind':'frame','source_file':'dev/synthetic.mp4',
               'indices':[0],'timestamps_exact':['0'],'label':0}
        cfg = replace(f.experiment_f_config(),workers=0,batch_size=2)
        with patch.object(FrameReader,'read',return_value=[[image]]):
            loader = make_loader([row],'synthetic',cfg)
            self.assertIs(type(loader.dataset.reader),FrameReader)
            self.assertEqual(loader.dataset.flip_probability,0)
            tensor,_,_ = next(iter(loader))
            expected = torch.from_numpy(image).permute(2,0,1).float()/255
            mean,std = torch.tensor([.485,.456,.406])[:,None,None],torch.tensor([.229,.224,.225])[:,None,None]
            self.assertTrue(torch.equal(tensor[0],(expected-mean)/std))
            recovered = ((tensor[0]*std+mean)*255).round().to(torch.uint8).permute(1,2,0).numpy()
            np.testing.assert_array_equal(recovered,image)  # Includes every background/corner pixel.
            row = dict(row,split='train',source_file='train/synthetic.mp4')
            original = replace(experiment_config('resnet50','fixed20k_lr15e6'),workers=0,batch_size=2)
            actual = next(iter(make_loader([row],'synthetic',cfg,training=True,epoch=4)))[0]
            reference = next(iter(make_loader([row],'synthetic',original,training=True,epoch=4)))[0]
            self.assertTrue(torch.equal(actual,reference))

    def test_imagenet_initialization_head_warmup_final_stage_and_frozen_bn(self):
        from cnn_experiment.models import StagedCNN, make_optimizer
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv1=torch.nn.Linear(2,2)
                self.bn=torch.nn.BatchNorm1d(2)
                self.layer4=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
                self.fc=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(2,1))
        cfg=f.experiment_f_config()
        with patch('cnn_experiment.models.build_model',return_value=Network()) as build:
            model=StagedCNN(cfg,'imagenet',pretrained=True)
            build.assert_called_once_with('resnet50','imagenet',pretrained=True,dropout=.3)
            self.assertEqual(model.phase,'head')
            self.assertFalse(any(p.requires_grad for p in model.final_parameters()))
            model.set_phase(True);model.train()
            self.assertTrue(all(p.requires_grad for p in model.final_parameters()))
            self.assertFalse(model.network.conv1.weight.requires_grad)
            self.assertTrue(model.trainable_report()['batchnorm_running_stats_frozen'])
            optimizer=make_optimizer(model,cfg)
            self.assertEqual({g['name']:g['lr'] for g in optimizer.param_groups},
                             {'head':5e-5,'final_stage':1.5e-5})
            self.assertFalse(optimizer.state)  # No step performed.

    def test_wrong_metadata_hash_fails_without_rebuilding_or_decoding(self):
        with patch('cnn_experiment.checkpoints.digest',return_value='changed'), \
             patch('cnn_experiment.pool.FixedPool') as pool,patch('cnn_experiment.pool.DevSchedule') as dev:
            with self.assertRaisesRegex(ValueError,'unchanged approved'):
                f.validate_f_inputs('index','pool','videos')
            pool.assert_not_called();dev.assert_not_called()

    def test_preflight_uses_only_existing_full_frame_probes_and_json_serializes(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/f.EXPERIMENT_ID/'session'/'run'
            def loader(*a,**kw):
                folder=Path(a[4]);folder.mkdir(parents=True)
                (folder/'diagnostic_results.json').write_text('[{"optimizer_steps":0}]')
                return folder
            with patch.object(f,'validate_f_inputs',return_value=input_report()), \
                 patch('cnn_experiment.workflow.verify_epoch_sampling',return_value=[{'epoch':i+1} for i in range(15)]) as sampling, \
                 patch('cnn_experiment.workflow.diagnose_training_start',side_effect=loader) as loading, \
                 patch('cnn_experiment.workflow.probe_gpu',return_value={'optimizer_steps':0}) as gpu:
                receipt=f.preflight_f('index','pool','videos','imagenet',output)
                self.assertEqual(len(receipt['sampling']),15)
                self.assertEqual(receipt['optimizer_steps'],0)
                self.assertEqual(receipt['config']['head_lr'],5e-5)
                for call in (sampling.call_args,loading.call_args,gpu.call_args):
                    self.assertEqual(call.kwargs['pool_path'],'pool')
                self.assertTrue((output.parent/'preflight/preflight.json').is_file())

    def test_training_routes_none_pipeline_and_fresh_weights_into_isolated_run(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/f.EXPERIMENT_ID/'session'/'run'
            with patch.object(f,'validate_f_inputs',return_value=input_report()), \
                 patch('cnn_experiment.training.train',return_value=out) as train, \
                 patch.object(f,'display_f_run') as display:
                result=f.train_f('index','pool','videos','imagenet',out,preflight=proof(out))
                self.assertEqual(result,out.resolve())
                self.assertIsNone(train.call_args.kwargs['input_pipeline'])
                self.assertIsNone(train.call_args.args[5])
                self.assertEqual(train.call_args.args[3],'imagenet')
                display.assert_called_once_with(out)
                plan=json.loads((out.parent/'experiment_plans.json').read_text())['resnet50']
                self.assertEqual(plan['preprocessing_mode'],'full_frame')
                self.assertEqual(plan['scheduler_monitor'],'DEV video ROC-AUC (unchanged)')
                self.assertEqual(plan['checkpoint_selection'],'frame_roc_auc')
                with self.assertRaises(FileExistsError):
                    f.train_f('index','pool','videos','imagenet',out,preflight=proof(out))
            other=Path(temp)/'resnet50_face_mask_roc_e'/'run';other.mkdir(parents=True)
            (other/'last.pt').write_bytes(b'KEEP E MASK')
            with self.assertRaises(ValueError):f._check_output(other)
            self.assertEqual((other/'last.pt').read_bytes(),b'KEEP E MASK')

    def test_stale_or_de_receipt_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(f,'validate_f_inputs',return_value=input_report()), \
             patch('cnn_experiment.training.train') as train:
            out=Path(temp)/f.EXPERIMENT_ID/'session'/'run';bad=proof(out);bad['input']='face_mask'
            with self.assertRaisesRegex(ValueError,'matching F-only preflight'):
                f.train_f('index','pool','videos','imagenet',out,preflight=bad)
            train.assert_not_called();self.assertFalse(out.parent.exists())

    def test_output_uniqueness_and_masked_resume_report_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=f.create_f_output(temp),f.create_f_output(temp)
            self.assertNotEqual(a,b)
            self.assertIn(f.EXPERIMENT_ID,a.parts)
            source=Path(temp)/'saved';saved_fixture(source)
            (source/'input_pipeline.json').write_text('{"kind":"face_mask"}')
            with self.assertRaisesRegex(ValueError,'cannot be relabelled'):f._saved_f(source)
            with patch.object(f,'validate_f_inputs',return_value=input_report()),patch('cnn_experiment.training.train') as train:
                with self.assertRaises(ValueError):
                    f.train_f('index','pool','videos','imagenet',a,preflight=proof(a),resume_from=source/'last.pt')
                train.assert_not_called()

    def test_reporting_all_columns_highlight_graphs_and_flat_zip_recovery(self):
        from cnn_experiment.reporting import epoch_table,export_epoch_report,GRAPH_ORDER
        from cnn_experiment.evaluation import plot_history
        from cnn_experiment.backup import package_model_run,restore_local_zip
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/f.EXPERIMENT_ID/'run';run.parent.mkdir();saved_fixture(run)
            rolling=package_model_run(run,'resnet50',3)
            rows,selected,_=epoch_table(run);self.assertEqual((len(rows),len(rows[0]),selected),(3,19,2))
            export_epoch_report(run,run,checkpoint_selection='frame_roc_auc',in_run=True)
            html=(run/'epoch_metrics.html').read_text();self.assertEqual(html.count('background:#fff0b3'),1)
            self.assertIn('DEV frame ROC-AUC',html)
            history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
            for row in history:row['train']['roc_auc']=.8
            plot_history(run,history,best_epoch=2,selection_metric='frame_roc_auc')
            final=package_model_run(run,'resnet50',3,final=True)
            with zipfile.ZipFile(rolling['archive']) as archive:
                for name in ('best.pt','last.pt','config.json','history.csv','history.jsonl'):
                    self.assertIn(name,archive.namelist())
            with zipfile.ZipFile(final['archive']) as archive:
                for name in ('best.pt','last.pt','epoch_metrics.csv','epoch_metrics.json','epoch_metrics.html',*GRAPH_ORDER):
                    self.assertIn(name,archive.namelist())
            restored=restore_local_zip(final['archive'],Path(temp)/'restored')
            self.assertEqual(epoch_table(restored),epoch_table(run))
            with patch('cnn_experiment.reporting.display_completed_run',return_value=rows) as display:
                self.assertEqual(f.display_f_run(final['archive']),rows)
                self.assertEqual(display.call_args.kwargs['kind'],'frame')

    def test_f_only_resume_delegates_to_existing_resume_engine_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'restored_F';saved_fixture(source)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            out=Path(temp)/f.EXPERIMENT_ID/'new_session'/'run'
            with patch.object(f,'validate_f_inputs',return_value=input_report()), \
                 patch('cnn_experiment.training.train',return_value=out) as train:
                f.train_f('index','pool','videos','imagenet',out,preflight=proof(out),
                          resume_from=source/'last.pt',display=False)
                self.assertEqual(train.call_args.args[5],source/'last.pt')
                self.assertIsNone(train.call_args.kwargs['input_pipeline'])
                self.assertNotEqual(train.call_args.args[4],source)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})

    def test_help_does_not_start_preflight_or_training(self):
        with patch.object(f,'create_f_output',side_effect=AssertionError('must not start')),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:f.main(['--help'])
            self.assertEqual(result.exception.code,0)


if __name__ == '__main__':unittest.main()
