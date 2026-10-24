"""Synthetic DEV artifacts/gradients only. Never train, download assets or read TEST."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cnn_experiment.config import Config
from cnn_experiment.models import StagedCNN
from cnn_experiment.temporal import TemporalConfig,ResNetLSTM,TEMPORAL_PROTOCOL
from cnn_experiment.evaluation import binary_metrics,write_history_metrics
from cnn_experiment.data import PROTOCOL
from model_family_plan.plan import configuration
from model_family_plan.runner import INDEX_SHA256
from model_family_plan import comparison as c
from model_family_plan.interpretability import attribution,explain_master


def fixture(root,name='1a',complete=True,checkpoint=True):
    root=Path(root);root.mkdir(parents=True)
    cfg=configuration(name).as_dict();metrics=binary_metrics([0,0,1,1],[.1,.2,.8,.9])
    metrics.update(loss=.2,balanced_loss=.3,real_class_loss=.1,fake_class_loss=.5)
    video=binary_metrics([0,1],[.6,.9]);video.pop('f2')
    history=[{'epoch':i,'train':{'loss':1/i},'dev':dict(metrics),'dev_video':dict(video),
              'is_best':i<=2,'stale_epochs':max(0,i-2),'learning_rates':{'head':cfg['head_lr'],'final_stage':cfg['backbone_lr']}} for i in range(1,6)]
    write_history_metrics(root,history)
    (root/'config.json').write_text(json.dumps(cfg))
    (root/'experiment_plan.json').write_text(json.dumps({'experiment_id':name,'preprocessing_mode':'full_frame'}))
    (root/'best_dev_metrics.json').write_text(json.dumps({'unit':metrics,'video':video,'losses':{'loss':.2,'balanced_loss':.3}}))
    if complete:(root/'training_summary.json').write_text(json.dumps({'status':'trained','epochs_run':5,'best_epoch':2,'checkpoint_selection':'frame_roc_auc'}))
    if checkpoint:
        state={'protocol':PROTOCOL,'config':cfg,'epoch':1,'best_epoch':2,'best':1.,'index_sha256':INDEX_SHA256,
               'model':{'weight':torch.ones(1)},'history':history[:2]}
        torch.save(state,root/'best.pt')
    return root


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_only_1a_complete_pending_metrics_levels_video_f2(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'model_from_state',return_value=(nn.Identity(),None)):
            run=fixture(Path(tmp)/'1a')
            result=c.build_comparison({'1a':run},new_only=True)
            self.assertEqual(len(result['rows']),5)
            row=next(r for r in result['rows'] if r['Experiment ID']=='1a')
            self.assertEqual(row['Status'],'Completed');self.assertTrue(row['Verified'])
            self.assertAlmostEqual(row['Video F2'],5/6);self.assertEqual(row['Frame F2'],1.)
            self.assertEqual(row['TRAIN Loss'],.5);self.assertIsNone(row['Clip F2'])
            self.assertEqual(result['ranking']['gradcam_winner'],'1a')
            self.assertEqual(sum(r['Status']=='Pending' for r in result['rows']),4)
            html=c.comparison_html(result['rows'],'new')
            self.assertIn('overflow:auto',html);self.assertIn('background:#fff0b3',html)

    def test_partial_missing_and_invalid_checkpoints_never_rank(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'model_from_state',return_value=(nn.Identity(),None)):
            root=Path(tmp);partial=fixture(root/'partial',complete=False)
            missing=fixture(root/'missing',checkpoint=False)
            for path,status in ((partial,'Partial'),(missing,'Missing checkpoint')):
                result=c.build_comparison({'1a':path},new_only=True);row=result['rows'][0]
                self.assertEqual(row['Status'],status);self.assertFalse(row['Rank Eligible'])
                self.assertIsNone(result['ranking']['gradcam_winner'])
            state=torch.load(partial/'best.pt',weights_only=True);state['epoch']=0;torch.save(state,partial/'best.pt')
            self.assertEqual(c.load_row('1a',partial)[0]['Status'],'Invalid/unverified')
            self.assertEqual(c.load_row('1a',root/'not_attached')[0]['Status'],'Pending')

    def test_corrupt_or_incompatible_checkpoint_keeps_comparison_displayable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=fixture(root/'corrupt')
            for payload in (b'not a PyTorch checkpoint',b''):
                (run/'best.pt').write_bytes(payload)
                result=c.build_comparison({'1a':run})
                self.assertEqual(len(result['rows']),13)
                row=next(r for r in result['rows'] if r['Experiment ID']=='1a')
                self.assertEqual(row['Status'],'Invalid/unverified');self.assertFalse(row['Rank Eligible'])
                self.assertIn('Cannot safely load best.pt',row['Issue'])
                self.assertIn('Invalid/unverified',c.comparison_html(result['rows'],'audit'))
            torch.save(torch.ones(1),run/'best.pt')
            self.assertEqual(c.load_row('1a',run)[0]['Status'],'Invalid/unverified')

    def test_ranking_tie_break_separate_f1_f2_and_missing_leader_no_substitution(self):
        rows=[]
        for name,roc,pr,f1,f2 in (('1a',.9,.6,.7,.8),('2a',.9,.7,.6,.7),('3a',.8,.8,.8,.9)):
            row=c.blank(name);row.update({'Training Status':'Completed','Verified':True,'Rank Eligible':True,
                'Video ROC-AUC':roc,'Video PR-AUC':pr,'Video F1':f1,'Video F2':f2});rows.append(row)
        result=c.rank(rows);self.assertEqual(result['gradcam_winner'],'2a')
        self.assertEqual(result['leaders']['Video F1'],['3a']);self.assertEqual(result['leaders']['Video F2'],['3a'])
        rows[0].update({'Video ROC-AUC':.99,'Rank Eligible':False,'Verified':False,'Checkpoint Status':'Missing checkpoint'})
        result=c.rank(rows);self.assertIsNone(result['gradcam_winner']);self.assertEqual(result['verified_winner'],'2a')
        self.assertIn('no substitution',result['gradcam_blocked_reason'])
        with patch('model_family_plan.interpretability.build_comparison',return_value={'ranking':result}):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError,'no substitution'):explain_master({},'index','videos',Path(tmp)/'figures')
        rows[0].update({'Video ROC-AUC':.9,'Video PR-AUC':None})
        self.assertIn('tie-break',c.rank(rows)['gradcam_blocked_reason'])

    def test_thresholds_remain_separate_and_video_accuracy_not_borrowed(self):
        row=c.blank('3a');unit={'precision':1.,'recall':1.,'threshold':.5,'roc_auc':.9}
        video={'precision':.5,'recall':.8,'accuracy':.7,'threshold':.4,'roc_auc':.8}
        c._fill(row,unit,video,{},{});self.assertEqual(row['Clip F2'],1.)
        self.assertIsNone(row['Frame F2']);self.assertIsNone(row['Video F2']);self.assertIsNone(row['Video Accuracy'])
        self.assertEqual(row['Video ROC-AUC'],.8)

    def test_master_identities_duplicates_import_and_refresh(self):
        self.assertEqual(len(c.REGISTRY),13)
        names=list(c.REGISTRY);self.assertLess(names.index('1b'),names.index('original_efficientnet_b0'))
        self.assertLess(names.index('2b'),names.index('original_resnet50_lstm'))
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'model_from_state',return_value=(nn.Identity(),None)):
            root=Path(tmp);run=fixture(root/'run')
            with self.assertRaises(ValueError):c.build_comparison({'1a':run,'2a':run})
            copied=root/'copied';shutil.copytree(run,copied)
            wrong=c.load_row('original_resnet50',copied)[0]
            self.assertEqual(wrong['Status'],'Invalid/unverified')
            with self.assertRaises(ValueError):
                c._identity('original_resnet50',{'model':'resnet50'},None,{'experiment_id':'resnet50_full_frame_head5e5_roc_f'})
            ns={'NEW_RUNS':{},'NEW_OUTPUTS':{'1a':run}}
            self.assertEqual(c.collect_sources(ns)['1a'],run)
            replacement=root/'saved.zip';ns['NEW_RUNS']['1a']=replacement
            self.assertEqual(c.collect_sources(ns)['1a'],replacement)
            ns.update({'SAVED_EXPERIMENT_RUNS':{'resnet50_face_crop':replacement},'RUN_PATHS':{'resnet50_face_crop':run}})
            self.assertEqual(c.collect_sources(ns)['A'],replacement)  # do not replace saved completion with a partial new folder
            log={'output_type':'stream','text':['ResNet50 | completed | {}']}
            html='<table><tr><th>model</th><th>best_epoch</th><th>dev_video_roc_auc</th><th>dev_video_precision</th><th>dev_video_recall</th></tr><tr><td>resnet50</td><td>9</td><td>0.8</td><td>0.5</td><td>0.8</td></tr></table>'
            notebook=root/'saved.ipynb';notebook.write_text(json.dumps({'cells':[{'id':'actual','outputs':[{'data':{'text/html':html}},log]}]}))
            imported=c.import_notebook_table(notebook,0,0,{'resnet50':'original_resnet50'},root/'audit.json',completed_log_cells=(0,))
            result=c.build_comparison({},(imported,));row=result['rows'][0]
            self.assertEqual(row['Status'],'Missing checkpoint');self.assertAlmostEqual(row['Video F2'],5*.5*.8/(4*.5+.8))
            self.assertIsNone(row['Frame F2']);self.assertIn('sha256',row['Source'])
            self.assertEqual(result['ranking']['ordered_verified_ids'],[])
            clip_html=html.replace('resnet50','resnet50_lstm').replace('dev_video_precision','dev_clip_precision').replace('dev_video_recall','dev_clip_recall')
            notebook.write_text(json.dumps({'cells':[{'id':'actual_clip','outputs':[{'data':{'text/html':clip_html}},
                {'output_type':'stream','text':['resnet50_lstm | completed | {}']}]}]}))
            imported=c.import_notebook_table(notebook,0,0,{'resnet50_lstm':'original_resnet50_lstm'},root/'clip_audit.json',selection='clip_pr_auc',completed_log_cells=(0,))
            clip_row=next(r for r in c.build_comparison({},(imported,))['rows'] if r['Experiment ID']=='original_resnet50_lstm')
            self.assertIsNotNone(clip_row['Clip F2']);self.assertIsNone(clip_row['Video F2']);self.assertIsNone(clip_row['Frame F2'])

    def test_real_architecture_layers_finite_nonzero_gradient_and_weight_preservation(self):
        for name in ('resnet50','efficientnet_b0'):
            model=StagedCNN(Config(model=name),'.',pretrained=False).eval()
            for p in model.parameters():p.requires_grad_(False)
            before={k:v.clone() for k,v in model.state_dict().items()}
            result=attribution(model,torch.rand(1,3,224,224))
            self.assertEqual(result['heatmaps'].shape,(1,224,224))
            self.assertGreater(sum(result['gradient_l1_per_frame']),0)
            self.assertIn('layer4[-1]' if name=='resnet50' else 'features[-1]',result['target_layer'])
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_temporal_gradients_bypass_no_grad_equal_original_prediction_and_restore_flags(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__();self.layer4=nn.Sequential(nn.Conv2d(3,4,1));self.pool=nn.AdaptiveAvgPool2d(1)
            def forward(self,x):return self.pool(self.layer4(x)).flatten(1).repeat(1,512)
        cfg=replace(TemporalConfig(),hidden_size=4)
        model=ResNetLSTM(Encoder(),cfg).eval();clips=torch.rand(1,16,3,224,224)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():expected=float(model(clips).sigmoid()[0,0])
        with torch.no_grad():result=attribution(model,clips,temporal=True)
        self.assertAlmostEqual(result['probability'],expected,places=6)
        self.assertEqual(result['heatmaps'].shape,(16,224,224))
        self.assertTrue(all(n>0 for n in result['gradient_l1_per_frame']))
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertFalse(model.encoder.training);self.assertFalse(model.training)
        self.assertEqual(len(model.encoder.layer4[-1]._forward_hooks),0)
        with patch.object(model,'classify_features',side_effect=RuntimeError('synthetic error')):
            with self.assertRaises(RuntimeError):attribution(model,clips,temporal=True)
        self.assertEqual(len(model.encoder.layer4[-1]._forward_hooks),0)

    def test_strict_checkpoint_reconstruction_and_dev_guard(self):
        from model_family_plan.runner import INDEX_SHA256
        for name in ('resnet50','efficientnet_b0'):
            cfg=Config(model=name);model=StagedCNN(cfg,'.',pretrained=False)
            state={'protocol':PROTOCOL,'config':cfg.as_dict(),'model':model.state_dict()}
            loaded,_=c.model_from_state(state)
            self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items()))
        cfg=TemporalConfig(hidden_size=4)
        encoder=StagedCNN(Config(),'.',pretrained=False).network;encoder.fc=nn.Identity()
        model=ResNetLSTM(encoder,cfg)
        loaded,_=c.model_from_state({'protocol':TEMPORAL_PROTOCOL,'config':cfg.as_dict(),'model':model.state_dict()})
        self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(not p.requires_grad for p in loaded.encoder.parameters()))
        with self.assertRaises(PermissionError):c.guarded('/data/test/predictions.csv')

    def test_end_to_end_synthetic_png_csv_and_checkpoint_bytes_unchanged(self):
        import contextlib,csv,io
        from types import SimpleNamespace
        from cnn_experiment.gradcam import ExplanationData
        class Model(StagedCNN):
            def __init__(self):
                nn.Module.__init__(self);self.name='resnet50';self.phase='head'
                self.network=nn.Module();self.network.layer4=nn.Sequential(nn.Conv2d(3,1,1,bias=False))
                self.network.fc=nn.Identity()
                with torch.no_grad():self.network.layer4[0].weight.fill_(1)
            def forward(self,x):return self.network.layer4(x).mean((2,3))
        scores=[.9,.1,.95,.05];targets=[1,0,0,1]
        records=[{'id':i,'split':'dev','source_file':f'dev/synthetic{i}.mp4','timestamps_exact':['1/2'],
                  'label':targets[i],'video_label':targets[i],'indices':[5],'kind':'frame'} for i in range(4)]
        predictions=[{'record_index':i,'target':targets[i],'probability':p} for i,p in enumerate(scores)]
        data=ExplanationData('dev',records,predictions,.5,.5,'synthetic','synthetic')
        class Dataset:
            def __init__(self,selected,*a,**kw):
                self.selected=selected;self.reader=SimpleNamespace(read=lambda rows:[[np.full((224,224,3),128,np.uint8)] for r in rows])
            def __getitem__(self,i):
                p=scores[self.selected[i]['id']];return torch.full((3,224,224),np.log(p/(1-p))/3),0,i
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            run=fixture(Path(tmp)/'source');before=(run/'best.pt').read_bytes()
            stack.enter_context(patch('model_family_plan.interpretability.build_comparison',return_value={'ranking':{'gradcam_winner':'1a'}}))
            stack.enter_context(patch('model_family_plan.interpretability.model_from_state',return_value=(Model(),configuration('1a'))))
            stack.enter_context(patch('model_family_plan.interpretability.load_explanation_data',return_value=data))
            stack.enter_context(patch('model_family_plan.interpretability.LazyDataset',Dataset))
            out=explain_master({'1a':run},'index','videos',Path(tmp)/'interpretation')
            self.assertEqual(len(list(out.glob('*.png'))),4)
            with (out/'examples.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual({r['category'] for r in rows},{'tp','tn','fp','fn'})
            self.assertTrue(all(r['experiment_id']=='1a' and r['split']=='dev' for r in rows))
            self.assertEqual((run/'best.pt').read_bytes(),before)
            self.assertTrue(json.loads((out/'provenance.json').read_text())['weights_unchanged'])

    def test_notebook_append_only_and_separate_execution_interfaces(self):
        import ast
        path=Path(__file__).resolve().parents[1]/'notebooks/experiment1_face_cnn.ipynb'
        if not path.is_file():self.skipTest('Intentionally local-only notebook')
        nb=json.loads(path.read_text(encoding='utf-8'))
        cells=[cell for cell in nb['cells'] if cell['id'].startswith('family-comparison-')]
        self.assertEqual(len(cells),7)
        for cell in cells:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']));self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
        code=''.join(next(cell['source'] for cell in cells if cell['id']=='family-comparison-gradcam'))
        self.assertIn('explain_master(',code);self.assertNotIn('train(',code)


if __name__=='__main__':unittest.main()
