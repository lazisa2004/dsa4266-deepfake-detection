"""Bounded synthetic CPU tests; no dataset access or real-model training."""
from dataclasses import replace
from fractions import Fraction
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from cnn_experiment.temporal import (TemporalConfig,ResNetLSTM,trained_resnet,clip_rows,ClipIndex,
    ClipPool,TEMPORAL_PROTOCOL)
from cnn_experiment.pool import FixedPoolSampler
from cnn_experiment.models import StagedCNN
from cnn_experiment.config import Config
from cnn_experiment.checkpoints import rng_state,load_resume_bundle,restore_training_state
from cnn_experiment.data import PROTOCOL
from cnn_experiment.backup import package_model_run,restore_local_zip


def manifest_fixture(path):
    videos=[];clips=[]
    for split in ('train','dev'):
      for family in range(8):
        for y in (0,1):
            vid=len(videos);times=[Fraction(i,10) for i in range(64)];indices=list(range(64))
            # One fake frame in the first clip, later clips genuine.
            periods=[['1/2','3/5']] if y else []
            videos.append({'id':vid,'split':split,'source_file':f'{split}/{vid}.mp4',
                'source_family':f'{split}/original{family}.mp4','video_label':y,'periods':periods,'source_bytes':0,
                'sampled_indices':indices,'sampled_timestamps':list(map(str,times))})
            clips.extend(clip_rows(vid,indices,times,periods,bool(y)))
    for i,r in enumerate(clips):r['id']=i
    path.write_text(json.dumps({'protocol':TEMPORAL_PROTOCOL,'complete':True,'videos':videos,'clips':clips,
        'policy':{'fps':10,'length':16,'stride':12,'min_fake_frames':1}}))
    return ClipIndex(path)


class TemporalTests(unittest.TestCase):
    def test_shared_temporal_engine_fresh_and_resume_without_gpu_or_optimizer_updates(self):
        import contextlib,io
        from unittest.mock import patch
        from cnn_experiment import training
        from cnn_experiment.evaluation import binary_metrics,class_loss_metrics
        from cnn_experiment.checkpoints import digest
        with tempfile.TemporaryDirectory() as folder,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);manifest=root/'clips.json';index=manifest_fixture(manifest)
            cfg=TemporalConfig(hidden_size=4,clip_per_class=1,epochs=2,amp=False,heartbeat_seconds=0)
            records=index.records('dev','clip')
            predictions=[{'record_index':i,'target':r['label'],'probability':.8 if r['label'] else .1} for i,r in enumerate(records)]
            metrics=binary_metrics([r['label'] for r in records],[p['probability'] for p in predictions])
            counts={y:sum(r['label']==y for r in records) for y in (0,1)}
            metrics.update(class_loss_metrics({y:counts[y]*.2 for y in (0,1)},counts));metrics['loss']=.2
            stack.enter_context(patch('cnn_experiment.temporal.ClipIndex',return_value=index))
            stack.enter_context(patch('cnn_experiment.temporal.trained_resnet',side_effect=lambda config,path:(ResNetLSTM(nn.Identity(),config),{'cnn_checkpoint_sha256':'synthetic'})))
            stack.enter_context(patch.object(nn.Module,'cuda',lambda self:self))
            stack.enter_context(patch.object(training,'make_loader',return_value=object()))
            stack.enter_context(patch.object(training,'run_epoch',side_effect=lambda *args,**kwargs:(copy.deepcopy(metrics),copy.deepcopy(predictions),.01)))
            for name,value in (('get_device_name','mock GPU'),('reset_peak_memory_stats',None),('max_memory_allocated',0),('empty_cache',None)):
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            out=root/'run';out.mkdir()
            with training.RunProgress(out,cfg.model,0) as progress:
                training._train(cfg,manifest,root,root/'cnn_best.pt',out,None,progress,None)
            before={p.name:digest(p) for p in out.glob('*.pt')};new=root/'new_session'/'resumed';new.mkdir(parents=True)
            with training.RunProgress(new,cfg.model,0) as progress:
                training._train(replace(cfg,epochs=3),manifest,root,root/'cnn_best.pt',new,out/'last.pt',progress,None)
            state=torch.load(new/'last.pt',weights_only=True)
            self.assertEqual(state['protocol'],TEMPORAL_PROTOCOL);self.assertEqual(state['epoch'],2)
            self.assertEqual([r['epoch'] for r in state['history']],[1,2,3])
            self.assertEqual(state['feature_source'],{'cnn_checkpoint_sha256':'synthetic'})
            self.assertTrue((new/'dev_f2_calibration.json').is_file());self.assertTrue((new/'best_dev_confusion.png').is_file())
            self.assertTrue(list((root/'downloads').glob('resnet50_lstm_final_*.zip')))
            self.assertEqual(before,{p.name:digest(p) for p in out.glob('*.pt')})

    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_one_fake_chronology_stride_audio_real_and_split_guard(self):
        times=[Fraction(i,10) for i in range(40)]
        fake=clip_rows(1,list(range(40)),times,[[.5,.6]],True)
        self.assertEqual(fake[0]['fake_count'],1);self.assertEqual(fake[0]['label'],1)
        self.assertEqual([r['position'] for r in fake],[0,12,24])
        self.assertTrue(all(r['label']==0 for r in clip_rows(1,list(range(40)),times,[[.5,.6]],False)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'clips.json';index=manifest_fixture(path)
            records=index.records('dev','clip');self.assertTrue(all(len(r['indices'])==16 for r in records))
            with self.assertRaises(ValueError):index.records('test','clip')
            raw=json.loads(path.read_text());raw['videos'][0]['source_family']='dev/leak.mp4';path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):ClipIndex(path)

    def test_clip_class_cycles_without_replacement_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            index=manifest_fixture(Path(folder)/'clips.json');pool=ClipPool(index)
            a=FixedPoolSampler(pool,1);a.select(0);saved=copy.deepcopy(a.state_dict())
            b=FixedPoolSampler(pool,1);b.load_state_dict(saved)
            self.assertEqual(a.select(1),b.select(1))
            for epoch in range(2,10):
                ids,report=a.select(epoch);self.assertEqual(len(ids),len(set(ids)));self.assertEqual(report['kind'],'clip')
            self.assertEqual(TemporalConfig().clip_per_class,1500)

    def test_actual_resnet_prehead2048_selected_checkpoint_and_frozen_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'best.pt';cfg=TemporalConfig(hidden_size=4,feature_microbatch=2)
            with self.assertRaises(FileNotFoundError):trained_resnet(cfg,path)
            cnn=StagedCNN(Config(),'.',pretrained=False)
            torch.save({'protocol':PROTOCOL,'config':Config().as_dict(),'epoch':0,'best_epoch':1,
                'history':[{'epoch':1}],'model':cnn.state_dict()},path)
            model,identity=trained_resnet(cfg,path)
            self.assertEqual(identity['feature_dim'],2048);self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
            model.train();self.assertFalse(model.encoder.training)
            with torch.no_grad():features=model.encoder(torch.zeros(1,3,224,224))
            self.assertEqual(tuple(features.shape),(1,2048))
            features=features[:,None].expand(1,16,2048).clone()
            logits=model.classify_features(features);self.assertEqual(tuple(logits.shape),(1,1))
            nn.functional.binary_cross_entropy_with_logits(logits,torch.ones_like(logits)).backward()
            self.assertIsNotNone(model.lstm.weight_ih_l0.grad)
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            bad=torch.load(path,weights_only=True);bad['config']['model']='efficientnet_b0';torch.save(bad,path)
            with self.assertRaises(ValueError):trained_resnet(cfg,path)

    def test_chunked_features_shapes_and_zero_encoder_updates(self):
        class Encoder(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.ones(1));self.sizes=[]
            def forward(self,x):self.sizes.append(len(x));return x.mean((1,2,3))[:,None].expand(-1,2048)*self.scale
        cfg=TemporalConfig(hidden_size=4,lstm_layers=2,feature_microbatch=3)
        encoder=Encoder();model=ResNetLSTM(encoder,cfg);model.train()
        logits=model(torch.zeros(1,16,3,224,224));logits.sum().backward()
        self.assertEqual(sum(encoder.sizes),16);self.assertLessEqual(max(encoder.sizes),3)
        self.assertEqual(logits.shape,(1,1));self.assertIsNone(encoder.scale.grad)
        with self.assertRaises(ValueError):model.classify_features(torch.zeros(1,16,1))

    def test_temporal_checkpoint_zip_restore_exact_optimizer_rng_sampler(self):
        from cnn_experiment.training import GradientGuard
        from cnn_experiment.evaluation import write_history_metrics
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'run';out.mkdir();index=manifest_fixture(out/'clip_manifest.json')
            cfg=TemporalConfig(hidden_size=4,clip_per_class=1,epochs=3,amp=False)
            model=ResNetLSTM(nn.Identity(),cfg);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.head_lr)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max');scaler=torch.amp.GradScaler('cpu',enabled=False)
            guard=GradientGuard(cfg);guard.begin_epoch(0)
            loss=model.classify_features(torch.ones(2,16,2048)).square().mean();loss.backward()
            guard.finish_step(opt,scaler,epoch=0,batch=1,emit=lambda e:None)
            sampler=FixedPoolSampler(ClipPool(index),1);sampler.select(0)
            history=[{'epoch':1,'dev':{'roc_auc':.7},'dev_video':{'roc_auc':.6}}]
            state={'protocol':TEMPORAL_PROTOCOL,'config':cfg.as_dict(),'index_sha256':index.sha256,
                'pool_sha256':None,'epoch':0,'history':history,'model':model.state_dict(),'optimizer':opt.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':{},'rng':rng_state(),'sampler':sampler.state_dict(),
                'best':.4,'best_epoch':1,'stale':0,'best_balanced_dev_loss':.5,'feature_source':{'cnn_checkpoint_sha256':'synthetic'}}
            for name in ('best.pt','last.pt'):torch.save(state,out/name)
            (out/'config.json').write_text(json.dumps(cfg.as_dict()));write_history_metrics(out,history)
            receipt=package_model_run(out,'resnet50_lstm',1)
            restored=restore_local_zip(receipt['archive'],root/'restored')
            bundle=load_resume_bundle(restored/'last.pt',cfg,index.sha256,protocol=TEMPORAL_PROTOCOL)
            new=ResNetLSTM(nn.Identity(),cfg);newopt=torch.optim.AdamW(new.parameters(),lr=cfg.head_lr)
            newsched=torch.optim.lr_scheduler.ReduceLROnPlateau(newopt,mode='max');newsampler=FixedPoolSampler(ClipPool(index),1)
            restore_training_state(bundle,new,newopt,newsched,scaler,newsampler)
            self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in model.state_dict().items()))
            self.assertEqual(sampler.select(1),newsampler.select(1));self.assertEqual(bundle['report']['next_epoch'],2)
            self.assertTrue(newopt.state);self.assertEqual((restored/'last.pt').read_bytes(),(out/'last.pt').read_bytes())
            with self.assertRaises(FileExistsError):restore_local_zip(receipt['archive'],restored)


if __name__=='__main__':unittest.main()
