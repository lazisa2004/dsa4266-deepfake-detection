"""Synthetic fixed-pool balance, cycles, diversity and lazy decoder checks."""
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import copy,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from cnn_experiment.pool import (PoolConfig,POOL_PROTOCOL,FixedPool,FixedPoolSampler,build_pool,
    source_interval,safe_candidates,visual_label)
from cnn_experiment.config import experiment_config


def pool_fixture(path,n=10000):
    videos=[];frames=[]
    for y in (0,1):
        for j in range(n):
            vid=y*100+j%100
            if j<100:
                videos.append({'id':vid,'source_file':f'train/{vid}.mp4','source_family':f'train/original{j%25}.mp4',
                    'video_label':y,'periods':[['0','1000']] if y else [],'source_bytes':0,
                    'source_frame_interval_exact':'1/30','actual_fps':30.})
            frames.append({'id':len(frames),'video_id':vid,'source_index':3*(j//100+1),
                'timestamp_exact':str(Fraction(j//100+1,10)),'label':y,'decoded_verified':True})
    value={'protocol':POOL_PROTOCOL,'complete':True,'policy':asdict(PoolConfig(per_class=n)),
        'train_metadata_sha256':'synthetic','videos':videos,'frames':frames,
        'audit':{'intervals_without_safe_grid_frame':0}}
    path.write_text(json.dumps(value),encoding='utf-8');return FixedPool(path,expected_per_class=n)


class PoolTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[2]/'outputs/experiment1_v2_20261019_metadata/candidates.sqlite').is_file(), 'Optional existing local DEV metadata unavailable')
    def test_dev_adapter_preserves_every_existing_frame_and_clip(self):
        from cnn_experiment.data import Index
        from cnn_experiment.pool import DevSchedule
        root=Path(__file__).resolve().parents[2];path=root/'outputs/experiment1_v2_20261019_metadata/candidates.sqlite'
        original=Index(path);dev=DevSchedule(path)
        try:
            for kind,n in (('frame',41746),('clip',3551)):
                before=original.records('dev',kind);after=dev.records('dev',kind)
                self.assertEqual(before,after);self.assertEqual(len(after),n)
            self.assertEqual(len(dev.videos),500)
            with self.assertRaises(ValueError):dev.candidates('train','frame')
        finally:original.close();dev.close()

    def test_shared_training_engine_commits_pool_identity_and_final_artifacts(self):
        """Mock GPU/epoch compute; exercise actual control flow, persistence and plots."""
        import contextlib,io,torch
        from dataclasses import replace
        from cnn_experiment import training
        from cnn_experiment.evaluation import binary_metrics,class_loss_metrics
        class Model(torch.nn.Module):
            def __init__(self,*args,**kwargs):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.phase='head'
            def cuda(self):return self
            def set_phase(self,active):self.phase='final_stage' if active else 'head'
            def trainable_report(self):return {'trainable_parameters':1}
        class Dev:
            def records(self,*args):return records
            def close(self):pass
        records=[]
        for family in range(8):
            for y in (0,1):
                records.append({'id':len(records),'split':'dev','kind':'frame','source_file':f'dev/{family}_{y}',
                    'source_family':f'dev/{family}','video_label':y,'label':y,'metric_eligible':True,'timestamps_exact':['1']})
        predictions=[{'record_index':i,'target':r['label'],'probability':.8 if r['label'] else .1} for i,r in enumerate(records)]
        metrics=binary_metrics([r['label'] for r in records],[p['probability'] for p in predictions])
        metrics.update(class_loss_metrics({0:1.,1:2.},{0:8,1:8}));metrics['loss']=3/16
        def epoch(*args,**kwargs):return copy.deepcopy(metrics),copy.deepcopy(predictions),.01
        with tempfile.TemporaryDirectory() as folder,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);pool=pool_fixture(root/'pool.json');index_path=root/'dev.sqlite';index_path.write_bytes(b'synthetic')
            out=root/'run';out.mkdir();cfg=replace(experiment_config('resnet50','fixed20k_lr15e6'),epochs=3,warmup_epochs=1,amp=False,heartbeat_seconds=0)
            stack.enter_context(patch.object(training,'cnn_inputs',return_value=(Dev(),FixedPoolSampler(pool),pool)))
            stack.enter_context(patch.object(training,'make_loader',return_value=object()))
            stack.enter_context(patch.object(training,'StagedCNN',Model))
            stack.enter_context(patch.object(training,'make_optimizer',side_effect=lambda m,c:torch.optim.AdamW([{'params':m.parameters(),'name':'head','lr':c.head_lr}])) )
            stack.enter_context(patch.object(training,'run_epoch',side_effect=epoch))
            for name,value in (('get_device_name','mock GPU'),('reset_peak_memory_stats',None),('max_memory_allocated',0),('empty_cache',None)):
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            with training.RunProgress(out,'resnet50',0) as progress:
                training._train(cfg,index_path,root,'.',out,None,progress,None,pool.path)
            state=torch.load(out/'last.pt',weights_only=True)
            self.assertEqual(state['pool_sha256'],pool.sha256);self.assertEqual(state['sampler']['epoch'],2)
            self.assertEqual([r['epoch'] for r in state['history']],[1,2,3])
            self.assertTrue((out/'dev_f2_calibration.json').is_file());self.assertTrue((out/'best_dev_roc_pr.png').is_file())
            self.assertTrue(list((root/'downloads').glob('resnet50_final_*.zip')))
            from cnn_experiment.checkpoints import load_resume_bundle
            with self.assertRaisesRegex(ValueError,'pool'):
                load_resume_bundle(out/'last.pt',replace(cfg,epochs=4),training.digest(index_path),pool_hash='different')

    def test_exact20k_same_cnn_ids_cycles_resume_and_family_diversity(self):
        with tempfile.TemporaryDirectory() as folder:
            pool=pool_fixture(Path(folder)/'train_pool.json')
            self.assertEqual(pool.report()['total'],20000)
            left=FixedPoolSampler(pool);right=FixedPoolSampler(pool);seen=set();third=None
            for epoch in range(10):
                ids,report=left.select(epoch);other,_=right.select(epoch)
                self.assertEqual(ids,other);self.assertEqual(len(set(ids)),8000)
                self.assertEqual(Counter(pool.frames[i]['label'] for i in ids),{0:4000,1:4000})
                self.assertEqual(report['unique_families'],25)
                if epoch<2:self.assertFalse(seen&set(ids))
                seen.update(ids)
                if epoch==1:
                    restored=FixedPoolSampler(pool);restored.load_state_dict(copy.deepcopy(left.state_dict()))
                    third=restored.select(2)[0]
                if epoch==2:
                    self.assertEqual(third,ids);self.assertEqual(len(seen),20000)
                for start in range(0,len(ids),32):self.assertEqual(sum(pool.frames[i]['label'] for i in ids[start:start+32]),16)
            for y in (0,1):
                uses=[left.uses[i] for i in left.ids[y]]
                self.assertEqual(min(uses),4);self.assertEqual(max(uses),4)
            a=experiment_config('resnet50','fixed20k_lr15e6');b=experiment_config('efficientnet_b0','fixed20k_lr15e6')
            self.assertEqual(a.seed,b.seed);self.assertEqual(a.backbone_lr,1.5e-5);self.assertEqual(a.dropout,.3)
            wrong=copy.deepcopy(left.state_dict());wrong['pool_sha256']='changed'
            with self.assertRaises(ValueError):right.load_state_dict(wrong)

    def test_actual_fps_boundary_guard_halfopen_and_audio_only(self):
        times=[Fraction(i,30) for i in range(301)];self.assertEqual(source_interval(times),Fraction(1,30))
        row={'modify_video':True,'fake_periods':[[1,2],[3.01,3.02]]}
        groups=safe_candidates(times,row,PoolConfig())
        selected=[i for y in groups.values() for group in y.values() for i in group]
        self.assertNotIn(30,selected);self.assertNotIn(60,selected)
        self.assertEqual(groups[1].get(1,[]),[])
        self.assertEqual(visual_label(Fraction(1),[[1,2]]),1);self.assertEqual(visual_label(Fraction(2),[[1,2]]),0)
        row['modify_video']=False
        self.assertFalse(safe_candidates(times,row,PoolConfig())[1])

    def test_manifest_rejects_label_duplicate_and_split_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'pool.json';pool_fixture(path)
            original=json.loads(path.read_text())
            for mutation in ('label','duplicate','split'):
                value=copy.deepcopy(original)
                if mutation=='label':value['frames'][0]['label']=1
                elif mutation=='duplicate':value['frames'][1]=value['frames'][0].copy()
                else:value['videos'][0]['source_file']='dev/a.mp4'
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):FixedPool(path)

    def test_small_real_decoded_build_and_spawned_loader(self):
        import av,numpy as np,torch
        from cnn_experiment.data import make_loader
        from cnn_experiment.config import Config
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'train').mkdir();rows=[]
            for y,name in ((0,'real'),(1,'fake')):
                path=root/'train'/f'{name}.mp4'
                with av.open(str(path),'w') as out:
                    stream=out.add_stream('mpeg4',rate=30);stream.width=224;stream.height=224;stream.pix_fmt='yuv420p'
                    for i in range(91):
                        frame=av.VideoFrame.from_ndarray(np.full((224,224,3),i%200,dtype=np.uint8),format='rgb24')
                        for packet in stream.encode(frame):out.mux(packet)
                    for packet in stream.encode():out.mux(packet)
                rows.append({'file':f'train/{name}.mp4','original':'train/real.mp4' if y else None,'split':'train',
                    'modify_video':bool(y),'modify_audio':False,'fake_periods':[[.5,2.5],[2.81,2.82]] if y else [],'video_frames':91})
            path=build_pool(root,root/'pool',PoolConfig(per_class=4),rows=rows)
            pool=FixedPool(path,expected_per_class=4);self.assertEqual(pool.report()['total'],8)
            self.assertEqual(pool.content['audit']['intervals_without_safe_grid_frame'],1)
            records=pool.records();cfg=Config(batch_size=2,workers=0)
            plain=torch.cat([x for x,_,_ in make_loader(records,root,cfg)])
            spawned=torch.cat([x for x,_,_ in make_loader(records,root,replace(cfg,workers=2))])
            self.assertTrue(torch.equal(plain,spawned))
            with self.assertRaises(FileExistsError):build_pool(root,root/'pool',PoolConfig(per_class=4),rows=rows)


if __name__=='__main__':unittest.main()
