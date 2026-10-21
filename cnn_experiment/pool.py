"""Fixed, decoded-PTS-verified TRAIN pool. No full CNN candidate index is built."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

from .data import choose_indices, packet_timeline, merge_periods, rows_hash, stable_seed, FrameReader, write_json

POOL_PROTOCOL = 'cnn_fixed20k_pts10_safe_boundary_v1'


@dataclass(frozen=True)
class PoolConfig:
    per_class: int = 10000
    seed: int = 4266
    fps: int = 10
    boundary_guard: str = 'one_source_frame'
    preferred_spacing_seconds: float = .2

    def validate(self):
        if self.per_class < 1 or self.fps != 10 or self.boundary_guard != 'one_source_frame' or self.preferred_spacing_seconds < 0:
            raise ValueError('Invalid fixed-pool policy; 10 FPS remains fixed')
        return self


def training_rows(root):
    # Only split is examined before discarding other rows. No held-out annotation/path is processed.
    rows = [r for r in json.loads((Path(root)/'metadata.json').read_text(encoding='utf-8')) if r['split']=='train']
    if len(rows)!=4000:raise ValueError('Production pool requires all 4,000 TRAIN metadata rows')
    validate_training_rows(rows)
    return sorted(rows, key=lambda r:r['file'])


def validate_training_rows(rows):
    by_file = {r['file']:r for r in rows}
    if len(by_file)!=len(rows) or not rows:
        raise ValueError('Nonempty unique TRAIN metadata required')
    for r in rows:
        original = by_file.get(r['original'] or r['file'])
        if (r['split']!='train' or not r['file'].startswith('train/') or original is None or
            original['split']!='train' or original['modify_video'] or original['modify_audio']):
            raise ValueError('TRAIN source-family leakage or invalid original')


def visual_label(timestamp, periods, modify_video=True):
    return int(modify_video and any(Fraction(str(a))<=timestamp<Fraction(str(b)) for a,b in periods))


def safe_candidates(times, row, policy):
    periods = merge_periods(row['fake_periods']) if row['modify_video'] else []
    guard = source_interval(times)
    groups = {0:defaultdict(list),1:defaultdict(list)}
    for i in choose_indices(times, policy.fps):
        t = times[i]
        if any(abs(t-a)<=guard or abs(t-b)<=guard for a,b in periods):
            continue
        label = visual_label(t,periods)
        segment = next((j for j,(a,b) in enumerate(periods) if a<=t<b),
                       sum(t>=b for a,b in periods))
        groups[label][segment].append(i)
    return groups


def source_interval(times):
    """One frame at the actual mean FPS measured from the source PTS span."""
    if len(times)<2 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('At least two increasing source timestamps required')
    return (times[-1]-times[0])/(len(times)-1)


def allocate_quotas(capacities, videos, target, seed):
    families = defaultdict(list)
    for vid,cap in capacities.items():
        if cap: families[videos[vid]['source_family']].append(vid)
    if sum(capacities.values())<target:
        raise ValueError(f'Only {sum(capacities.values())} safely eligible frames; need {target}. No rule relaxed.')
    rng=random.Random(seed); family_names=sorted(families);rng.shuffle(family_names)
    queues={}
    for family in family_names:
        ids=sorted(families[family]);rng.shuffle(ids);queues[family]=deque(ids)
    quota=Counter(); active=deque(family_names)
    for _ in range(target):
        family=active.popleft(); q=queues[family];vid=q.popleft();quota[vid]+=1
        if quota[vid]<capacities[vid]:q.append(vid)
        if q:active.append(family)
    return quota


def spread_indices(groups, times, count, seed):
    """Represent short segments first, then maximize temporal separation; spacing is soft."""
    rng=random.Random(seed)
    keys=sorted(groups,key=lambda k:(len(groups[k]),stable_seed(seed,k)))
    chosen=[]; remaining={k:list(groups[k]) for k in keys}
    while len(chosen)<count:
        for key in keys:
            candidates=remaining[key]
            if not candidates:continue
            if chosen:
                pick=max(candidates,key=lambda i:(min(abs(times[i]-times[j]) for j in chosen),stable_seed(seed,i)))
            else:pick=candidates[rng.randrange(len(candidates))]
            chosen.append(pick);candidates.remove(pick)
            if len(chosen)==count:break
    return chosen


def _source(root, source):
    path=(Path(root)/source).resolve()
    if not source.startswith('train/') or Path(root).resolve() not in path.parents:
        raise ValueError('TRAIN source escaped root')
    return path


def build_pool(data_root, output, policy=PoolConfig(), *, rows=None, timeline_fn=None, reader=None):
    """Two bounded metadata passes; decode ONLY final selections, grouped by video.

    Injection arguments support synthetic tests, not a production verification bypass.
    A partial build is never accepted as a complete pool; existing outputs are refused.
    """
    policy.validate();out=Path(output)
    if out.exists():raise FileExistsError('Pool exists; reuse its complete manifest or choose a new explicit pool version.')
    out.mkdir(parents=True)
    rows=training_rows(data_root) if rows is None else rows
    validate_training_rows(rows);timeline_fn=timeline_fn or packet_timeline
    videos={};capacities={0:{},1:{}};short=[]
    for vid,row in enumerate(rows,1):
        path=_source(data_root,row['file']);times=timeline_fn(path,row['video_frames'])
        groups=safe_candidates(times,row,policy)
        periods=merge_periods(row['fake_periods']) if row['modify_video'] else []
        videos[vid]={'id':vid,'source_file':row['file'],'source_family':row['original'] or row['file'],
            'video_label':int(row['modify_video']),'modify_audio':row['modify_audio'],
            'periods':[[str(a),str(b)] for a,b in periods], 'source_bytes':path.stat().st_size,
            'source_frame_interval_exact':str(source_interval(times)), 'actual_fps':float(1/source_interval(times)),
            'timeline_sha256':hashlib.sha256('|'.join(map(str,times)).encode()).hexdigest()}
        for y in (0,1):capacities[y][vid]=sum(map(len,groups[y].values()))
        for j,(a,b) in enumerate(periods):
            short.append({'video_id':vid,'interval':j,'duration':float(b-a),
                          'eligible':len(groups[1].get(j,[]))})
        if vid%100==0:print(f'Pool eligibility: {vid}/{len(rows)} TRAIN timelines; no full frame index stored',flush=True)
    audit={'eligible_by_class':{str(y):sum(capacities[y].values()) for y in (0,1)},
           'intervals_without_safe_grid_frame':sum(s['eligible']==0 for s in short),
           'excluded_intervals':[s for s in short if not s['eligible']], 'policy':asdict(policy)}
    write_json(out/'eligibility_report.json',audit)
    quotas={y:allocate_quotas(capacities[y],videos,policy.per_class,stable_seed(policy.seed,y)) for y in (0,1)}
    frames=[];reader=reader or FrameReader(data_root,0)
    for vid,row in enumerate(rows,1):
        if not any(quotas[y][vid] for y in (0,1)):continue
        times=timeline_fn(_source(data_root,row['file']),row['video_frames'])
        if hashlib.sha256('|'.join(map(str,times)).encode()).hexdigest()!=videos[vid]['timeline_sha256']:
            raise ValueError('Source timeline changed between pool passes')
        groups=safe_candidates(times,row,policy); selected=[]
        for y in (0,1):
            for i in spread_indices(groups[y],times,quotas[y][vid],stable_seed(policy.seed,vid,y)):
                selected.append({'video_id':vid,'source_index':i,'timestamp_exact':str(times[i]),'label':y})
        selected.sort(key=lambda r:r['source_index'])
        records=[{'id':len(frames)+j,'split':'train','kind':'frame','source_file':row['file'],
                  'indices':[r['source_index']],'timestamps_exact':[r['timestamp_exact']]} for j,r in enumerate(selected)]
        pixels=reader.read(records)  # PTS, index, corruption and dimensions verified by existing decoder.
        del pixels
        for r in selected:
            r['id']=len(frames);r['decoded_verified']=True;frames.append(r)
        if vid%100==0:print(f'Pool decode verification: {len(frames):,}/{2*policy.per_class:,} selected frames',flush=True)
    content={'protocol':POOL_PROTOCOL,'policy':asdict(policy),'train_metadata_sha256':rows_hash(rows),
             'videos':list(videos.values()),'frames':frames,'complete':True,'audit':audit}
    path=out/'train_pool.json';temp=out/'train_pool.json.partial'
    temp.write_text(json.dumps(content,separators=(',',':'),allow_nan=False),encoding='utf-8')
    FixedPool(temp,expected_per_class=policy.per_class)
    temp.replace(path)
    pool=FixedPool(path,expected_per_class=policy.per_class)
    write_json(out/'pool_report.json',pool.report())
    return path


class FixedPool:
    def __init__(self,path,data_root=None,expected_per_class=10000):
        self.path=Path(path);raw=self.path.read_bytes();self.sha256=hashlib.sha256(raw).hexdigest()
        self.content=json.loads(raw);c=self.content
        if c.get('protocol')!=POOL_PROTOCOL or c.get('complete') is not True or c['policy']['per_class']!=expected_per_class:
            raise ValueError('Incomplete/incompatible fixed pool')
        PoolConfig(**c['policy']).validate()
        self.videos={v['id']:v for v in c['videos']};self.frames={r['id']:r for r in c['frames']}
        keys=set();counts=Counter()
        if len(self.frames)!=2*expected_per_class or len(self.frames)!=len(c['frames']):raise ValueError('Pool ID/count mismatch')
        for r in self.frames.values():
            v=self.videos[r['video_id']];t=Fraction(r['timestamp_exact']);key=(v['source_file'],r['source_index'])
            guard=Fraction(v['source_frame_interval_exact'])
            if (r['source_index']<0 or t<0 or guard<=0 or not v['source_file'].startswith('train/') or not v['source_family'].startswith('train/') or key in keys or
                not r['decoded_verified'] or r['label']!=visual_label(t,v['periods'],bool(v['video_label'])) or
                any(abs(t-Fraction(a))<=guard or abs(t-Fraction(b))<=guard for a,b in v['periods'])):
                raise ValueError('Pool uniqueness, split, boundary or label verification failed')
            keys.add(key);counts[r['label']]+=1
        if counts!={0:expected_per_class,1:expected_per_class}:raise ValueError('Pool must be exactly balanced')
        if data_root is not None:
            if rows_hash(training_rows(data_root))!=c['train_metadata_sha256']:raise ValueError('TRAIN metadata changed')
            for v in self.videos.values():
                if _source(data_root,v['source_file']).stat().st_size!=v['source_bytes']:raise ValueError('TRAIN video size changed')

    def records(self,ids=None):
        result=[]
        for cid in (sorted(self.frames) if ids is None else ids):
            r=self.frames[cid];v=self.videos[r['video_id']]
            result.append(r|{'split':'train','kind':'frame','source_file':v['source_file'],
                'source_family':v['source_family'],'video_label':v['video_label'],'indices':[r['source_index']],
                'timestamps_exact':[r['timestamp_exact']],'metric_eligible':True})
        return result

    def report(self):
        rows=self.records();pairs=defaultdict(list)
        for r in rows:pairs[r['source_file']].append(float(Fraction(r['timestamp_exact'])))
        gaps=[b-a for ts in pairs.values() for a,b in zip(sorted(ts),sorted(ts)[1:])]
        represented={(r['video_id'],j) for r in rows if r['label'] for j,(a,b) in enumerate(self.videos[r['video_id']]['periods'])
                     if Fraction(a)<=Fraction(r['timestamp_exact'])<Fraction(b)}
        return {'pool_sha256':self.sha256,'total':len(rows),'real':sum(r['label']==0 for r in rows),
            'fake':sum(r['label']==1 for r in rows),'unique_videos':len(pairs),
            'unique_families':len({r['source_family'] for r in rows}),
            'class_coverage':{str(y):{'videos':len({r['source_file'] for r in rows if r['label']==y}),
                'families':len({r['source_family'] for r in rows if r['label']==y})} for y in (0,1)},
            'visual_intervals_represented':len(represented),
            'visual_intervals_total':sum(len(v['periods']) for v in self.videos.values()),
            'boundary_guard':'one frame at actual mean source FPS measured from PTS; strict distance > guard',
            'adjacent_pairs_below_preferred_spacing':sum(g<self.content['policy']['preferred_spacing_seconds']-1e-9 for g in gaps),
            'minimum_selected_gap_seconds':min(gaps) if gaps else None,'all_decoded_verified':True,
            'test_used':False,'fixed_once':True,'audit':self.content['audit']}


class FixedPoolSampler:
    def __init__(self,pool,per_class=4000,seed=4266):
        self.pool=pool;self.per_class=per_class;self.seed=seed;self.epoch=-1
        self.protocol=getattr(pool,'protocol',POOL_PROTOCOL);self.kind=getattr(pool,'kind','frame')
        self.ids={y:sorted(i for i,r in pool.frames.items() if r['label']==y) for y in (0,1)}
        if not 0<per_class<=min(map(len,self.ids.values())):raise ValueError('Insufficient unique pool frames')
        self.cycles={0:-1,1:-1};self.queues={0:deque(),1:deque()};self.uses=Counter();self.previous=set()

    def state_dict(self):
        return {'protocol':self.protocol,'pool_sha256':self.pool.sha256,'per_class':self.per_class,'seed':self.seed,
            'epoch':self.epoch,'cycles':dict(self.cycles),'queues':{y:list(q) for y,q in self.queues.items()},
            'uses':dict(self.uses),'previous':sorted(self.previous)}

    def load_state_dict(self,s):
        if (s['protocol'],s['pool_sha256'],s['per_class'],s['seed'])!=(self.protocol,self.pool.sha256,self.per_class,self.seed):
            raise ValueError('Fixed-pool resume identity/policy mismatch')
        self.epoch=s['epoch'];self.cycles={int(y):v for y,v in s['cycles'].items()}
        self.queues={int(y):deque(q) for y,q in s['queues'].items()};self.uses=Counter({int(i):v for i,v in s['uses'].items()})
        self.previous=set(s['previous'])
        all_ids=set(self.pool.frames)
        if (self.epoch < -1 or set(self.queues)!={0,1} or set(self.cycles)!={0,1} or
            not self.previous<=all_ids or not set(self.uses)<=all_ids or
            any(type(n) is not int or n<1 for n in self.uses.values()) or
            sum(self.uses.values())!=(self.epoch+1)*2*self.per_class or
            (self.epoch>=0 and len(self.previous)!=2*self.per_class)):
            raise ValueError('Corrupt cycle/coverage state')
        for y,q in self.queues.items():
            if len(q)!=len(set(q)) or not set(q)<=set(self.ids[y]) or self.cycles[y]<-1:
                raise ValueError('Corrupt class-cycle queue')

    def select(self,epoch):
        if epoch!=self.epoch+1:raise ValueError('Fixed pool epochs must be sequential; restore sampler state or replay from zero')
        chosen=[]
        for y in (0,1):
            selected=[];seen=set();deferred=[]
            while len(selected)<self.per_class:
                if not self.queues[y]:
                    self.cycles[y]+=1
                    order=diverse_order(self.pool.records(self.ids[y]),stable_seed(self.seed,y,self.cycles[y],'fixed_cycle'))
                    self.queues[y]=deque(order)
                cid=self.queues[y].popleft()
                if cid in seen:deferred.append(cid);continue
                selected.append(cid);seen.add(cid)
            self.queues[y].extendleft(reversed(deferred));chosen.extend(selected)
        rng=random.Random(stable_seed(self.seed,epoch,'epoch_order'))
        real,fake=chosen[:self.per_class],chosen[self.per_class:]
        pairs=list(zip(real,fake));rng.shuffle(pairs);chosen=[]
        for pair in pairs:
            pair=list(pair);rng.shuffle(pair);chosen.extend(pair)
        reused=sum(i in self.uses for i in chosen);overlap=len(self.previous&set(chosen))
        self.uses.update(chosen);self.previous=set(chosen);self.epoch=epoch
        rows=self.pool.records(chosen);vc=Counter(r['source_file'] for r in rows);fc=Counter(r['source_family'] for r in rows)
        report={'epoch':epoch+1,'kind':self.kind,'samples':len(chosen),'real':self.per_class,'fake':self.per_class,
            'unique_samples':len(set(chosen)),'duplicate_rate':1-len(set(chosen))/len(chosen),
            'unique_videos':len(vc),'unique_families':len(fc),'max_samples_one_video':max(vc.values()),
            'max_samples_one_family':max(fc.values()),'previous_epoch_overlap':overlap,
            'previous_epoch_overlap_rate':overlap/len(chosen),'previously_seen_samples':reused,'reuse_rate':reused/len(chosen),
            'cumulative_unique':len(self.uses),'cumulative_coverage':len(self.uses)/len(self.pool.frames),
            'class_cumulative_coverage':{str(y):sum(i in self.uses for i in self.ids[y])/len(self.ids[y]) for y in (0,1)},
            'pool_sha256':self.pool.sha256,'cycles':dict(self.cycles),'cycle_remaining':{y:len(q) for y,q in self.queues.items()}}
        return chosen,report


def diverse_order(records,seed):
    """Every candidate exactly once; sweep families and videos before reuse."""
    rng=random.Random(seed);groups=defaultdict(lambda:defaultdict(list))
    for r in records:groups[r['source_family']][r['source_file']].append(r['id'])
    families=sorted(groups);rng.shuffle(families);active=deque(families);queues={}
    for family,videos in groups.items():
        names=sorted(videos);rng.shuffle(names);queues[family]=deque(names)
        for ids in videos.values():rng.shuffle(ids)
    order=[]
    while active:
        family=active.popleft();q=queues[family];video=q.popleft()
        order.append(groups[family][video].pop())
        if groups[family][video]:q.append(video)
        if q:active.append(family)
    return order


class DevSchedule:
    """Read only the existing full DEV schedule; TRAIN frame tables are never queried."""
    def __init__(self,path,data_root=None):
        import sqlite3
        from .data import PROTOCOL, train_dev_rows
        self.path=Path(path).resolve();self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        self.db.row_factory=sqlite3.Row;self.info=dict(self.db.execute('SELECT key,value FROM info'))
        if self.info.get('protocol')!=PROTOCOL or self.info.get('complete')!='true':raise ValueError('Incompatible DEV index')
        if data_root is not None and rows_hash(train_dev_rows(data_root))!=self.info['train_dev_metadata_sha256']:
            raise ValueError('Reattached metadata differs from the fixed DEV schedule')
        self.videos={}
        for row in self.db.execute("SELECT * FROM videos WHERE split='dev'"):
            v=dict(row);v['pts']=list(map(Fraction,json.loads(v['pts'])));v['sampled_indices']=json.loads(v['sampled_indices'])
            self.videos[v['id']]=v
            if not all(v[k].startswith('dev/') for k in ('source_file','source_family')):
                raise ValueError('DEV family/split leakage')
            if data_root is not None:
                source=(Path(data_root)/v['source_file']).resolve()
                if Path(data_root).resolve() not in source.parents or source.stat().st_size!=v['source_bytes']:
                    raise ValueError('DEV video changed or escaped root')
        policy=json.loads(self.info['policy'])
        if any(policy.get(k)!=v for k,v in {'fps':10,'clip_length':16,'clip_stride':12,'min_fake_frames':1,'boundary_exclusion':False}.items()):
            raise ValueError('DEV schedule preprocessing policy changed')
        n,fake=self.db.execute("SELECT COUNT(*),SUM(f.label) FROM frames f JOIN videos v ON v.id=f.video_id WHERE v.split='dev'").fetchone()
        if (len(self.videos),n,fake)!=(500,41746,1631):raise ValueError('Full fixed DEV schedule changed')
        self.report={'dev_frames':n,'dev_real':n-fake,'dev_fake':fake,'dev_videos':500,'test_used':False}

    def candidates(self,split,kind):
        from .data import Index
        if split!='dev':raise ValueError('This adapter exposes DEV only')
        return Index.candidates(self,split,kind)

    def records(self,split,kind,ids=None):
        from .data import Index
        if split!='dev':raise ValueError('This adapter exposes DEV only')
        return Index.records(self,split,kind,ids)

    def close(self):self.db.close()


def cnn_inputs(config,index_path,data_root,pool_path=None):
    from .data import Index, EpochSampler
    if config.training_pool=='fixed20k':
        if pool_path is None:raise ValueError('The fixed TRAIN pool manifest is required; no full-index fallback')
        pool=FixedPool(pool_path,data_root)
        return DevSchedule(index_path,data_root),FixedPoolSampler(pool,config.train_per_class,config.seed),pool
    index=Index(index_path,data_root)
    return index,EpochSampler(index,per_class=config.train_per_class,seed=config.seed),None


def cnn_records(index,pool,ids):
    return pool.records(ids) if pool is not None else index.records('train','frame',ids)
