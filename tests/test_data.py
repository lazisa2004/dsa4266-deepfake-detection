"""Candidate/sampling and mounted-input regression checks; no CNN training."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from cnn_experiment.data import choose_indices, clip_grid, merge_periods, EpochSampler
from cnn_experiment.workflow import input_files, unique_path, discover_attached_inputs


class FixtureIndex:
    def __init__(self):
        self.videos = {}
        self.rows = []
        for family in range(4):
            for v in range(3):
                vid = family * 3 + v
                self.videos[vid] = {'source_family': str(family)}
                for label in (0, 1):
                    for j in range(8):
                        self.rows.append({'id': len(self.rows), 'video_id': vid, 'label': label,
                                          'fake_count': label * (j + 1), 'onset': int(label and j % 2 == 0), 'offset': label})

    def candidates(self, split, kind):
        return self.rows


class MetadataTests(unittest.TestCase):
    def test_audited_functions_match(self):
        import hashlib, json
        timelines = [[Fraction(i,25) for i in range(157)],
                     [Fraction(1,50)+Fraction(i,24) for i in range(95)],
                     [Fraction(i*i,10000) for i in range(100)]]
        expected = [
            '678cb6a2d5df67eadbc06382bfebdf92b7bead83e5fe25c834ba76690cb815a9',
            '4dcb1d02fab0129b27e998a4e220290acb0592db2c8086a13cf9ad0b3acb5ee7',
            '81652fb1755fb4c20a07f56cbc9d8fe2fe7e4de868017c41cc2d9418abd57751',
            'd7b01a682b3fbc2b8cf279eab9985461fcc0936a2e92ecaa5cb262e344012990',
            '5862946416765467341a7a44a77df29a0f509a62c0c84d12dc980747bb643f4b',
            'f1c825b4bfb30145a55f5cc1be4a1796faca5139fcc91b76840143d0f72016b2',
            'f5abc38b97b6f60b88a439b50b3ab7d4bb30aef1056a2db77323459a1b1dc861',
            'a072459dcaa7ba824075a9c61bc53d8d86f8e60c69612e2cf07fe00b9b7489d2',
            'a52be68c16186580a52cad10c349fe85b0d2a91c628e7e4ff547379f094c7e9c',
        ]
        actual = [hashlib.sha256(json.dumps(choose_indices(times,fps)).encode()).hexdigest()
                  for times in timelines for fps in (5,8,10)]
        self.assertEqual(actual, expected)
        grids = [[list(range(p,p+16)) for p in clip_grid(n)] for n in range(16,130)]
        self.assertEqual(hashlib.sha256(json.dumps(grids).encode()).hexdigest(), 'ff9d1d57f10caaebfb414d089f7fcb8c2b36bc20b56418b1419bff1bc71c5c6f')
        periods = [[.2,.4],[.4,.6],[.3,.5],[.8,1.]]
        self.assertEqual(merge_periods(periods), [(Fraction(1,5),Fraction(3,5)),(Fraction(4,5),Fraction(1))])

    def test_no_replacement_rotation_and_balance(self):
        sampler = EpochSampler(FixtureIndex(), per_class=24)
        first, d = sampler.select(0)
        second, d2 = sampler.select(1)
        self.assertEqual(len(set(first)), 48)
        self.assertEqual(sum(sampler.rows[i][1] for i in first), 24)
        self.assertEqual(d['unique_families'], 4)
        self.assertEqual(d['unique_videos'], 12)
        self.assertNotEqual(first, second)
        self.assertEqual(d2['previous_epoch_overlap'], 0)
        for p in range(0, len(first), 8):
            self.assertEqual(sum(sampler.rows[i][1] for i in first[p:p+8]), 4)

    def test_resume_and_epoch_replay(self):
        sampler = EpochSampler(FixtureIndex(), kind='clip', per_class=24)
        sampler.select(0)
        saved = sampler.state_dict()
        expected = sampler.select(1)[0]
        restored = EpochSampler(FixtureIndex(), kind='clip', per_class=24)
        restored.load_state_dict(saved)
        self.assertEqual(expected, restored.select(1)[0])
        replay = EpochSampler(FixtureIndex(), kind='clip', per_class=24)
        self.assertEqual(expected, replay.select(1)[0])
        self.assertEqual(saved['epoch'], 0)
        self.assertTrue(all(x == 0 for x in saved['last_used'].values()))

    def test_insufficient_pool(self):
        with self.assertRaises(ValueError):
            EpochSampler(FixtureIndex(), per_class=1000)


class KaggleInputLayoutTests(unittest.TestCase):
    def helpers(self, base):
        return {'find_input_markers': lambda name: input_files(name, base),
                'one_input': unique_path}

    def make_layout(self, base):
        files = {
            'lav-df-5000-dsa4266/LAV-DF-5000/metadata.json': '[]',
            'meta-dataa/candidates.sqlite': '',
            'weights/resnet50-11ad3fa6.pth': '',
            'weights/efficientnet_b0_rwightman-7f5810bc.pth': '',
        }
        for name, content in files.items():
            path = base/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        for split in ('train', 'dev', 'test'):
            (base/'lav-df-5000-dsa4266/LAV-DF-5000'/split).mkdir()
        return files

    def check_layout(self, base, discovery_base):
        self.make_layout(base)
        h = self.helpers(discovery_base)
        mounted = discover_attached_inputs(input_base=discovery_base)
        self.assertEqual(mounted['index'], base/'meta-dataa/candidates.sqlite')
        self.assertEqual(h['find_input_markers']('candidates.sqlite'), [base/'meta-dataa/candidates.sqlite'])
        self.assertEqual(h['find_input_markers']('resnet50-11ad3fa6.pth'), [base/'weights/resnet50-11ad3fa6.pth'])
        self.assertEqual(h['find_input_markers']('metadata.json'), [base/'lav-df-5000-dsa4266/LAV-DF-5000/metadata.json'])

    def test_flat_sidebar_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); self.check_layout(base, base)

    def test_nested_owner_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); self.check_layout(base/'datasets/leezile', base)

    def test_never_descends_into_video_split_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); self.make_layout(base)
            # Synthetic decoys prove that no held-out video folder is searched.
            for split in ('train', 'dev', 'test'):
                p = base/'lav-df-5000-dsa4266/LAV-DF-5000'/split
                (p/'candidates.sqlite').write_text('decoy')
                (p/'metadata.json').write_text('decoy')
                (p/'experiment1_v2').mkdir(); (p/'experiment1').mkdir()
                (p/'experiment1_v2/training.py').write_text('decoy')
                (p/'experiment1/models.py').write_text('decoy')
            h = self.helpers(base)
            self.assertEqual(h['find_input_markers']('candidates.sqlite'), [base/'meta-dataa/candidates.sqlite'])

    def test_ambiguous_and_missing_inputs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            h = self.helpers(Path(folder))
            with self.assertRaisesRegex(ValueError, 'INDEX_OVERRIDE'):
                h['one_input']([], 'INDEX_OVERRIDE')
            with self.assertRaisesRegex(ValueError, 'INDEX_OVERRIDE'):
                h['one_input']([Path('old'), Path('new')], 'INDEX_OVERRIDE')
