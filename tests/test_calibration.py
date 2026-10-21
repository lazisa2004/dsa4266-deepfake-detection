import unittest
from cnn_experiment.calibration import calibrated_f2,family_partition
from cnn_experiment.evaluation import binary_metrics


class CalibrationTests(unittest.TestCase):
    def fixture(self):
        records=[];predictions=[]
        for family in range(8):
            for label in (0,1):
                i=len(records)
                records.append({'id':i,'split':'dev','source_family':f'dev/{family}',
                    'source_file':f'dev/{family}_{label}','label':label})
                predictions.append({'record_index':i,'probability':.4 if label else .3,'target':label})
        return records,predictions

    def test_family_partition_disjoint_reproducible_f2_and_ranking_unchanged(self):
        records,preds=self.fixture();report=calibrated_f2(records,preds)
        self.assertFalse(set(report['calibration_families'])&set(report['reporting_families']))
        self.assertEqual(report,calibrated_f2(records,list(reversed(preds))))
        self.assertEqual(report['threshold'],.4);self.assertEqual(report['reporting']['f2'],1.)
        self.assertEqual(report['reporting_at_0_5']['recall'],0.)
        a=report['reporting'];b=report['reporting_at_0_5']
        self.assertEqual(a['pr_auc'],b['pr_auc']);self.assertEqual(a['roc_auc'],b['roc_auc'])
        self.assertEqual(set(report['calibration_ids'])|set(report['reporting_ids']),set(range(16)))

    def test_no_test_and_no_in_sample_fallback(self):
        records,preds=self.fixture();records[0]['split']='test'
        with self.assertRaises(ValueError):calibrated_f2(records,preds)
        with self.assertRaises(ValueError):family_partition(self.fixture()[0][:2])


if __name__=='__main__':unittest.main()
