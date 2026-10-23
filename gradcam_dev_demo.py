"""Small DEV-only demonstration using one completed CNN run; no training or TEST."""
import argparse
import json
from pathlib import Path

from cnn_experiment.gradcam import (load_explanation_data, select_examples,
                                    save_explanations, top5_contributors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed run with best.pt and best_dev_predictions.csv')
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output folder')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=4266)
    args = parser.parse_args()
    data = load_explanation_data(args.run/'best.pt', args.run/'best_dev_predictions.csv',
                                 split='dev', index_path=args.index)
    categories = {name: select_examples(data, name, count=1, seed=args.seed) for name in
                  ('tp', 'tn', 'fp', 'fn', 'high_confidence_correct', 'high_confidence_incorrect')}
    selected = list(dict.fromkeys(i for indices in categories.values() for i in indices))
    if not selected:
        parser.error('No labelled examples available')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'selection_categories.json').write_text(json.dumps(categories, indent=2), encoding='utf-8')
    save_explanations(data, args.run/'best.pt', args.data_root, args.output/'categories',
                      selected, device=args.device, selection='seeded_dev_category_demo')
    source = data.records[selected[0]]['source_file']
    video = top5_contributors(data, source)
    save_explanations(data, args.run/'best.pt', args.data_root, args.output/'top5_video',
                      [p['record_index'] for p in video['contributors']], device=args.device,
                      selection='top5_contributors', video=video)
    print('Saved DEV categories and all Top-5% contributing frames to', args.output, flush=True)


if __name__ == '__main__':
    main()
