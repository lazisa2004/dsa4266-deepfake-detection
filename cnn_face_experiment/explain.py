"""Paired DEV Grad-CAM using the SAME underlying source frames for both models."""
from pathlib import Path
import json

from cnn_experiment.gradcam import load_explanation_data, select_examples, save_explanations


def matched_gradcam(original_run, face_run, index_path, data_root, pipeline, output, *, device='cpu', count=1):
    original_run, face_run, output = Path(original_run), Path(face_run), Path(output)
    original = load_explanation_data(original_run/'best.pt', original_run/'best_dev_predictions.csv', index_path=index_path)
    face = load_explanation_data(face_run/'best.pt', face_run/'best_dev_predictions.csv', index_path=index_path)
    def identities(data):
        return [(r['source_file'], r['timestamps_exact'], r['label']) for r in data.records]
    if identities(original) != identities(face):
        raise ValueError('Matched explanation requires identical full DEV schedules')
    selections = {category: select_examples(original, category, count, seed=4266) for category in ('tp', 'tn', 'fp', 'fn')}
    indices = list(dict.fromkeys(i for values in selections.values() for i in values))
    output.mkdir(parents=True, exist_ok=False)
    (output/'matched_selection.json').write_text(json.dumps({'selected_by': 'original model DEV frame confusion categories',
        'selection': selections, 'warning': 'Coarse post-hoc evidence; not proof of manipulated pixels.'}, indent=2), encoding='utf-8')
    save_explanations(original, original_run/'best.pt', data_root, output/'original', indices,
                      device=device, selection='matched_original_confusion')
    save_explanations(face, face_run/'best.pt', data_root, output/'face_crop', indices,
                      device=device, selection='matched_original_confusion', input_pipeline=pipeline)
    return output
