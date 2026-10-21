"""Recalculate metrics from recorded notebook confusion matrices; no model execution."""
import json
from pathlib import Path
source = Path(__file__).resolve().parents[1] / 'reports/confusion_matrices.json'
print('Evaluation,Precision,Recall,F1,Accuracy,Prevalence,Alerts')
for name, (tn, fp, fn, tp) in json.loads(source.read_text()).items():
    n = tn + fp + fn + tp
    print(f'{name},{tp/(tp+fp):.6f},{tp/(tp+fn):.6f},{2*tp/(2*tp+fp+fn):.6f},{(tn+tp)/n:.6f},{(tp+fn)/n:.6f},{tp+fp}')
