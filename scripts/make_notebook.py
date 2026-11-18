"""Build notebooks/phase1_results.ipynb: a walkthrough of every Phase 1 output.

The notebook only *reads* files produced by the scripts (no numbers typed in).
    python scripts/make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/phase1_results.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# PathSnap: Phase 1 results\nEvery table and figure here is loaded from `results/`, as written by the "
       "scripts in `scripts/` (see README for the run order)."),
    code("import json\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
         "R = Path('..') / 'results'\npd.set_option('display.precision', 4)\npd.set_option('display.width', 200)"),

    md("## 1. EDA"),
    code("summary = json.loads((R / 'eda' / 'eda_summary.json').read_text())\npd.DataFrame(summary)"),
    code("pd.read_csv(R / 'eda' / 'per_trip_stats.csv').describe().T"),
    code("for p in ['eda_trip_locations', 'eda_trips', 'eda_candidates', 'eda_features']:\n"
         "    display(Image(R / 'plots' / f'{p}.png'))"),
    md("Folium EDA maps: `results/maps/eda_train_*.html` (open in a browser)."),

    md("## 2. HMM/Viterbi baseline (Gaussian distance emission)"),
    code("pd.read_csv(R / 'hmm_baseline_summary.csv')"),

    md("## 3. Classifiers: GroupKFold CV on train, then student_test\n"
       "`*_clf` = classifier only (argmax per point); `*_hmm` = classifier probabilities as HMM emissions."),
    code("mc = pd.read_csv(R / 'model_comparison.csv')\n"
         "cols = ['model', 'family', 'cv_macro_f1', 'cv_macro_f1_std', 'cv_log_loss', 'top1_clf', 'top1_hmm',\n"
         "        'route_f1_clf', 'route_f1_hmm', 'continuity_clf', 'continuity_hmm', 'test_macro_f1',\n"
         "        'test_log_loss', 'train_fit_s', 'infer_ms_per_point_model']\n"
         "mc[[c for c in cols if c in mc]]"),
    code("pd.read_csv(R / 'cv_folds.csv').pivot_table(index='model', columns='fold', values='macro_f1')"),
    code("for p in ['cv_macro_f1', 'student_test_top1', 'student_test_route_f1', 'log_loss',\n"
         "          'confusion_matrices', 'loss_curves']:\n"
         "    display(Image(R / 'plots' / f'{p}.png'))"),

    md("## 4. Bayesian optimisation (Optuna TPE, train only)"),
    code("T = R / 'tuning'\nfor name in ['clf_lightgbm', 'hmm_baseline', 'hmm_hybrid']:\n"
         "    f = T / f'{name}_best.json'\n"
         "    if f.exists():\n"
         "        print(name, json.dumps(json.loads(f.read_text()), indent=2))"),
    code("for name in ['clf_lightgbm', 'hmm_baseline', 'hmm_hybrid']:\n"
         "    f = T / f'{name}_trials.csv'\n"
         "    if f.exists():\n"
         "        t = pd.read_csv(f)\n"
         "        print(f'{name}: {len(t)} trials, states: {t.state.value_counts().to_dict()}')\n"
         "        display(t.head(10))\n"
         "    p = R / 'plots' / f'tuning_{name}.png'\n"
         "    if p.exists():\n"
         "        display(Image(p))"),

    md("## 5. Final selection (by train objective) and student_test report"),
    code("print(json.dumps(json.loads((T / 'selection.json').read_text()), indent=2, default=str))"),
    code("pd.read_csv(T / 'final_comparison.csv')"),
    code("pd.read_csv(T / 'final_per_trip.csv')"),

    md("## 6. Exported bundle"),
    code("print((Path('..') / 'artifacts' / 'pathsnap_config.json').read_text())"),
    md("Prediction maps: `results/maps/index.html`."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "phase1_results.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
