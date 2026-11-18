"""Export the final model bundle used by the API.

Reads the frozen selection from results/tuning/selection.json (chosen on
train only) and writes:
  artifacts/pathsnap_model.joblib   model + feature list + HMM config + emission params
  artifacts/pathsnap_config.json    the same metadata, human-readable (no model)

    python scripts/export_artifacts.py
"""
import json
from datetime import datetime, timezone

import joblib

from pathsnap.config import ARTIFACTS_DIR, RESULTS_DIR
from pathsnap.features import FEATURES
from pathsnap.inference import BUNDLE_PATH


def main():
    sel = json.loads((RESULTS_DIR / "tuning" / "selection.json").read_text())
    common = dict(features=FEATURES, radius=50.0, k=8,
                  created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  trained_on="train (80 trips); hyperparameters tuned with Optuna TPE on train only",
                  selection_rule=sel["rule"])
    if sel["chosen"] == "hmm_baseline_tuned":
        bundle = dict(model_name="hmm_baseline_tuned", model=None, hmm=sel["hmm_baseline"]["hmm"],
                      emission_weight=1.0, dist_sigma=None, **common)
    else:
        h = sel["hybrid"]
        bundle = dict(model_name="lightgbm_tuned+hmm",
                      model=joblib.load(RESULTS_DIR / "cache" / "models" / "lightgbm_tuned.joblib"),
                      model_params=h["lgbm_params"], hmm=h["hmm"],
                      emission_weight=h["emission_weight"], dist_sigma=h["dist_sigma"], **common)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    meta = {k: v for k, v in bundle.items() if k != "model"}
    (ARTIFACTS_DIR / "pathsnap_config.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"wrote {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2, default=float))


if __name__ == "__main__":
    main()
