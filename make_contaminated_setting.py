"""Create contaminated copies of a data setting (contamination as DATA, not pipeline).

For each pi, writes data/{setting}_pi{P}/ containing:
- tabular_dataset_train.csv with a random pi-fraction of the resulting fraud
  pile being flipped bogus rows (n_flip = pi * n_fraud / (1 - pi))
- tabular_dataset_validation.csv / tabular_dataset_test.csv copied UNCHANGED
- flipped_ids.csv (the ground-truth flip list)
- contamination_manifest.json

The unmodified pipeline scripts (run_score_reliability.py etc.) then run on the
new setting name exactly as on clean data — the scoring mechanism never knows
the data is contaminated.

Usage:
    python make_contaminated_setting.py --setting setting5 --pi 0.15,0.30
    python run_score_reliability.py --setting setting5_pi15 --probe-design anchored --K 40
"""
import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "supporting_set_constr"))

from config import DataConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create label-contaminated copies of a setting")
    parser.add_argument("--setting", type=str, required=True, help="Source setting (e.g., setting5)")
    parser.add_argument("--pi", type=str, default="0.15,0.30",
                        help="Comma-separated injected fractions of the resulting fraud pile")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    src_config = DataConfig(setting=args.setting)
    src_dir = src_config.data_dir
    train = pd.read_csv(src_dir / src_config.train_file)
    label_col, id_col = src_config.label_col, src_config.id_col
    pos_c, neg_c = src_config.positive_class, src_config.negative_class

    n_pos = int((train[label_col] == pos_c).sum())
    n_neg = int((train[label_col] == neg_c).sum())
    logger.info("Source %s train: %d rows (bogus=%d, fraud=%d)", args.setting, len(train), n_pos, n_neg)

    for pi_str in args.pi.split(","):
        pi = float(pi_str.strip())
        n_flip = int(round(pi * n_neg / (1.0 - pi)))
        if n_flip > n_pos:
            raise ValueError(f"pi={pi} needs {n_flip} flips but only {n_pos} bogus rows exist")

        dst_setting = f"{args.setting}_pi{int(round(pi * 100))}"
        dst_dir = PROJECT_ROOT / "data" / dst_setting
        dst_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(args.seed + int(round(pi * 1000)))
        pos_positions = np.flatnonzero((train[label_col] == pos_c).values)
        flipped_pos = np.sort(rng.choice(pos_positions, size=n_flip, replace=False))

        train_c = train.copy()
        train_c.iloc[flipped_pos, train_c.columns.get_loc(label_col)] = neg_c
        train_c.to_csv(dst_dir / src_config.train_file, index=False)

        for fname in (src_config.val_file, src_config.test_file):
            shutil.copyfile(src_dir / fname, dst_dir / fname)

        flipped_ids = train.iloc[flipped_pos][id_col]
        pd.DataFrame({id_col: flipped_ids.values}).to_csv(dst_dir / "flipped_ids.csv", index=False)

        manifest = {
            "artifact_type": "contaminated_setting",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_setting": args.setting,
            "pi": pi,
            "pi_definition": "injected fraction of resulting fraud pile: n_flip = pi*n_fraud/(1-pi)",
            "n_flipped": n_flip,
            "fraud_pile_after": n_neg + n_flip,
            "bogus_after": n_pos - n_flip,
            "seed": args.seed,
            "contaminated_files": [src_config.train_file],
            "unchanged_files": [src_config.val_file, src_config.test_file],
        }
        with open(dst_dir / "contamination_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Wrote %s: flipped %d bogus -> fraud (injected fraction %.3f of %d-row fraud pile)",
                    dst_setting, n_flip, n_flip / (n_neg + n_flip), n_neg + n_flip)


if __name__ == "__main__":
    main()
