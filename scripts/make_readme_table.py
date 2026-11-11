"""Regenerate the results tables in README.md from results/metrics.csv.

    python scripts/make_readme_table.py --config configs/full.yaml

Rebuilds results/tables.md from the metric files (no training, no figures)
and copies it into the marked block of README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from btc_aml import analysis, report  # noqa: E402
from btc_aml.utils import config_arg_parser, load_config  # noqa: E402

if __name__ == "__main__":
    args = config_arg_parser("Regenerate the README results tables.").parse_args()
    cfg = load_config(args.config)
    analysis.refresh_tables(cfg)
    report.update_readme(cfg)
