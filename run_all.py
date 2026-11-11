"""Run the whole project end to end.

    python run_all.py --config configs/quick.yaml     # minutes, 1 seed
    python run_all.py --config configs/full.yaml      # hours, 5 seeds

Useful options:
    --only gnn,analysis     run just these phases (results of earlier phases must exist)
    --skip rolling,explain  run everything except these phases

Phases, in order:
    data          download + validate + cache the dataset
    eda           exploratory figures and dataset statistics
    baselines     Logistic Regression, Random Forest, LightGBM, MLP
    gnn           GCN, Skip-GCN, GraphSAGE, GAT
    ablation      edge direction: undirected vs directed vs bidirectional
    hybrid        tree models on features + GNN output
    rolling       stretch goal: retrain before every test time step
    random_split  optional illustration of random-split score inflation
    analysis      tables and figures
    explain       SHAP / permutation importance / GNNExplainer

Results go to results/ and models/. If those folders hold a run made with a
DIFFERENT config, they are first moved to archive/<config>-<timestamp>/ so a
quick test run can never silently destroy hours of full-run results.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from btc_aml import (  # noqa: E402
    analysis, data, eda, explain, hybrid, random_split, report, rolling, train_baselines, train_gnn,
)
from btc_aml.utils import (  # noqa: E402
    ROOT, config_arg_parser, get_device, get_logger, load_config, load_json, models_dir,
    results_dir, save_json, timed,
)

log = get_logger()

PHASES = {
    "data": lambda cfg: data.load_bundle(cfg),
    "eda": eda.run,
    "baselines": train_baselines.run,
    "gnn": train_gnn.run,
    "ablation": train_gnn.run_ablation,
    "hybrid": hybrid.run,
    "rolling": rolling.run,
    "random_split": random_split.run,
    "analysis": analysis.run,
    "explain": explain.run,
}


def archive_other_run(cfg: dict) -> None:
    """Move results/ and models/ aside if they were produced by a different config."""
    info_path = results_dir(cfg) / "run_info.json"
    if not info_path.exists():
        return
    previous = load_json(info_path).get("config")
    if previous == cfg["run_name"]:
        return
    target = ROOT / "archive" / f"{previous}-{time.strftime('%Y%m%d-%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    for folder in (results_dir(cfg), models_dir(cfg)):
        shutil.move(str(folder), str(target / folder.name))
    log.info("Existing '%s' results archived to %s", previous, target)


def main() -> None:
    parser = config_arg_parser("Run the full pipeline.")
    parser.add_argument("--only", default="", help="comma-separated phases to run")
    parser.add_argument("--skip", default="", help="comma-separated phases to skip")
    args = parser.parse_args()
    cfg = load_config(args.config)

    only = [p for p in args.only.split(",") if p]
    skip = [p for p in args.skip.split(",") if p]
    unknown = [p for p in only + skip if p not in PHASES]
    if unknown:
        parser.error(f"unknown phase(s) {unknown}; choose from {list(PHASES)}")
    phases = [p for p in PHASES if (not only or p in only) and p not in skip]

    archive_other_run(cfg)
    save_json({"config": cfg["run_name"], "config_path": cfg["_config_path"]},
              results_dir(cfg) / "run_info.json")

    device = get_device(cfg)
    log.info("config=%s seeds=%s device=%s phases=%s", cfg["run_name"], cfg["seeds"], device, phases)

    # Keep timings of phases from earlier partial runs (--only), update the ones run now.
    timing_path = results_dir(cfg) / "timings.json"
    phase_seconds = load_json(timing_path)["phases"] if timing_path.exists() else {}
    for name in phases:
        with timed(name, phase_seconds):
            PHASES[name](cfg)

    ordered = {p: phase_seconds[p] for p in PHASES if p in phase_seconds}
    save_json(
        {
            "config": cfg["run_name"],
            "seeds": len(cfg["seeds"]),
            "device": str(device),
            "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPU cores, "
                       f"Python {platform.python_version()}",
            "phases": ordered,
            "total_seconds": round(sum(ordered.values()), 1),
        },
        timing_path,
    )
    # The timing table lives in tables.md, so rebuild the tables now that the
    # timings are known, then refresh the README block.
    if (results_dir(cfg) / "metrics.csv").exists():
        analysis.refresh_tables(cfg)
        report.update_readme(cfg)
    log.info("All done in %.1f min. Results: %s", sum(ordered.values()) / 60, results_dir(cfg))


if __name__ == "__main__":
    main()
