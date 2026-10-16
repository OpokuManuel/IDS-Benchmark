"""Run every experiment in the report. Resumable: finished (protocol, model, fold) jobs are skipped.

    python scripts/02_run_experiments.py            # everything (~1 h on 1 CPU)
    python scripts/02_run_experiments.py --stage main
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from idsbench import SEED
from idsbench.models import MODEL_NAMES
from idsbench.run import run_protocol

ABLATIONS = {"_noTTL": ("sttl", "dttl", "ct_state_ttl"),
             "_noSEQ": ("stcpb", "dtcpb"),
             "_noCAT": ("proto", "service", "state")}

def main(stage):
    if stage in ("all", "main"):
        for m in MODEL_NAMES:
            run_protocol("P0_official", m)
            run_protocol("P1_random_clean", m)
            run_protocol("P2_grouped", m)                      # primary: 2 size-bins / octave
    if stage in ("all", "ablation"):
        for tag, drop in ABLATIONS.items():
            run_protocol("P2_grouped", "random_forest", drop=drop, tag=tag)
    if stage in ("all", "sensitivity"):
        for bins in (1, 4):
            for m in MODEL_NAMES:
                run_protocol("P2_grouped", m, bins=bins, tag=f"_bins{bins}")
    if stage in ("all", "seeds"):
        for seed in (SEED + 1, SEED + 2):
            for m in MODEL_NAMES:
                run_protocol("P2_grouped", m, seed=seed)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all",
        choices=["all", "main", "ablation", "sensitivity", "seeds"]); main(ap.parse_args().stage)
