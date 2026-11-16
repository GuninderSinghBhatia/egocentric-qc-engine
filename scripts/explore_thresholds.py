# Recheck the Phase 2 thresholds on a dataset: one scan per clip, saves every per-sample score
# plus the results table, then prints how the scores sit relative to config.yaml's thresholds.
#
#   python -m scripts.explore_thresholds data/humanola_labeled [--limit N]
#
# Writes outputs/scores_<dataset>.csv (one row per sample) and outputs/report_<dataset>.csv.
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from egoqc.checks.runner import clip_row, results_from_scan, results_table, scan_with_config
from egoqc.config import load_config
from egoqc.loaders.humanola import load_humanola


def percentiles(values):
    v = np.asarray(values, dtype=float)
    return " ".join(f"p{p}={np.percentile(v, p):.2f}" for p in (0, 1, 5, 50, 95, 99, 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    checks = config["checks"]
    name = args.dataset.name
    samples, rows, gap_rows = [], [], []

    for i, clip in enumerate(load_humanola(args.dataset)):
        if args.limit is not None and i >= args.limit:
            break
        start = time.perf_counter()
        scan = scan_with_config(clip, config)
        results = results_from_scan(clip, scan, config)
        rows.append(clip_row(clip, results))

        frozen = dict(scan.frozen)
        brightness = dict(scan.brightness)
        for t, blur in scan.blur:
            samples.append({"clip": clip.path.name, "time": t, "blur": blur,
                            "brightness": brightness.get(t), "frozen": frozen.get(t)})
        if len(scan.timestamps) > 1:
            gaps = np.diff(scan.timestamps)
            gap_rows.append({"clip": clip.path.name, "min_gap": gaps.min(), "max_gap": gaps.max(),
                             "expected": 1 / clip.fps})
        failed = rows[-1]["failed_checks"] or "-"
        print(f"{clip.path.name}  failed: {failed}  [{time.perf_counter() - start:.0f} s]", flush=True)

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    scores = pd.DataFrame(samples)
    scores.to_csv(out / f"scores_{name}.csv", index=False)
    table = results_table(rows)
    table.to_csv(out / f"report_{name}.csv", index=False)

    print(f"\n=== {name}: {len(table)} clips, {len(scores)} samples ===")
    b = checks["blur"]
    print(f"blur        {percentiles(scores.blur)}")
    print(f"            below {b['score_threshold']}: {(scores.blur < b['score_threshold']).mean():.2%} of samples")
    br = checks["brightness"]
    print(f"brightness  {percentiles(scores.brightness)}")
    outside = (scores.brightness < br["minimum_brightness"]) | (scores.brightness > br["maximum_brightness"])
    print(f"            outside {br['minimum_brightness']}-{br['maximum_brightness']}: {outside.mean():.2%} of samples")
    fr = checks["frozen"]
    frozen = scores.frozen.dropna()
    print(f"frozen      {percentiles(frozen)}")
    print(f"            below {fr['difference_threshold']}: {(frozen < fr['difference_threshold']).mean():.2%} of samples")
    if gap_rows:
        g = pd.DataFrame(gap_rows)
        print(f"frame gaps  min {g.min_gap.min():.4f} s, max {g.max_gap.max():.4f} s, expected {g.expected.iloc[0]:.4f} s")
    print(f"\nclips passed: {int(table.passed.sum())} of {len(table)}")
    for check in ["spec", "corrupt", "frame_drops", "frozen", "blur", "brightness"]:
        failing = table[~table[f"{check}_passed"]]
        print(f"  {check:12s} failed {len(failing)}")
        for _, r in failing.iterrows():
            print(f"      {r['clip']}: {r[f'{check}_note']}")


if __name__ == "__main__":
    main()
