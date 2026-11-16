"""Command line entry point: python -m egoqc list|check data/humanola_sample"""
from pathlib import Path

import typer

from egoqc.loaders.humanola import load_humanola

app = typer.Typer(help="Egocentric data quality engine")


@app.callback()
def main():
    """Keeps `list` a named subcommand (Typer drops the name when there is only one)."""


@app.command("list")
def list_clips(dataset: Path):
    """List every clip in a dataset folder, with its error if it has one."""
    total = 0
    bad = 0
    for clip in load_humanola(dataset):
        total += 1
        if clip.error:
            bad += 1
            typer.echo(f"{clip.path.name}  ERROR: {clip.error}")
        else:
            w, h = clip.resolution
            typer.echo(
                f"{clip.path.name}  {w}x{h}  {clip.fps:.2f} fps  {clip.duration:.1f} s"
            )
    typer.echo(f"{total} clips, {bad} with errors")


@app.command("check")
def check_clips(
    dataset: Path,
    out: Path = typer.Option(None, help="CSV file for the results table (default: outputs/report_<dataset>.csv)"),
    limit: int = typer.Option(None, help="Only check the first N clips"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Settings file"),
):
    """Run every QC check on every clip and write the results table, one row per clip."""
    import time

    import pandas as pd

    from egoqc.checks.runner import (
        CHECKS, SAMPLE_COLUMNS, clip_row, results_from_scan, results_table, sample_rows,
        scan_with_config,
    )
    from egoqc.config import load_config
    from egoqc.quiet import quiet_stderr

    config = load_config(config_path)  # read once, passed to every check
    out = out or Path("outputs") / f"report_{dataset.name}.csv"

    rows = []
    scores = []  # one row per sample, for the dashboard's timelines
    for i, clip in enumerate(load_humanola(dataset, config_path)):
        if limit is not None and i >= limit:
            break
        start = time.perf_counter()
        with quiet_stderr():
            scan = scan_with_config(clip, config)
        results = results_from_scan(clip, scan, config)
        row = clip_row(clip, results)
        scores.extend(sample_rows(clip, scan))
        rows.append(row)
        verdict = "PASS" if row["passed"] else f"FAIL ({row['failed_checks']})"
        typer.echo(f"{clip.path.name}  {verdict}  [{time.perf_counter() - start:.0f} s]")
        for name, r in results.items():
            if not r.passed:
                typer.echo(f"    {name}: {r.note}")

    if not rows:
        typer.echo("no clips found")
        raise typer.Exit(1)

    table = results_table(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    scores_out = out.with_name(out.name.replace("report_", "scores_", 1))
    if scores_out == out:
        scores_out = out.with_name(f"scores_{out.name}")
    pd.DataFrame(scores, columns=SAMPLE_COLUMNS).to_csv(scores_out, index=False)

    # Compact view: overall verdict plus each check's bad share
    view = table[["clip", "passed"] + [f"{name}_bad_share" for name in CHECKS]]
    view.columns = ["clip", "passed"] + list(CHECKS)
    typer.echo("")
    typer.echo(view.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    typer.echo(f"\n{int(table['passed'].sum())} of {len(table)} clips passed. Table written to {out}, per-sample scores to {scores_out}")


if __name__ == "__main__":
    app()
