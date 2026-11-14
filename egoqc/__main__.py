"""Command line entry point: python -m egoqc list data/humanola_sample"""
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


if __name__ == "__main__":
    app()
