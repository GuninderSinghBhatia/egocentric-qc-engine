# Runs every QC check on every clip and builds the results table (one row per clip)
import pandas as pd

from egoqc.checks.blur import summarize_scores as summarize_blur
from egoqc.checks.brightness import summarize_brightness
from egoqc.checks.corrupt import marks_from_decoded, summarize_decoding
from egoqc.checks.frame_drops import summarize_drops
from egoqc.checks.frozen import summarize_frozen
from egoqc.checks.hands import summarize_hands
from egoqc.checks.result import CheckResult
from egoqc.checks.scan import ClipScan, scan_clip
from egoqc.checks.spec import spec_check_with_config
from egoqc.clip import Clip

# Order = column order in the table. Cheap whole-clip checks first.
CHECKS = ["spec", "corrupt", "frame_drops", "frozen", "blur", "brightness", "hands"]


def _settings(config: dict, name: str) -> dict:
    settings = config.get("checks", {}).get(name)
    if settings is None:
        raise ValueError(f"config.yaml has no 'checks:' -> '{name}:' section")
    return settings


def run_all_checks(clip: Clip, config: dict) -> dict[str, CheckResult]:
    """Every check on one clip, decoding the video only once.

    Gives the same results as calling each <check>_check_with_config on its own.
    A clip with an error still gets a (failing, NaN) result per check.
    """
    return results_from_scan(clip, scan_with_config(clip, config), config)


def scan_with_config(clip: Clip, config: dict) -> ClipScan:
    """The one decoding pass, with the widths and spacing from config.yaml."""
    return scan_clip(
        clip,
        every_nth=_settings(config, "corrupt")["every_nth"],
        blur_width=_settings(config, "blur")["width"],
        frozen_width=_settings(config, "frozen")["width"],
        hand_model=_settings(config, "hands")["model_path"],
        hand_width=_settings(config, "hands")["width"],
    )


def results_from_scan(clip: Clip, scan: ClipScan, config: dict) -> dict[str, CheckResult]:
    """Every check's verdict from an existing scan (no video is read here)."""
    s = {name: _settings(config, name) for name in CHECKS}
    return {
        "spec": spec_check_with_config(clip, config),
        "corrupt": summarize_decoding(
            marks_from_decoded(clip, scan.decoded, s["corrupt"]["every_nth"]),
            s["corrupt"]["max_bad_share"],
        ),
        "frame_drops": summarize_drops(
            scan.timestamps, clip.fps, s["frame_drops"]["max_dropped_share"]
        ),
        "frozen": summarize_frozen(
            scan.frozen, s["frozen"]["difference_threshold"], s["frozen"]["max_bad_share"]
        ),
        "blur": summarize_blur(
            scan.blur, s["blur"]["score_threshold"], s["blur"]["max_bad_share"]
        ),
        "brightness": summarize_brightness(
            scan.brightness,
            s["brightness"]["minimum_brightness"],
            s["brightness"]["maximum_brightness"],
            s["brightness"]["max_bad_share"],
        ),
        "hands": summarize_hands(scan.hands, s["hands"]["max_bad_share"]),
    }


def clip_row(clip: Clip, results: dict[str, CheckResult]) -> dict:
    """One table row: the clip, the overall verdict, then bad_share, passed and note per check."""
    failed = [name for name, r in results.items() if not r.passed]
    row = {
        "clip": clip.path.name,
        "dataset": clip.source_dataset,
        "error": clip.error,
        "passed": not failed,
        "failed_checks": ", ".join(failed),
        "duration_s": clip.duration,
        "width": clip.resolution[0] if clip.resolution else None,
        "height": clip.resolution[1] if clip.resolution else None,
        "fps": clip.fps,
        "path": str(clip.path),
    }
    for name, r in results.items():
        row[f"{name}_bad_share"] = r.bad_share  # the per-check score: 0 = clean, NaN = unmeasured
        row[f"{name}_passed"] = r.passed
        row[f"{name}_note"] = r.note
        row[f"{name}_start_s"] = r.longest_bad_start_s  # where the worst stretch starts
    return row


SAMPLE_COLUMNS = ["clip", "time", "blur", "brightness", "frozen", "hands"]


def sample_rows(clip: Clip, scan: ClipScan) -> list[dict]:
    """One row per sampled frame with its blur, brightness and frozen scores (dashboard timelines)."""
    brightness = dict(scan.brightness)
    frozen = dict(scan.frozen)  # the last sample may have no next frame, so no frozen score
    hands = dict(scan.hands)
    return [
        {"clip": clip.path.name, "time": t, "blur": blur,
         "brightness": brightness.get(t), "frozen": frozen.get(t), "hands": hands.get(t)}
        for t, blur in scan.blur
    ]


def results_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
