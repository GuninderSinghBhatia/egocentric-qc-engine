# Spec check: is the clip's measured resolution and frame rate what the target spec asks for?
from egoqc.checks.result import CheckResult
from egoqc.clip import Clip


def spec_check(clip: Clip, min_width, min_height, fps, fps_tolerance) -> CheckResult:
    """Whole-clip check: either the clip meets the spec or it does not (no sampling needed)."""
    if clip.resolution is None or clip.fps != clip.fps:  # fps != fps is True only for NaN
        return CheckResult(
            check="spec", passed=False, bad_share=float("nan"), longest_bad_s=float("nan"),
            longest_bad_start_s=None, note="resolution / fps could not be measured", scores=[],
        )

    width, height = clip.resolution
    problems = []
    if width < min_width or height < min_height:
        problems.append(f"resolution {width}x{height} is below {min_width}x{min_height}")
    if abs(clip.fps - fps) > fps_tolerance:
        problems.append(f"{clip.fps:.2f} fps is not within {fps} +/- {fps_tolerance}")

    passed = not problems
    if passed:
        note = f"{width}x{height} at {clip.fps:.2f} fps meets the spec"
    else:
        note = "; ".join(problems)
    # The whole clip shares one verdict, so it is either all bad or all good
    return CheckResult(
        check="spec", passed=passed, bad_share=0.0 if passed else 1.0,
        longest_bad_s=0.0 if passed else clip.duration,
        longest_bad_start_s=None if passed else 0.0, note=note, scores=[],
    )


def spec_check_with_config(clip: Clip, config: dict) -> CheckResult:
    """Run the spec check with settings from an already loaded config dict."""
    settings = config.get("checks", {}).get("spec")
    if settings is None:
        raise ValueError("config.yaml has no 'checks:' -> 'spec:' section")
    return spec_check(
        clip,
        min_width=settings["min_width"],
        min_height=settings["min_height"],
        fps=settings["fps"],
        fps_tolerance=settings["fps_tolerance"],
    )
