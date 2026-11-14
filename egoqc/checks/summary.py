# Shared by every check: turn per-sample bad/good marks into the facts and the verdict
from egoqc.checks.result import CheckResult


def summarize_marks(
    check: str,
    scores: list[tuple[float, float]],
    bad: list[bool],
    max_bad_share: float,
    what: str,
) -> CheckResult:
    """scores: (time, score) per sample; bad: one True/False per sample, same length;
    what: the word used in the note, e.g. "blurry"."""
    if not scores:  # nothing could be scored, for example the clip did not open
        return CheckResult(
            check=check, passed=False, bad_share=float("nan"), longest_bad_s=float("nan"),
            longest_bad_start_s=None, note="no frames could be scored", scores=[],
        )

    interval = scores[1][0] - scores[0][0] if len(scores) > 1 else 1.0  # seconds per sample
    bad_share = sum(bad) / len(scores)

    # Walk through the samples once, remembering where the current bad run started
    longest = 0  # length of the longest run, in samples
    longest_start = None  # index where that run started
    run = 0  # length of the run we are in right now
    for i, is_bad in enumerate(bad):
        if is_bad:
            run += 1
            if run > longest:
                longest = run
                longest_start = i - run + 1
        else:
            run = 0

    longest_bad_s = longest * interval
    start_s = scores[longest_start][0] if longest_start is not None else None
    if longest == 0:
        note = f"no {what} frames"
    else:
        note = (
            f"{bad_share:.1%} of sampled frames are {what}; "
            f"longest {what} stretch {longest_bad_s:.0f} s starting at {start_s:.0f} s"
        )
    return CheckResult(
        check=check, passed=bad_share <= max_bad_share, bad_share=bad_share,
        longest_bad_s=longest_bad_s, longest_bad_start_s=start_s, note=note, scores=scores,
    )
