# One result shape shared by every QC check, so the final table can treat them alike
from pydantic import BaseModel


class CheckResult(BaseModel):
    check: str  # e.g. "blur"
    passed: bool
    bad_share: float  # share of sampled frames that were bad (0 to 1)
    longest_bad_s: float  # longest run of bad samples, in seconds
    longest_bad_start_s: float | None  # where that run started; None if there was no bad sample
    note: str  # readable sentence built from the numbers above
    scores: list[tuple[float, float]]  # (time in seconds, score) for every sample, nothing dropped
