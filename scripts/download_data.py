"""Download head-camera video + metadata + pose for the Humanola datasets."""
from huggingface_hub import snapshot_download

PATTERNS = ["README.md", "meta/*", "data/*", "videos/chunk-000/observation.images.head/*"]
SETS = {
    "humanola-inc/egocentric-sample": "data/humanola_sample",
    "humanola-inc/egocentric_labeled": "data/humanola_labeled",
}
for repo, dest in SETS.items():
    snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=PATTERNS, max_workers=4)
    print("done", repo)
