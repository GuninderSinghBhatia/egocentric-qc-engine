"""Sample set names its head camera 'main_stereo_left'."""
from huggingface_hub import snapshot_download
snapshot_download("humanola-inc/egocentric-sample", repo_type="dataset", local_dir="data/humanola_sample",
                  allow_patterns=["videos/chunk-000/observation.images.main_stereo_left/*"], max_workers=4)
print("done sample video")
