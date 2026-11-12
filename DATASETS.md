# Datasets

| Folder | Source | Licence | Used |
| --- | --- | --- | --- |
| `data/humanola_sample` | [humanola-inc/egocentric-sample](https://huggingface.co/datasets/humanola-inc/egocentric-sample) | CC BY 4.0 | 9 episodes, head camera only |
| `data/humanola_labeled` | [humanola-inc/egocentric_labeled](https://huggingface.co/datasets/humanola-inc/egocentric_labeled) | CC BY-NC 4.0 (non-commercial) | 38 episodes (~3.2 h), head camera only |
| `data/ha_ego_1k` | [humanarchive/HA-Ego-1K](https://huggingface.co/datasets/humanarchive/HA-Ego-1K) | CC BY-NC 4.0 (non-commercial) | Not available: the repo returns 401/404 even when logged in (checked 2026-11-13), so it was skipped |

Both Humanola sets are LeRobot v2.1 with 3D hand pose, finger angles and 200 Hz IMU. Videos are 1920x1200 at 30 fps. Only the head camera stream was downloaded.
Raw data is git-ignored. This project is non-commercial; the CC BY-NC sets are for portfolio use only. Confirm each licence at the source before relying on it.

Download command (example):

    uvx --from huggingface_hub hf download humanola-inc/egocentric-sample --repo-type dataset --local-dir data/humanola_sample --include "README.md" "meta/*" "data/*" "videos/chunk-000/observation.images.head/*"
