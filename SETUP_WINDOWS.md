# Windows Inference Setup

How to run [inference_local.py](inference_local.py) on the Windows machine that hosts the Dobot.

## What's missing from this repo

- `vjepa2_src/` — V-JEPA 2-AC source (not committed; lives on cluster with a URL patch)
- `ckpt_*.pt` — trained checkpoint (lives in `~/robot-arm/data/` on cluster)
- Dobot SDK DLLs (you already have these from data collection)

## Steps

### 1. Clone the repo on Windows
```bash
git clone <repo-url>
cd le-wm
```

### 2. Pull `vjepa2_src/` from the cluster (already URL-patched)
```bash
scp -r -i ~/.ssh/id_uwaterloo \
  tevillen@datasci-gpu.cs.uwaterloo.ca:~/robot-arm/vjepa2_src .
```

### 3. Pull the trained checkpoint
```bash
scp -i ~/.ssh/id_uwaterloo \
  tevillen@datasci-gpu.cs.uwaterloo.ca:~/robot-arm/data/ckpt_e128_h512_d2.pt .
```

### 4. Drop in the Dobot DLLs
Copy `DobotDll.dll`, `msvcp120.dll`, `msvcr120.dll`, `Qt5Core.dll`, `Qt5Network.dll`, etc. into the repo root next to `DobotDllType.py`. (Same set you used during teleop / data collection.)

### 5. Install with uv
```bash
# Default (CPU torch):
uv sync

# If you have an NVIDIA GPU with CUDA 12.8+ driver (faster inference):
uv sync --extra cu128

# If you have an older NVIDIA GPU stuck on CUDA 12.4:
uv sync --extra cu124
```

### 6. Sanity check (no arm motion)
```bash
uv run python inference_local.py --ckpt ckpt_e128_h512_d2.pt --dry_run
```
You should see V-JEPA load, the camera open, and per-tick action logs without the arm moving.

### 7. Real run
```bash
uv run python inference_local.py --ckpt ckpt_e128_h512_d2.pt --hz 5
```

Press `q` in the camera window or Ctrl+C to stop.

## Knobs
- `--hz` — control loop rate (default 5; raise toward 20 to match training rate if the GPU keeps up)
- `--cam` — OpenCV camera index (default 0)
- `--port` — Dobot serial port (default COM3)
- `--dry_run` — print actions, don't move

## Common issues

- **`DobotConnect_NotFound`** — wrong COM port; check Device Manager.
- **`CUDA out of memory`** — V-JEPA ViT-g needs ~8 GB VRAM. On smaller GPUs, fall back to CPU (`uv sync` with no index override) and lower `--hz`.
- **`vjepa2_src` ImportError** — make sure it's a folder, not a zip; should contain `src/hub/backbones.py` with the patched download URL.
- **Camera opens index 0 but it's the wrong camera** — try `--cam 1`.
