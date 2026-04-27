"""
Local inference for V-JEPA 2-AC + action head on Dobot Magician (Windows-only).

Single-process script that:
    - loads a trained checkpoint + V-JEPA 2-AC encoder
    - captures camera frames into an 8-frame rolling buffer
    - reads arm pose
    - predicts XYZ delta + gripper at HZ Hz
    - applies safety clamps and sends MOVE / suction-cup commands

Setup notes
-----------
The Dobot DLL is Windows-only, so this whole script must run on Windows.

Files needed in the same directory (or on PYTHONPATH):
    DobotDllType.py   + the matching DLLs from the Dobot SDK
    dataset.py        (re-used for IMG_SIZE, ImageNet stats, CONTEXT_FRAMES)
    train_head.py     (re-used for ActionHead, encode_frames, load_encoder)
    vjepa2_src/       cloned + URL-patched as on the cluster
    ckpt_*.pt         the trained checkpoint

Python deps:
    torch (CUDA strongly recommended; ViT-g is ~1B params),
    timm, einops, opencv-python, numpy

FPS / context window
--------------------
Training data was recorded at 20 Hz, so an 8-frame clip spans 0.4 s. Running
inference at, say, 5 Hz means the 8-frame buffer covers 1.6 s — a different
temporal distribution than training. If model behaviour seems sluggish or
out-of-distribution, raise --hz toward 20 (limited by V-JEPA forward time).

Camera
------
Training used the default camera resolution (whatever VideoCapture(0) returned)
and resized to 256x256 with no aspect-ratio preservation. We do the same here,
so the same camera and lens position should work without changes.
"""
import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize, resize

import DobotDllType as dType
from dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, CONTEXT_FRAMES
from train_head import ActionHead, encode_frames, load_encoder

# Safety limits in mm — copied from run_inference.py / dobot_server_windows.py
MAX_STEP_MM  = 20.0
Z_MIN, Z_MAX = -50.0, 150.0
MIN_RADIUS   = 140.0
MAX_RADIUS   = 310.0


def preprocess(img_bgr):
    """OpenCV BGR uint8 → ImageNet-normalised (3, 256, 256) torch tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    t = resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
    return normalize(t, IMAGENET_MEAN, IMAGENET_STD)


def safety_clamp(target_xyz, current_xyz):
    """Clamp Z, radial reach, and per-step velocity. All in mm."""
    x, y, z = target_xyz
    z = float(np.clip(z, Z_MIN, Z_MAX))
    r = float(np.sqrt(x * x + y * y))
    if r > MAX_RADIUS or r < MIN_RADIUS:
        scale = float(np.clip(r, MIN_RADIUS, MAX_RADIUS)) / r
        x *= scale
        y *= scale
    delta = np.array([x, y, z]) - current_xyz
    n = float(np.linalg.norm(delta))
    if n > MAX_STEP_MM:
        delta = delta * (MAX_STEP_MM / n)
        x, y, z = current_xyz + delta
    return float(x), float(y), float(z)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="ckpt_e128_h512_d2.pt")
    p.add_argument("--port",       default="COM3", help="Dobot serial port")
    p.add_argument("--cam",        type=int,   default=0,   help="OpenCV VideoCapture index")
    p.add_argument("--hz",         type=float, default=5.0, help="Control loop frequency")
    p.add_argument("--dry_run",    action="store_true",     help="Print actions but don't move the arm")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Model ---------------------------------------------------------------
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=device)

    print("Loading V-JEPA 2-AC encoder...")
    encoder = load_encoder(device)

    head = ActionHead(ckpt["embed_dim"], ckpt["mlp_hidden"], ckpt["mlp_depth"]).to(device).eval()
    head.load_state_dict(ckpt["head_state_dict"])
    print(f"Action head: e{ckpt['embed_dim']}_h{ckpt['mlp_hidden']}_d{ckpt['mlp_depth']}")

    s_mean = ckpt["s_mean"].to(device)
    s_std  = ckpt["s_std"].to(device)
    a_mean = ckpt["a_mean"]   # CPU; applied to numpy after the forward pass
    a_std  = ckpt["a_std"]

    # -- Dobot ---------------------------------------------------------------
    api = dType.load()
    print(f"Connecting to Dobot on {args.port}...")
    res = dType.ConnectDobot(api, args.port, 115200)[0]
    if res != dType.DobotConnect.DobotConnect_NoError:
        print("Failed to connect to Dobot.")
        sys.exit(1)
    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdClear(api)
    print("Dobot connected.")

    # -- Camera --------------------------------------------------------------
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Camera index {args.cam} failed to open.")
        sys.exit(1)

    # -- Initial buffer ------------------------------------------------------
    print("Filling 8-frame buffer...")
    buffer = deque(maxlen=CONTEXT_FRAMES)
    while len(buffer) < CONTEXT_FRAMES:
        ret, frame = cap.read()
        if ret:
            buffer.append(preprocess(frame))
        time.sleep(1.0 / 30.0)

    # Track gripper in the same {-1, 1} form used in training so the state input matches.
    # Action output is binary {0=open, 1=closed} via sigmoid threshold on the logit.
    current_gripper_raw = -1.0   # start "open"

    print(f"\n>>> Inference loop running at {args.hz} Hz. Press 'q' in window or Ctrl+C to stop.")
    if args.dry_run:
        print(">>> DRY RUN — no MOVE / GRIP commands will be sent.\n")

    try:
        while True:
            t0 = time.time()

            # 1. New frame
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            buffer.append(preprocess(frame_bgr))

            # 2. Pose
            pose = dType.GetPose(api)
            current_xyz = np.array(pose[:3], dtype=np.float32)

            # 3. State = [x, y, z, gripper_raw]; z-score normalize
            state = torch.tensor(
                [pose[0], pose[1], pose[2], current_gripper_raw],
                dtype=torch.float32, device=device,
            )
            state_norm = (state - s_mean) / s_std

            # 4. Frames batch (1, 8, 3, 256, 256)
            frames = torch.stack(list(buffer)).unsqueeze(0).to(device)

            # 5. Forward pass
            with torch.no_grad():
                feats = encode_frames(encoder, frames)
                pred  = head(feats, state_norm.unsqueeze(0))[0].cpu()

            # 6. Decode action
            xyz_delta_mm   = (pred[:3] * a_std[:3] + a_mean[:3]).numpy()
            grip_logit     = float(pred[3])
            new_gripper_01 = 1 if grip_logit > 0 else 0
            new_gripper_raw = 1.0 if new_gripper_01 == 1 else -1.0

            # 7. Safety clamp
            target_xyz   = current_xyz + xyz_delta_mm
            sx, sy, sz   = safety_clamp(target_xyz, current_xyz)

            # 8. Log
            ms = (time.time() - t0) * 1000
            print(f"[{ms:5.0f}ms] Δ={np.round(xyz_delta_mm, 1)}mm  "
                  f"target=({sx:6.1f}, {sy:6.1f}, {sz:6.1f})  "
                  f"grip={grip_logit:+.2f}→{new_gripper_01}")

            # 9. Execute
            if not args.dry_run:
                dType.ClearAllAlarmsState(api)
                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, sx, sy, sz, 0, isQueued=0)
                if new_gripper_raw != current_gripper_raw:
                    dType.SetEndEffectorSuctionCup(api, 1, new_gripper_01, isQueued=0)
            current_gripper_raw = new_gripper_raw

            # 10. Visual debug
            cv2.imshow("Dobot View", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Pace
            dt = time.time() - t0
            time.sleep(max(0.0, 1.0 / args.hz - dt))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        dType.DisconnectDobot(api)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
