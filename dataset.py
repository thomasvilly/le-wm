import glob
import os
import random

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, resize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Active dims: x, y, z, gripper — drops the constant (dim 3) and zero (dims 4,5) channels
ACTION_DIMS    = [0, 1, 2, 6]
IMG_SIZE       = 256
CONTEXT_FRAMES = 8  # V-JEPA 2-AC was post-trained on 8-frame clips


def _session_of(path):
    """Parent directory name = session identifier."""
    return os.path.basename(os.path.dirname(path))


class RobotEpisodeDataset(Dataset):
    """
    Sliding-window dataset over robot manipulation episodes stored as HDF5 files.

    Each sample is a window of CONTEXT_FRAMES consecutive frames ending at time t,
    plus the robot state at t and the target action at t+1.

    Expected HDF5 layout (per episode_XXX.h5):
        observations/images/top  : (T, H, W, 3) uint8
        observations/state       : (T, 7)        float32
        action                   : (T, 7)        float32

    Split logic:
        test_sessions  : session directory names held out entirely (never in train/val)
        split="test"   : windows from test_sessions only
        split="train/val": episode-level split within remaining episodes (no temporal leakage)
    """

    def __init__(
        self,
        data_dir,
        split="train",
        val_fraction=0.1,
        seed=42,
        state_mean=None,
        state_std=None,
        action_mean=None,
        action_std=None,
        test_sessions=None,
        augment=False,
    ):
        self.state_mean    = state_mean
        self.state_std     = state_std
        self.action_mean   = action_mean
        self.action_std    = action_std
        self.augment       = augment
        self.test_sessions = set(test_sessions) if test_sessions else set()

        h5_files = sorted(set(
            glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
            + glob.glob(os.path.join(data_dir, "*.h5"))
        ))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir} (searched recursively)")

        test_files = [p for p in h5_files if _session_of(p) in self.test_sessions]
        pool_files = [p for p in h5_files if _session_of(p) not in self.test_sessions]

        if split == "test":
            chosen_files = test_files
        else:
            rng      = np.random.default_rng(seed)
            ep_order = np.arange(len(pool_files))
            rng.shuffle(ep_order)
            n_val       = max(1, int(len(ep_order) * val_fraction))
            val_eps     = set(ep_order[:n_val].tolist())
            train_eps   = set(ep_order[n_val:].tolist())
            chosen_idxs = train_eps if split == "train" else val_eps
            chosen_files = [pool_files[i] for i in sorted(chosen_idxs)]

        self.episode_paths   = []
        self.episode_states  = []
        self.episode_actions = []
        self.windows         = []

        for path in chosen_files:
            with h5py.File(path, "r") as f:
                T      = f["action"].shape[0]
                state  = torch.tensor(f["observations/state"][:, ACTION_DIMS],  dtype=torch.float32)
                action = torch.tensor(f["action"][:, ACTION_DIMS],               dtype=torch.float32)

            ep_idx = len(self.episode_paths)
            self.episode_paths.append(path)
            self.episode_states.append(state)
            self.episode_actions.append(action)

            for start in range(T - CONTEXT_FRAMES):
                self.windows.append((ep_idx, start))

    @classmethod
    def fit_normalizers(cls, data_dir):
        """Compute z-score statistics across all episodes. Call once before creating splits.

        Action stats are computed on deltas (action[t+1] - state[t]) so the normalizer
        is calibrated to the actual prediction target, not absolute positions.
        """
        all_states, all_deltas = [], []
        all_h5 = sorted(set(
            glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
            + glob.glob(os.path.join(data_dir, "*.h5"))
        ))
        for path in all_h5:
            with h5py.File(path, "r") as f:
                state  = f["observations/state"][:, ACTION_DIMS].astype(np.float32)
                action = f["action"][:, ACTION_DIMS].astype(np.float32)
            all_states.append(state)
            all_deltas.append(action[1:] - state[:-1])
        states = np.concatenate(all_states, axis=0)
        deltas = np.concatenate(all_deltas, axis=0)
        s_mean = torch.tensor(states.mean(0),            dtype=torch.float32)
        s_std  = torch.tensor(states.std(0).clip(1e-6),  dtype=torch.float32)
        a_mean = torch.tensor(deltas.mean(0),            dtype=torch.float32)
        a_std  = torch.tensor(deltas.std(0).clip(1e-6),  dtype=torch.float32)
        return s_mean, s_std, a_mean, a_std

    def _load_frames(self, ep_idx, start):
        with h5py.File(self.episode_paths[ep_idx], "r") as f:
            imgs = f["observations/images/top"][start : start + CONTEXT_FRAMES]  # (8, H, W, 3)

        # Sample augmentation params once so all 8 frames in the clip are treated consistently.
        # Horizontal flip is excluded: flipping the image without negating the X-axis delta
        # label would corrupt XYZ prediction (camera→robot axis mapping is unknown).
        if self.augment:
            brightness = random.uniform(0.7, 1.3)
            contrast   = random.uniform(0.7, 1.3)
            saturation = random.uniform(0.8, 1.2)
            hue        = random.uniform(-0.1, 0.1)

        frames = []
        for img in imgs:
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div_(255.0)
            t = resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
            if self.augment:
                t = TF.adjust_brightness(t, brightness)
                t = TF.adjust_contrast(t, contrast)
                t = TF.adjust_saturation(t, saturation)
                t = TF.adjust_hue(t, hue)
            t = normalize(t, IMAGENET_MEAN, IMAGENET_STD)
            frames.append(t)
        return torch.stack(frames)  # (8, 3, 256, 256)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep_idx, start = self.windows[idx]

        frames    = self._load_frames(ep_idx, start)
        t         = start + CONTEXT_FRAMES - 1
        state     = self.episode_states[ep_idx][t].clone()
        action    = (self.episode_actions[ep_idx][t + 1] - self.episode_states[ep_idx][t]).clone()

        if self.state_mean is not None:
            state = (state - self.state_mean) / self.state_std

        # XYZ: normalise delta; gripper: absolute binary {0=open, 1=closed}
        abs_grip  = self.episode_actions[ep_idx][t + 1][3]       # {-1, 1}
        xyz_delta = action[:3]
        if self.action_mean is not None:
            xyz_delta = (xyz_delta - self.action_mean[:3]) / self.action_std[:3]
        gripper_binary = (abs_grip > 0).float().unsqueeze(0)
        action = torch.cat([xyz_delta, gripper_binary])

        return {"frames": frames, "state": state, "action": action}
