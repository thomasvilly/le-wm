"""
V-JEPA 2-AC frozen encoder + MLP action head for 4-DOF robot arm.

Architecture:
    frozen V-JEPA 2-AC encoder  (ViT-g, ~1B params, not updated)
    → mean-pool patch tokens     → (B, ENCODER_DIM)
    → Linear + LayerNorm         → (B, embed_dim)          [projection]
    → concat robot state         → (B, embed_dim + STATE_DIM)
    → MLP (depth × hidden_dim)   → (B, ACTION_DIM)         [action head]

Feature caching (--cache_features):
    All windows are pushed through the frozen encoder once, features saved to disk.
    Subsequent runs load from cache — training becomes a lightweight CPU MLP loop.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import RobotEpisodeDataset

ENCODER_DIM = 1408  # ViT-g/16 hidden dimension
STATE_DIM   = 4     # dims [x, y, z, gripper] after ACTION_DIMS filtering
ACTION_DIM  = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActionHead(nn.Module):
    def __init__(self, embed_dim, mlp_hidden, mlp_depth):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ENCODER_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.mlp = MLP(embed_dim + STATE_DIM, mlp_hidden, ACTION_DIM, mlp_depth)

    def forward(self, features, state):
        """
        features : (B, ENCODER_DIM)  — mean-pooled encoder output
        state    : (B, STATE_DIM)    — normalised robot state
        returns  : (B, ACTION_DIM)
        """
        x = self.proj(features)
        x = torch.cat([x, state], dim=-1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

def load_encoder(device):
    """Load frozen V-JEPA 2-AC encoder from torch.hub."""
    # Returns (encoder, predictor); predictor is discarded
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_frames(encoder, frames):
    """
    frames  : (B, 8, 3, 256, 256)
    returns : (B, ENCODER_DIM)

    V-JEPA 2 expects (B, C, T, H, W). Encoder returns patch tokens
    (B, N_tokens, D); we mean-pool over tokens to get one vector per clip.
    """
    x      = frames.permute(0, 2, 1, 3, 4)  # (B, 3, 8, 256, 256)
    tokens = encoder(x)                       # (B, N_tokens, ENCODER_DIM)
    return tokens.mean(dim=1)                 # (B, ENCODER_DIM)


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------

CACHE_FILE = "feature_cache.pt"


def build_feature_cache(encoder, data_dir, device, batch_size):
    """
    Run every dataset window through the frozen encoder once and save to disk.
    Uses val_fraction=0.0 so every window is included (split happens at load time).
    """
    s_mean, s_std, a_mean, a_std = RobotEpisodeDataset.fit_normalizers(data_dir)
    full_ds = RobotEpisodeDataset(
        data_dir, split="train", val_fraction=0.0,
        state_mean=s_mean, state_std=s_std,
        action_mean=a_mean, action_std=a_std,
    )
    loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))

    all_feats, all_states, all_actions = [], [], []
    print(f"Extracting features for {len(full_ds)} windows …")
    for i, batch in enumerate(loader):
        feats = encode_frames(encoder, batch["frames"].to(device))
        all_feats.append(feats.cpu())
        all_states.append(batch["state"])
        all_actions.append(batch["action"])
        if (i + 1) % 20 == 0:
            print(f"  {(i + 1) * batch_size}/{len(full_ds)}")

    cache = {
        "features": torch.cat(all_feats),
        "states":   torch.cat(all_states),
        "actions":  torch.cat(all_actions),
        "s_mean": s_mean, "s_std": s_std,
        "a_mean": a_mean, "a_std": a_std,
    }
    path = os.path.join(data_dir, CACHE_FILE)
    torch.save(cache, path)
    print(f"Cache saved → {path}  "
          f"({cache['features'].shape[0]} windows, dim={cache['features'].shape[1]})")
    return cache


def load_cached_datasets(data_dir, val_fraction, seed):
    cache   = torch.load(os.path.join(data_dir, CACHE_FILE), weights_only=False)
    N       = cache["features"].shape[0]
    g       = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=g)
    n_val   = max(1, int(N * val_fraction))

    def subset(idx):
        return TensorDataset(
            cache["features"][idx],
            cache["states"][idx],
            cache["actions"][idx],
        )

    return subset(indices[n_val:]), subset(indices[:n_val]), cache


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _get_features_state_action(encoder, batch, device):
    """Unpack a batch from either the live or cached path."""
    if isinstance(batch, (list, tuple)):
        # TensorDataset yields (features, state, action)
        feats, state, action = batch
        return feats.to(device), state.to(device), action.to(device)
    feats = encode_frames(encoder, batch["frames"].to(device))
    return feats, batch["state"].to(device), batch["action"].to(device)


def _epoch(head, encoder, loader, device, loss_fn, opt=None):
    """Single train or eval epoch. Pass opt=None for eval."""
    total = xyz_total = grip_total = 0.0
    for batch in loader:
        feats, state, action = _get_features_state_action(encoder, batch, device)
        pred = head(feats, state)
        loss = loss_fn(pred, action)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total      += loss.item()
        xyz_total  += loss_fn(pred[:, :3], action[:, :3]).item()
        grip_total += loss_fn(pred[:, 3:], action[:, 3:]).item()
    n = len(loader)
    return total / n, xyz_total / n, grip_total / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = None

    if args.cache_features:
        cache_path = os.path.join(args.data_dir, CACHE_FILE)
        if not os.path.exists(cache_path):
            encoder = load_encoder(device)
            build_feature_cache(encoder, args.data_dir, device, args.batch_size)
            del encoder
            encoder = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_ds, val_ds, cache = load_cached_datasets(
            args.data_dir, args.val_fraction, args.seed
        )
        s_mean, s_std = cache["s_mean"], cache["s_std"]
        a_mean, a_std = cache["a_mean"], cache["a_std"]
        train_loader  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
        val_loader    = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    else:
        encoder = load_encoder(device)
        s_mean, s_std, a_mean, a_std = RobotEpisodeDataset.fit_normalizers(args.data_dir)
        train_ds = RobotEpisodeDataset(
            args.data_dir, split="train", val_fraction=args.val_fraction, seed=args.seed,
            state_mean=s_mean, state_std=s_std, action_mean=a_mean, action_std=a_std,
        )
        val_ds = RobotEpisodeDataset(
            args.data_dir, split="val", val_fraction=args.val_fraction, seed=args.seed,
            state_mean=s_mean, state_std=s_std, action_mean=a_mean, action_std=a_std,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} windows  |  Val: {len(val_ds)} windows")

    head = ActionHead(args.embed_dim, args.mlp_hidden, args.mlp_depth).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Action head: {n_params:,} params  "
          f"(embed_dim={args.embed_dim}, mlp_hidden={args.mlp_hidden}, depth={args.mlp_depth})")

    opt       = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn   = nn.MSELoss()

    best_val  = float("inf")
    ckpt_path = os.path.join(
        args.data_dir,
        f"ckpt_e{args.embed_dim}_h{args.mlp_hidden}_d{args.mlp_depth}.pt",
    )

    for epoch in range(1, args.epochs + 1):
        head.train()
        trn_loss, _, _ = _epoch(head, encoder, train_loader, device, loss_fn, opt)

        head.eval()
        with torch.no_grad():
            val_loss, val_xyz, val_grip = _epoch(head, encoder, val_loader, device, loss_fn)

        scheduler.step()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "head_state_dict": head.state_dict(),
                "embed_dim": args.embed_dim,
                "mlp_hidden": args.mlp_hidden,
                "mlp_depth": args.mlp_depth,
                "s_mean": s_mean, "s_std": s_std,
                "a_mean": a_mean, "a_std": a_std,
            }, ckpt_path)
            marker = " ← best"

        print(f"[{epoch:03d}/{args.epochs}]  "
              f"train={trn_loss:.4f}  val={val_loss:.4f}  "
              f"(xyz={val_xyz:.4f}  grip={val_grip:.4f}){marker}")

    print(f"\nBest val loss: {best_val:.4f}  →  {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train action head on frozen V-JEPA 2-AC features")

    p.add_argument("--data_dir",       default=".",   help="Directory with episode .h5 files")
    p.add_argument("--cache_features", action="store_true",
                   help="Pre-extract encoder features once and cache to disk (recommended)")

    # Ablation knobs
    p.add_argument("--embed_dim",  type=int, default=256,
                   help="Project 1408 → embed_dim before MLP  (try 64 / 128 / 256 / 512)")
    p.add_argument("--mlp_hidden", type=int, default=256,
                   help="Hidden dim of action MLP              (try 128 / 256 / 512)")
    p.add_argument("--mlp_depth",  type=int, default=2,
                   help="Number of hidden layers               (try 1 / 2 / 3)")

    # Training
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)

    train(p.parse_args())
