"""Train a GIN model on the polymer dataset with cross-validation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch.utils.data import Subset

from data_generation import PolymerDataset


class GIN(nn.Module):
    """Simple 3-layer GIN network."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()

        def make_block(in_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.conv1 = GINConv(make_block(in_dim))
        self.conv2 = GINConv(make_block(hidden_dim))
        self.conv3 = GINConv(make_block(hidden_dim))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index)
        h = global_add_pool(h, batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h).view(-1)


def train_epoch(model: nn.Module, loader: DataLoader, mean: torch.Tensor,
                std: torch.Tensor, device: torch.device,
                optimizer: torch.optim.Optimizer,
                target_ID: int) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        targets = data.y.view(data.num_graphs, -1)[:, target_ID]
        targets_norm = (targets - mean) / std
        loss = F.mse_loss(out, targets_norm, reduction="mean")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model: nn.Module, loader: DataLoader, mean: torch.Tensor,
             std: torch.Tensor, device: torch.device,
             target_ID: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate the model."""
    model.eval()
    sum_sq = 0.0
    total = 0
    preds = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out_norm = model(data)
            out = out_norm * std + mean
            y = data.y.view(data.num_graphs, -1)[:, target_ID]
            sum_sq += ((out - y) ** 2).sum().item()
            total += data.num_graphs
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    rmse = math.sqrt(sum_sq / total)
    preds_np = np.concatenate(preds)
    targets_np = np.concatenate(targets)
    r2 = r2_score(targets_np, preds_np)
    return rmse, r2, preds_np, targets_np


def main() -> None:
    """Entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train a GIN model on the polymer dataset with cross-validation."
    )
    parser.add_argument(
        "--target",
        choices=["area", "rg", "rdf"],
        default="area",
        help="Target variable to predict.",
    )
    args = parser.parse_args()

    target_map = {"area": 0, "rg": 1, "rdf": 2}
    target_ID = target_map[args.target]

    dataset = PolymerDataset(root=".")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rmse: list[float] = []
    fold_r2: list[float] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        targets = torch.tensor(
            [
                train_subset[i].y.view(-1)[target_ID].item()
                for i in range(len(train_subset))
            ],
            dtype=torch.float32,
        )
        target_mean = targets.mean()
        target_std = targets.std() if targets.std() > 1e-6 else torch.tensor(1.0)

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16)

        model = GIN(in_dim=3, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)

        target_mean = target_mean.to(device)
        target_std = target_std.to(device)

        for _ in range(1500):
            train_epoch(
                model,
                train_loader,
                target_mean,
                target_std,
                device,
                optimizer,
                target_ID,
            )

        rmse, r2, preds_fold, targets_fold = validate(
            model,
            val_loader,
            target_mean,
            target_std,
            device,
            target_ID,
        )
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        all_preds.append(preds_fold)
        all_targets.append(targets_fold)
        print(f"Fold {fold} RMSE: {rmse:.4f}, R2: {r2:.4f}")

    print(f"CV RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")
    print(f"CV R2: {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")

    all_preds_np = np.concatenate(all_preds)
    all_targets_np = np.concatenate(all_targets)

    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets_np, all_preds_np, alpha=0.5)
    lims = [
        min(all_targets_np.min(), all_preds_np.min()),
        max(all_targets_np.max(), all_preds_np.max()),
    ]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted {args.target.capitalize()}")
    plt.tight_layout()
    plt.savefig(Path(f"plots/{args.target}.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
