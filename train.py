import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data_generation import PolymerDataset
from gnn import GIN, train_epoch


def main() -> None:
    target_map = {"area": 0, "rg": 1, "rdf": 2}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_dir = Path(__file__).resolve().parent
    models_dir = repo_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for name, idx in target_map.items():
        print(f"Training model for '{name}'...")
        dataset = PolymerDataset(root=str(repo_dir))
        targets = torch.tensor(
            [dataset[i].y.view(-1)[idx].item() for i in range(len(dataset))],
            dtype=torch.float32,
        )
        mean = targets.mean()
        std = targets.std() if targets.std() > 1e-6 else torch.tensor(1.0)

        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        model = GIN(in_dim=3, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1500,
            eta_min=1e-5,
        )

        mean = mean.to(device)
        std = std.to(device)

        for _ in range(1500):
            train_epoch(model, loader, mean, std, device, optimizer, idx)
            scheduler.step()

        torch.save(model.state_dict(), models_dir / f"model_{name}.pt")
        stats[name] = {"mean": float(mean.item()), "std": float(std.item())}

    with (models_dir / "normalization_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
