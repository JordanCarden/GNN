import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data_generation import PolymerDataset
from gnn import GIN, train_epoch


def main() -> None:
    target_map = {"area": 0, "rg": 2, "rdf": 4, "coor": 5}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = {}
    for name, idx in target_map.items():
        print(f"Training model for '{name}'...")
        dataset = PolymerDataset(root=".")
        targets = torch.tensor(
            [dataset[i].y.view(-1)[idx].item() for i in range(len(dataset))],
            dtype=torch.float32,
        )
        mean = targets.mean()
        std = targets.std() if targets.std() > 1e-6 else torch.tensor(1.0)

        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        model = GIN(in_dim=3, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=5e-5, weight_decay=5e-4
        )

        mean = mean.to(device)
        std = std.to(device)

        for _ in range(1500):
            train_epoch(model, loader, mean, std, device, optimizer, idx)

        torch.save(model.state_dict(), f"model_{name}.pt")
        stats[name] = {"mean": mean.item(), "std": std.item()}

    with Path("normalization_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    main()
