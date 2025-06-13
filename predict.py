import argparse
import ast
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data, Batch

from gnn import GIN


def input_to_graph(polymer_str: str) -> Data:
    input_list: List[Tuple[int, str]] = ast.literal_eval(polymer_str)

    features = []
    backbone_indices = {}
    for idx, (n, label) in enumerate(input_list):
        features.append([1, 0, 0])
        backbone_indices[n] = idx

    edge_index = [[], []]
    for i in range(len(input_list) - 1):
        edge_index[0].append(i)
        edge_index[1].append(i + 1)
        edge_index[0].append(i + 1)
        edge_index[1].append(i)

    next_node = len(features)
    for n, label in input_list:
        if label not in ("E0", "S0"):
            m = int(label[1:])
            bead_type = label[0]
            prev = backbone_indices[n]
            for _ in range(m):
                features.append([0, 1, 0] if bead_type == "S" else [0, 0, 1])
                curr = next_node
                edge_index[0].append(prev)
                edge_index[1].append(curr)
                edge_index[0].append(curr)
                edge_index[1].append(prev)
                prev = curr
                next_node += 1

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict polymer properties")
    parser.add_argument("polymer", help="Polymer input list string")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_map = {"area": 0, "rg": 2, "rdf": 4, "coor": 5}

    with Path("models/normalization_stats.json").open() as f:
        stats = json.load(f)

    models = {}
    for name in target_map:
        model = GIN(in_dim=3, hidden_dim=128).to(device)
        state_dict = torch.load(f"models/model_{name}.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[name] = model

    data = input_to_graph(args.polymer)
    batch = Batch.from_data_list([data]).to(device)

    predictions = {}
    with torch.no_grad():
        for name, idx in target_map.items():
            out = models[name](batch)
            mean = stats[name]["mean"]
            std = stats[name]["std"]
            pred = out.item() * std + mean
            predictions[name] = pred

    for name, value in predictions.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
