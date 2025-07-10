#!/usr/bin/env python3
"""Optimize polymers for minimal predicted RDF using CMA-ES."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cma
import torch
from torch_geometric.data import Batch

from gnn import GIN
from predict import input_to_graph


def vector_to_polymer(vector: List[float]) -> str:
    """Convert numerical vector to polymer input list string."""
    input_list: List[Tuple[int, str]] = []
    for idx, value in enumerate(vector, start=1):
        label_type = "S" if value > 0 else "E"
        length = int(round(abs(value)))
        length = min(max(length, 0), 20)
        input_list.append((idx, f"{label_type}{length}"))
    return str(input_list)


def make_objective(
    model: GIN, device: torch.device, stats: Dict[str, Dict[str, float]],
    log: List[Tuple[str, float]]
):
    """Create objective function that logs polymers and predicted RDF."""

    def objective(vector: List[float]) -> float:
        polymer = vector_to_polymer(vector)
        data = input_to_graph(polymer)
        batch = Batch.from_data_list([data]).to(device)
        with torch.no_grad():
            out = model(batch)
        rdf = out.item() * stats["rdf"]["std"] + stats["rdf"]["mean"]
        log.append((polymer, rdf))
        return rdf

    return objective


def optimize_for_length(
    length: int,
    model: GIN,
    device: torch.device,
    stats: Dict[str, Dict[str, float]],
    max_iter: int,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Run CMA-ES optimization for a specific backbone length.

    Args:
        length: Polymer backbone length.
        model: Trained GIN model used for predictions.
        device: Torch device for computation.
        stats: Normalization statistics for RDF.
        max_iter: Maximum number of CMA-ES iterations.

    Returns:
        Tuple of best polymer string, best RDF value, and the evaluation log.
    """
    log: List[Tuple[str, float]] = []
    objective = make_objective(model, device, stats, log)
    es = cma.CMAEvolutionStrategy(
        [0.0] * length,
        15,
        {
            "verb_disp": 0,
            "maxiter": max_iter,
        },
    )
    es.optimize(objective)
    best_vector = es.result.xbest
    best_polymer = vector_to_polymer(best_vector)
    best_value = es.result.fbest
    return best_polymer, float(best_value), log


def main() -> None:
    """Run CMA-ES search and save logs to CSV files."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_ITERATIONS = 7

    with Path("models/normalization_stats.json").open() as f:
        stats = json.load(f)

    model = GIN(in_dim=3, hidden_dim=128).to(device)
    state_dict = torch.load("models/model_rdf.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    best_overall: Tuple[str, int, float] | None = None
    logs_by_length: Dict[int, List[Tuple[str, float]]] = {}
    best_by_length: Dict[int, Tuple[str, float]] = {}

    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    for length in range(10, 21):
        best_polymer, best_value, logs = optimize_for_length(
            length, model, device, stats, MAX_ITERATIONS
        )
        log_path = log_dir / f"rdf_log_length_{length}.csv"
        with log_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Polymer", "RDF"])
            writer.writerows(logs)
        logs_by_length[length] = logs
        best_by_length[length] = (best_polymer, best_value)
        if best_overall is None or best_value < best_overall[2]:
            best_overall = (best_polymer, length, best_value)

    for length in sorted(best_by_length):
        polymer, value = best_by_length[length]
        print(f"Length {length}: best RDF {value:.4f} for polymer {polymer}")

    if best_overall is not None:
        polymer, length, value = best_overall
        print(
            (
                f"\nBest overall polymer: {polymer}\nBackbone length: {length}"
                f"\nPredicted RDF: {value:.4f}"
            )
        )


if __name__ == "__main__":
    main()
