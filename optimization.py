#!/usr/bin/env python3
"""
Flexible optimizer for backbone length-15 polymers.

Choose objectives:
  --use-area  --area-mode {max,min,target}   [--area-target A*] [--area-band BW] [--area-weight W]
  --use-rg    --rg-mode   {max,min,target}   [--rg-target   R*] [--rg-band   BW] [--rg-weight   W]
  --use-rdf   --rdf-mode  {max,min,target}   [--rdf-target D*] [--rdf-band  BW] [--rdf-weight  W]

Modes:
  max: maximize property          -> score += -W * z
  min: minimize property          -> score += +W * z
  target: aim for value A* (or band) -> score += W * penalty(z, z_target, band)

Assumes models output z-scores; stats JSON gives mean/std for un-normalizing (logging only).

Examples:
  # maximize AREA only
  python3 optimize_flex.py --use-area --area-mode max

  # maximize AREA, minimize RG, maximize RDF
  python3 optimize_flex.py --use-area --area-mode max --use-rg --rg-mode min --use-rdf --rdf-mode max

  # target AREA≈9000 with ±0.2 z band, ignore others
  python3 optimize_flex.py --use-area --area-mode target --area-target 9000 --area-band 0.2
"""

from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cma
import torch
from torch_geometric.data import Batch

from gnn import GIN
from predict import input_to_graph


# ---------- encoding ----------
def vector_to_polymer(vec: List[float]) -> str:
    """Len-15 real vector -> polymer string; sign -> S/E, |.| -> 1..20."""
    out: List[Tuple[int, str]] = []
    for i, v in enumerate(vec, start=1):
        label = "S" if v > 0 else "E"
        n = int(round(abs(v)))
        n = 0 if n < 0 else (20 if n > 20 else n)
        out.append((i, f"{label}{n}"))
    return str(out)


# ---------- inference ----------
@torch.no_grad()
def predict_z(model: torch.nn.Module, device: torch.device, polymer_str: str) -> float:
    data = input_to_graph(polymer_str)
    batch = Batch.from_data_list([data]).to(device)
    y = model(batch)
    return float(y.item())  # assumed standardized (z)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = GIN(in_dim=3, hidden_dim=128).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------- objective builder ----------
def hinge_band_penalty(z: float, z_target: float, band: float) -> float:
    """
    Zero penalty inside |z - z_target| <= band.
    Outside, squared relative violation ((|dz| - band)/max(band,1e-9))^2.
    """
    if band <= 0:
        dz = z - z_target
        return dz * dz
    gap = abs(z - z_target) - band
    if gap <= 0:
        return 0.0
    denom = band if band > 1e-9 else 1e-9
    t = gap / denom
    return t * t


class Term:
    def __init__(self, name: str, use: bool, mode: str, weight: float,
                 target_raw: Optional[float], band: float,
                 stats: Dict[str, float],
                 model: Optional[torch.nn.Module]):
        self.name = name
        self.use = use
        self.mode = mode  # 'max'|'min'|'target'
        self.weight = weight
        self.target_raw = target_raw
        self.band = band
        self.mean = stats.get("mean", 0.0)
        self.std  = max(stats.get("std", 1.0), 1e-9)
        self.model = model

    def need_model(self) -> bool:
        return self.use and self.model is not None

    def score_contrib(self, polymer: str, device: torch.device) -> Tuple[float, float, float]:
        """
        Returns (score_contribution, z, raw_value)
        """
        if not self.use or self.model is None:
            return (0.0, float("nan"), float("nan"))

        z = predict_z(self.model, device, polymer)
        raw = z * self.std + self.mean

        if self.mode == "max":
            contrib = -self.weight * z
        elif self.mode == "min":
            contrib = +self.weight * z
        elif self.mode == "target":
            if self.target_raw is None:
                raise ValueError(f"{self.name}: target mode requires --{self.name}-target")
            z_target = (self.target_raw - self.mean) / self.std
            contrib = self.weight * hinge_band_penalty(z, z_target, self.band)
        else:
            raise ValueError(f"Unknown mode for {self.name}: {self.mode}")

        return (contrib, z, raw)


def make_objective(terms: List[Term], device: torch.device,
                   log: List[Tuple[str, float, float, float, float]]):

    def objective(vec: List[float]) -> float:
        polymer = vector_to_polymer(vec)
        total = 0.0

        z_area = raw_area = float("nan")
        z_rg   = raw_rg   = float("nan")
        z_rdf  = raw_rdf  = float("nan")

        for t in terms:
            contrib, z, raw = t.score_contrib(polymer, device)
            total += contrib
            if t.name == "area":
                z_area, raw_area = z, raw
            elif t.name == "rg":
                z_rg, raw_rg = z, raw
            elif t.name == "rdf":
                z_rdf, raw_rdf = z, raw

        # log: polymer, AREA(raw), RG(raw), RDF(raw), score
        log.append((polymer, raw_area, raw_rg, raw_rdf, total))
        return float(total)

    return objective


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=15)

    # Which properties to use
    ap.add_argument("--use-area", action="store_true")
    ap.add_argument("--use-rg",   action="store_true")
    ap.add_argument("--use-rdf",  action="store_true")

    # Modes
    ap.add_argument("--area-mode", choices=["max","min","target"], default="min")
    ap.add_argument("--rg-mode",   choices=["max","min","target"], default="min")
    ap.add_argument("--rdf-mode",  choices=["max","min","target"], default="max")

    # Weights
    ap.add_argument("--area-weight", type=float, default=1.0)
    ap.add_argument("--rg-weight",   type=float, default=1.0)
    ap.add_argument("--rdf-weight",  type=float, default=1.0)

    # Targets (raw units) + bands (in z-units)
    ap.add_argument("--area-target", type=float)
    ap.add_argument("--rg-target",   type=float)
    ap.add_argument("--rdf-target",  type=float)
    ap.add_argument("--area-band",   type=float, default=0.0)
    ap.add_argument("--rg-band",     type=float, default=0.0)
    ap.add_argument("--rdf-band",    type=float, default=0.0)

    # Models / stats
    ap.add_argument("--model-area", type=str, default="models/model_area.pt")
    ap.add_argument("--model-rg",   type=str, default="models/model_rg.pt")
    ap.add_argument("--model-rdf",  type=str, default="models/model_rdf.pt")
    ap.add_argument("--stats-json", type=str, default="models/normalization_stats.json")

    # CMA-ES options (deeper defaults)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--popsize", type=int, default=28)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--incpopsize", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.length != 15:
        print(f"Warning: requested length={args.length}; continuing anyway.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load stats
    with Path(args.stats_json).open() as f:
        stats_all = json.load(f)
    s_area = stats_all.get("area", {"mean":0.0,"std":1.0})
    s_rg   = stats_all.get("rg",   {"mean":0.0,"std":1.0})
    s_rdf  = stats_all.get("rdf",  {"mean":0.0,"std":1.0})

    # Conditionally load models
    model_area = load_model(Path(args.model_area), device) if args.use_area else None
    model_rg   = load_model(Path(args.model_rg),   device) if args.use_rg   else None
    model_rdf  = load_model(Path(args.model_rdf),  device) if args.use_rdf  else None

    # Build terms
    terms: List[Term] = []
    terms.append(Term("area", args.use_area, args.area_mode, args.area_weight,
                      args.area_target, args.area_band, s_area, model_area))
    terms.append(Term("rg",   args.use_rg,   args.rg_mode,   args.rg_weight,
                      args.rg_target,   args.rg_band,   s_rg,   model_rg))
    terms.append(Term("rdf",  args.use_rdf,  args.rdf_mode,  args.rdf_weight,
                      args.rdf_target,  args.rdf_band,  s_rdf,  model_rdf))

    # Sanity: at least one active term
    if not any(t.use for t in terms):
        raise SystemExit("No objectives enabled. Use --use-area/--use-rg/--use-rdf.")

    # Objective + log
    log: List[Tuple[str, float, float, float, float]] = []
    objective = make_objective(terms, device, log)

    # CMA-ES run with restarts
    options = {
        "seed": args.seed,
        "verb_disp": 1,
        "timeout": args.timeout,
        "popsize": args.popsize,
        "maxiter": args.maxiter,
        "bounds": [-20, 20],
        "CMA_elitist": True,
        "CMA_active": True,
        "tolx": 1e-12,
        "tolfun": 1e-12,
        "tolfunhist": 1e-12,
    }

    x0 = [0.0] * args.length
    xopt, es = cma.fmin2(
        objective,
        x0,
        args.sigma,
        options=options,
        restarts=args.restarts,
        incpopsize=args.incpopsize,
        eval_initial_x=True,
    )

    best_polymer = vector_to_polymer(xopt)

    # Final report: compute raw(z->units) per enabled term
    def eval_term_raw(t: Term) -> Tuple[float, float]:
        if not t.use or t.model is None:
            return (float("nan"), float("nan"))
        z = predict_z(t.model, device, best_polymer)
        raw = z * t.std + t.mean
        return (z, raw)

    zA, Araw = eval_term_raw(terms[0])
    zG, Graw = eval_term_raw(terms[1])
    zD, Draw = eval_term_raw(terms[2])

    # Recompute score for best
    final_score = 0.0
    for t in terms:
        contrib, _, _ = t.score_contrib(best_polymer, device)
        final_score += contrib

    print("\n=== Best candidate (across restarts) ===")
    print(f"Backbone length: {args.length}")
    print(f"Polymer: {best_polymer}")
    if terms[0].use: print(f"AREA: {Araw:.6g}  [z={zA:.3f}]  mode={terms[0].mode}")
    if terms[1].use: print(f"RG:   {Graw:.6g}  [z={zG:.3f}]  mode={terms[1].mode}")
    if terms[2].use: print(f"RDF:  {Draw:.6g}  [z={zD:.3f}]  mode={terms[2].mode}")
    print(f"Score (lower better): {final_score:.4f}")

    # Save log
    out_dir = Path("log"); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"flex_len{args.length}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["polymer","AREA","RG","RDF","score"])
        w.writerows(log)
    print(f"\nLog saved to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
