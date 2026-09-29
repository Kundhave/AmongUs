"""Benchmark A* against BFS, Dijkstra and greedy over all 182 ordered room pairs (SPEC §8.5)."""

import csv
import itertools
from pathlib import Path

from amongus.search.astar import astar
from amongus.search.baselines import bfs, dijkstra, greedy
from amongus.search.costs import plain_cost_fn
from amongus.world.map import ROOMS, GraphView

_ALGORITHMS = {"astar": astar, "dijkstra": dijkstra, "bfs": bfs, "greedy": greedy}
_OUT_DIR = Path("runs/bench")
_OUT_CSV = _OUT_DIR / "search.csv"
_EPS = 1e-9  # float-equality tolerance when comparing path costs


def run_bench() -> list[dict[str, object]]:
    """Run every algorithm on every ordered room pair; return one row dict per (algo, pair)."""
    view = GraphView()
    rows: list[dict[str, object]] = []
    for src, dst in itertools.permutations(ROOMS, 2):
        optimal_cost = astar(view, src, dst, plain_cost_fn)[1]
        for name, fn in _ALGORITHMS.items():
            path, cost, stats = fn(view, src, dst, plain_cost_fn)
            rows.append(
                {
                    "algorithm": name,
                    "src": src,
                    "dst": dst,
                    "cost": cost,
                    "expanded": stats.expanded,
                    "frontier_max": stats.frontier_max,
                    "optimal": abs(cost - optimal_cost) < _EPS,
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    """Write per-(algorithm, pair) rows to runs/bench/search.csv, creating the dir if needed."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, object]]) -> None:
    """Print a Markdown summary table: mean cost, mean expansions, optimal-on-all-pairs."""
    print("| algorithm | mean path cost | mean expansions | optimal on all pairs |")
    print("|---|---|---|---|")
    for name in _ALGORITHMS:
        algo_rows = [r for r in rows if r["algorithm"] == name]
        mean_cost = sum(r["cost"] for r in algo_rows) / len(algo_rows)
        mean_expanded = sum(r["expanded"] for r in algo_rows) / len(algo_rows)
        all_optimal = all(r["optimal"] for r in algo_rows)
        optimal_str = "yes" if all_optimal else "no"
        print(f"| {name} | {mean_cost:.3f} | {mean_expanded:.2f} | {optimal_str} |")


def main() -> None:
    """Run the bench, write the CSV, and print the Markdown table."""
    rows = run_bench()
    write_csv(rows)
    print_table(rows)


if __name__ == "__main__":
    main()
