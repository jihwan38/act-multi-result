#!/usr/bin/env python3
"""Plot the completed controlled-graph measurements (requires matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/multipattern/results/aggregate.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/multipattern/results/joint-pass.svg"))
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.input.read_text())
    summary = evidence["measurement_sets"]["micro/measurements/final-default"]["summary"]
    modes = [("multi-result", "main", "#485563"),
             ("multipattern", "bounded discovery", "#0072B2"),
             ("multipattern-rewrite", "direct rewrite", "#D55E00")]
    order = {"shared": 0, "metadata": 1, "independent": 2, "disjoint": 3,
             "dependent": 4, "wrong-shape": 5}
    summary.sort(key=lambda item: (order[item["scenario"]], item["size"]))
    figure, axis = plt.subplots(figsize=(10, 8.4))
    for index, (mode, label, color) in enumerate(modes):
        positions, values, lows, highs = [], [], [], []
        for row, item in enumerate(summary):
            result = item["variants"][mode]
            if result["passed"] != result["samples"] or result["samples"] < 5:
                raise ValueError(f"Incomplete or failed measurements: {item['scenario']} {mode}")
            metric = result["metrics"]["joint_candidate_pass_ns"]
            positions.append(row + (index - 1) * 0.22)
            values.append(metric["median"] / 1e6)
            lows.append((metric["median"] - metric["min"]) / 1e6)
            highs.append((metric["max"] - metric["median"]) / 1e6)
        axis.errorbar(values, positions, xerr=[lows, highs], fmt="o", markersize=5,
                      capsize=3, elinewidth=1.2, color=color, label=label)
    axis.set_yticks(range(len(summary)), [f"{item['scenario']}  N={item['size']}" for item in summary])
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("One joint candidate pass (ms, log scale)")
    axis.grid(axis="x", alpha=0.22, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", bbox_to_anchor=(0, 1.09), ncol=3, frameon=False)
    figure.suptitle("Joint search and application on fixed e-graphs", x=0.08, ha="left", fontsize=15)
    figure.text(0.08, 0.925, "Median and min/max of five runs; candidate coverage differs between variants.", fontsize=10)
    figure.text(0.08, 0.025, "Excludes graph construction, IR rewrite iterations, extraction, allocation and Oracle.\n"
                "Compare coverage in REPORT.md before interpreting time differences.", fontsize=9)
    figure.subplots_adjust(left=0.24, right=0.97, top=0.84, bottom=0.13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={"Date": None})
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.preview, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
