"""
calculate_threshold.py - Compute the RouteLLM routing threshold via Pareto frontier + geometric elbow.

For every unique toughness score as a candidate threshold, we compute the strong-millis fraction
(strong-model inference time actually spent / strong-model inference time if everything went to strong)
and the resulting accuracy when both models' pass/fail outcomes are taken into account. This produces a
Pareto frontier of (cost_fraction, accuracy) points, bounded [0, 1]. The optimal threshold is the
elbow of that curve — the point of maximum perpendicular distance from the chord connecting the two
frontier endpoints. No tuning parameter is required.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from kneed import KneeLocator

import daos.model_task_result_dao as model_task_result_dao
import daos.tasks_dao as tasks_dao


@dataclass
class ThresholdPoint:
    threshold: float
    cost_fraction: float  # strong millis spent / total strong millis — range [0, 1]
    accuracy: float


def build_frontier(
    split_id: int,
    weak_model_name: str,
    strong_model_name: str,
    is_test: bool,
) -> list[ThresholdPoint]:
    task_score: dict[str, float] = {
        t.id: t.toughness_score
        for t in tasks_dao.get_all_for_split(split_id, is_test)
        if t.toughness_score is not None
    }
    if not task_score:
        raise ValueError(f"No toughness scores found for split_id={split_id}")

    weak_results = {
        r.task_id: r
        for r in model_task_result_dao.get_all_for_model_split(weak_model_name, split_id, is_test)
        if r.passed is not None and r.run_millis is not None
    }
    strong_results = {
        r.task_id: r
        for r in model_task_result_dao.get_all_for_model_split(strong_model_name, split_id, is_test)
        if r.passed is not None and r.run_millis is not None
    }

    task_ids = set(task_score) & set(weak_results) & set(strong_results)
    if not task_ids:
        raise ValueError(
            f"No tasks have toughness scores, pass labels, and run_millis for both models "
            f"(split_id={split_id}, weak={weak_model_name}, strong={strong_model_name})"
        )

    total_strong_millis = sum(strong_results[t].run_millis for t in task_ids)
    candidate_thresholds = sorted(
        {task_score[t] for t in task_ids}, reverse=True
    )

    frontier: list[ThresholdPoint] = []
    for threshold in candidate_thresholds:
        strong_ids = {t for t in task_ids if task_score[t] >= threshold}
        weak_ids = task_ids - strong_ids

        cost_fraction = (
            sum(strong_results[t].run_millis for t in strong_ids) +
            sum(weak_results[t].run_millis for t in weak_ids)
        ) / total_strong_millis
        correct = (
            sum(strong_results[t].passed for t in strong_ids)
            + sum(weak_results[t].passed for t in weak_ids)
        )
        frontier.append(ThresholdPoint(
            threshold=threshold,
            cost_fraction=cost_fraction,
            accuracy=correct / len(task_ids),
        ))

    return frontier


def find_elbow(frontier: list[ThresholdPoint]) -> tuple[ThresholdPoint, KneeLocator]:
    # Deduplicate points with the same cost_fraction (keep highest accuracy).
    # Duplicate x-values cause KneeLocator normalization to break silently.
    seen: dict[float, ThresholdPoint] = {}
    for p in frontier:
        if p.cost_fraction not in seen or p.accuracy > seen[p.cost_fraction].accuracy:
            seen[p.cost_fraction] = p
    deduped = sorted(seen.values(), key=lambda p: p.cost_fraction)

    pareto = []
    best_acc = -1

    for p in reversed(deduped):
        if p.accuracy > best_acc:
            pareto.append(p)
            best_acc = p.accuracy

    pareto = list(reversed(pareto))

    knee = KneeLocator(
        x=[p.cost_fraction for p in pareto],
        y=[p.accuracy      for p in pareto],
        curve="convex",
        direction="increasing",
    )

    if knee.knee is None:
        # No elbow found — pick the point of maximum marginal accuracy gain per
        # unit cost increase as a sensible fallback.
        best = max(
            (deduped[i].accuracy - deduped[i - 1].accuracy)
            / max(deduped[i].cost_fraction - deduped[i - 1].cost_fraction, 1e-9)
            for i in range(1, len(deduped))
        )
        best_idx = next(
            i for i in range(1, len(deduped))
            if (deduped[i].accuracy - deduped[i - 1].accuracy)
               / max(deduped[i].cost_fraction - deduped[i - 1].cost_fraction, 1e-9) == best
        )
        print("WARNING: KneeLocator found no elbow. Falling back to max-marginal-gain point.")
        return deduped[best_idx], knee

    return min(deduped, key=lambda p: abs(p.cost_fraction - knee.knee)), knee


def _save_plot(
    frontier: list[ThresholdPoint],
    elbow: ThresholdPoint,
    split_id: int,
) -> None:
    out_dir = Path("visuals")
    out_dir.mkdir(exist_ok=True)

    cost_fractions = [p.cost_fraction for p in frontier]
    accuracies     = [p.accuracy      for p in frontier]

    plt.figure(figsize=(7, 5))
    plt.plot(cost_fractions, accuracies, marker="o", markersize=3, linewidth=1.5)
    plt.axvline(
        elbow.cost_fraction, color="red", linestyle="--",
        label=f"elbow (threshold={elbow.threshold:.4f})",
    )
    plt.xlabel("Cost fraction (spent millis / all-strong millis)")
    plt.ylabel("Accuracy")
    plt.title(f"Pareto Frontier — Split {split_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"pareto_frontier_split{split_id}.png"
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to visuals/{filename}")


def calculate_threshold(
    split_id: int,
    weak_model_name: str,
    strong_model_name: str,
    is_test: bool = False,
) -> float:
    """Compute the routing threshold via Pareto frontier + geometric elbow.

    Args:
        split_id:          DB split to pull data from.
        weak_model_name:   Name of the weak model (must have results in model_task_result).
        strong_model_name: Name of the strong model (must have results in model_task_result).
        is_test:           Whether to use the test partition (default: train).

    Returns:
        Threshold float. Problems with toughness_score >= threshold are routed to strong.
    """
    frontier = build_frontier(split_id, weak_model_name, strong_model_name, is_test)
    elbow, knee = find_elbow(frontier)
    _save_plot(frontier, elbow, split_id)

    a, b       = frontier[0], frontier[-1]
    scores     = [p.threshold for p in frontier]
    accuracies = [p.accuracy  for p in frontier]
    print(
        f"Frontier: {len(frontier)} points  "
        f"[cost_fraction {a.cost_fraction:.1%}→{b.cost_fraction:.1%}, "
        f"accuracy {a.accuracy:.1%}→{b.accuracy:.1%}]"
    )
    print(
        f"Toughness scores: min={min(scores):.4f}  max={max(scores):.4f}  "
        f"range={max(scores)-min(scores):.4f}"
    )
    print(
        f"Accuracy range:   min={min(accuracies):.1%}  max={max(accuracies):.1%}  "
        f"spread={max(accuracies)-min(accuracies):.1%}"
    )
    print(f"KneeLocator knee: {knee.knee}")
    print(
        f"Elbow:    threshold={elbow.threshold:.5f}  "
        f"cost_fraction={elbow.cost_fraction:.1%}  "
        f"accuracy={elbow.accuracy:.1%}"
    )
    return elbow.threshold
