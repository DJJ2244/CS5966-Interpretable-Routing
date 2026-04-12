"""
calculate_threshold.py - Compute the RouteLLM routing threshold via Pareto frontier + geometric elbow.

For every unique toughness score as a candidate threshold, we compute the fraction of problems
routed to the strong model (strong_rate) and the resulting accuracy when both models' pass/fail
outcomes are taken into account. This produces a Pareto frontier of (strong_rate, accuracy) points.
The optimal threshold is the elbow of that curve — the point of maximum perpendicular distance
from the chord connecting the two frontier endpoints. No tuning parameter is required.
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
    strong_rate: float
    accuracy: float


def _build_frontier(
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
        raise ValueError(f"No toughness scores found for split_id={split_id}, is_test={is_test}")

    weak_pass: dict[str, bool] = {
        r.task_id: r.passed
        for r in model_task_result_dao.get_all_for_model_split(weak_model_name, split_id, is_test)
        if r.passed is not None
    }
    strong_pass: dict[str, bool] = {
        r.task_id: r.passed
        for r in model_task_result_dao.get_all_for_model_split(strong_model_name, split_id, is_test)
        if r.passed is not None
    }

    task_ids = set(task_score) & set(weak_pass) & set(strong_pass)
    if not task_ids:
        raise ValueError(
            f"No tasks have toughness scores and results for both models "
            f"(split_id={split_id}, weak={weak_model_name}, strong={strong_model_name})"
        )

    total = len(task_ids)
    candidate_thresholds = sorted(
        {task_score[t] for t in task_ids}, reverse=True
    )

    frontier: list[ThresholdPoint] = []
    for threshold in candidate_thresholds:
        strong_ids = {t for t in task_ids if task_score[t] >= threshold}
        weak_ids = task_ids - strong_ids
        correct = sum(strong_pass[t] for t in strong_ids) + sum(weak_pass[t] for t in weak_ids)
        frontier.append(ThresholdPoint(
            threshold=threshold,
            strong_rate=len(strong_ids) / total,
            accuracy=correct / total,
        ))

    return frontier


def _find_elbow(frontier: list[ThresholdPoint]) -> ThresholdPoint:
    if len(frontier) <= 2:
        return frontier[0]

    knee = KneeLocator(
        x=[p.strong_rate for p in frontier],
        y=[p.accuracy    for p in frontier],
        curve="concave",
        direction="increasing",
    )

    if knee.knee is None:
        return frontier[0]

    return min(frontier, key=lambda p: abs(p.strong_rate - knee.knee))


def _save_plot(
    frontier: list[ThresholdPoint],
    elbow: ThresholdPoint,
    split_id: int,
    weak_model_name: str,
    strong_model_name: str,
) -> None:
    out_dir = Path("visuals")
    out_dir.mkdir(exist_ok=True)

    xs = [p.strong_rate for p in frontier]
    ys = [p.accuracy    for p in frontier]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label="Pareto frontier")
    ax.scatter([elbow.strong_rate], [elbow.accuracy], color="red", zorder=5, label="elbow")
    ax.annotate(
        f"  threshold={elbow.threshold:.4f}",
        (elbow.strong_rate, elbow.accuracy),
        fontsize=8,
    )
    ax.set_xlabel("Strong model call rate")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Pareto Frontier — split {split_id}\n{weak_model_name} vs {strong_model_name}")
    ax.legend()

    filename = f"pareto_frontier_split{split_id}.png"
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
    frontier = _build_frontier(split_id, weak_model_name, strong_model_name, is_test)
    elbow = _find_elbow(frontier)
    _save_plot(frontier, elbow, split_id, weak_model_name, strong_model_name)

    a, b = frontier[0], frontier[-1]
    print(
        f"Frontier: {len(frontier)} points  "
        f"[strong_rate {a.strong_rate:.1%}→{b.strong_rate:.1%}, "
        f"accuracy {a.accuracy:.1%}→{b.accuracy:.1%}]"
    )
    print(
        f"Elbow:    threshold={elbow.threshold:.5f}  "
        f"strong_rate={elbow.strong_rate:.1%}  "
        f"accuracy={elbow.accuracy:.1%}"
    )
    return elbow.threshold