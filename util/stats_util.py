"""
stats_util.py - Router comparison statistics: score distributions, McNemar test, Δ accuracy.
"""

from __future__ import annotations
from pathlib import Path


def calculate(
    sae_router:        str  = "routing_decisions.jsonl",
    route_llm:         str  = "route_llm_decisions.jsonl",
    output:            str  = "visuals/routing_stats.png",
    weak_model_name:   str  = "",
    strong_model_name: str  = "",
    split_id:          int  = 1,
    is_test:           bool = True,
) -> None:
    """Compare SAE+MLP and RouteLLM routers: score distributions, McNemar test, Δ accuracy."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from scipy.stats import chi2
    from util.smart_file_util import load_jsonl

    sae_path = Path(sae_router)
    llm_path = Path(route_llm)
    for p in (sae_path, llm_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    sae_records = load_jsonl(sae_path)
    llm_records = load_jsonl(llm_path)

    # ── DB pass/fail + millis ─────────────────────────────────────────────────
    if not weak_model_name or not strong_model_name:
        raise ValueError("--weak-model and --strong-model are required.")

    import daos.model_task_result_dao as model_task_result_dao
    from route_llm.calculate_threshold import build_frontier

    weak_rows   = model_task_result_dao.get_all_for_model_split(weak_model_name,   split_id, is_test)
    strong_rows = model_task_result_dao.get_all_for_model_split(strong_model_name, split_id, is_test)

    weak_passed   = {r.task_id: r.passed     for r in weak_rows   if r.passed     is not None}
    strong_passed = {r.task_id: r.passed     for r in strong_rows if r.passed     is not None}
    weak_millis   = {r.task_id: r.run_millis for r in weak_rows   if r.run_millis is not None}
    strong_millis = {r.task_id: r.run_millis for r in strong_rows if r.run_millis is not None}

    def _correct(record: dict):
        """Actual pass/fail of the model that handled this task. None if not in DB."""
        return (strong_passed if record["route"] == "strong" else weak_passed).get(record["task_id"])

    def _cost_fraction(records: list) -> float:
        total_strong_millis = sum(strong_millis.get(r["task_id"], 0) for r in records)
        if total_strong_millis == 0:
            return sum(1 for r in records if r["route"] == "strong") / len(records)
        spent = sum(
            (strong_millis if r["route"] == "strong" else weak_millis).get(r["task_id"], 0)
            for r in records
        )
        return spent / total_strong_millis

    # ── McNemar + Δ accuracy (oracle routing correctness from JSONL) ─────────
    # "correct" here means the router sent easy tasks to weak (which passed) and
    # hard tasks to strong (which weak would have failed) — not actual task solve rate.
    sae_map = {r["task_id"]: r["correct"] for r in sae_records}
    llm_map = {r["task_id"]: r["correct"] for r in llm_records}
    shared  = sorted(set(sae_map) & set(llm_map))

    sae_correct_shared = np.array([sae_map[t] for t in shared], dtype=float)
    llm_correct_shared = np.array([llm_map[t] for t in shared], dtype=float)

    A = int(sum((sae_correct_shared == 1) & (llm_correct_shared == 1)))
    B = int(sum((sae_correct_shared == 1) & (llm_correct_shared == 0)))
    C = int(sum((sae_correct_shared == 0) & (llm_correct_shared == 1)))
    D = int(sum((sae_correct_shared == 0) & (llm_correct_shared == 0)))

    if B + C > 0:
        mcnemar_stat = (abs(B - C) - 1) ** 2 / (B + C)
        mcnemar_p    = chi2.sf(mcnemar_stat, df=1)
    else:
        mcnemar_stat, mcnemar_p = 0.0, 1.0

    p_str = "< 0.0001" if mcnemar_p < 0.0001 else f"= {mcnemar_p:.4f}"

    n       = len(shared)
    acc_sae = sae_correct_shared.mean()
    acc_llm = llm_correct_shared.mean()
    delta   = acc_sae - acc_llm

    # ── Terminal output ────────────────────────────────────────────────────────
    print("\n── Router comparison ────────────────────────────────")
    print(f"{'':20s} {'SAE+MLP':>10} {'RouteLLM':>10}")
    print(f"{'Oracle routing acc':20s} {acc_sae:>9.1%} {acc_llm:>9.1%}")
    print(f"{'Strong %':20s} "
          f"{sum(1 for r in sae_records if r['route']=='strong')/len(sae_records):>9.1%} "
          f"{sum(1 for r in llm_records if r['route']=='strong')/len(llm_records):>9.1%}")
    print(f"{'N shared tasks':20s} {n:>10}")
    print(f"\nMcNemar (N={n}): A={A}  B={B}  C={C}  D={D}")
    print(f"  χ²={mcnemar_stat:.2f}  p {p_str}")
    sign = "+" if delta >= 0 else ""
    print(f"Δ oracle routing acc (SAE−LLM):  {sign}{delta:.1%}")

    # ── Figure ────────────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.55)

    # ── Panel 1: Operating point scatter (cost vs accuracy) ───────────────────
    ax1 = fig.add_subplot(gs[0])

    try:
        frontier = build_frontier(split_id, weak_model_name, strong_model_name, is_test=True)
        ax1.plot(
            [p.cost_fraction for p in frontier],
            [p.accuracy      for p in frontier],
            color="grey", linewidth=1.2, alpha=0.6, zorder=1, label="Pareto frontier",
        )
    except ValueError:
        pass

    llm_cost = _cost_fraction(llm_records)
    sae_cost = _cost_fraction(sae_records)
    llm_acc  = np.mean([v for r in llm_records if (v := _correct(r)) is not None])
    sae_acc  = np.mean([v for r in sae_records if (v := _correct(r)) is not None])

    ax1.scatter([llm_cost], [llm_acc], color="tab:blue",   s=120, zorder=3, label="RouteLLM")
    ax1.scatter([sae_cost], [sae_acc], color="tab:orange",  s=120, zorder=3, label="SAE+MLP")

    for cost, acc, name, color in [
        (llm_cost, llm_acc, "RouteLLM", "tab:blue"),
        (sae_cost, sae_acc, "SAE+MLP",  "tab:orange"),
    ]:
        ax1.annotate(
            f"{name}\n({cost:.0%} cost, {acc:.1%} acc)",
            xy=(cost, acc), xytext=(8, -18), textcoords="offset points",
            fontsize=8, color=color,
        )

    ax1.set_xlabel("Cost fraction (spent millis / all-strong millis)", fontsize=10)
    ax1.set_ylabel("Task accuracy", fontsize=10)
    ax1.set_title("Accuracy vs Cost\n(operating points)", fontsize=11)
    ax1.autoscale(enable=True, tight=False)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    xpad = (x1 - x0) * 0.15
    ypad = (y1 - y0) * 0.15
    ax1.set_xlim(max(0, x0 - xpad), min(1, x1 + xpad))
    ax1.set_ylim(max(0, y0 - ypad), min(1, y1 + ypad))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: McNemar 2×2 heatmap ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    table = np.array([[A, B], [C, D]], dtype=float)
    im = ax2.imshow(table, cmap="Blues", vmin=0)

    cell_labels = [
        [f"A = {A}\n(both ✓)",        f"B = {B}\n(SAE ✓, LLM ✗)"],
        [f"C = {C}\n(SAE ✗, LLM ✓)", f"D = {D}\n(both ✗)"],
    ]
    for r in range(2):
        for c in range(2):
            bright = table[r, c] < table.max() * 0.55
            ax2.text(c, r, cell_labels[r][c],
                     ha="center", va="center", fontsize=9,
                     color="black" if bright else "white",
                     fontweight="bold")

    for (row, col) in [(0, 1), (1, 0)]:
        ax2.add_patch(mpatches.FancyBboxPatch(
            (col - 0.5, row - 0.5), 1, 1,
            boxstyle="square,pad=0",
            linewidth=2.5, edgecolor="crimson", facecolor="none", zorder=5,
        ))

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["LLM correct", "LLM wrong"], fontsize=9)
    ax2.set_yticklabels(["SAE correct", "SAE wrong"], fontsize=9)
    ax2.set_title(
        f"McNemar contingency table (oracle routing)\nχ² = {mcnemar_stat:.2f},  p {p_str}", fontsize=10,
    )
    cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("Count", fontsize=8)

    fig.suptitle("Router Comparison: SAE+MLP vs RouteLLM", fontsize=13, y=1.01)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_path}")
