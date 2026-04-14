"""
cli.py - CS5966 Interpretable Routing CLI.

Commands:
  python cli.py db init                     Initialize the SQLite database
  python cli.py server up/down/status       Manage vLLM + litellm servers
  python cli.py inference run               Baseline inference (no routing)
  python cli.py inference route             RouteLLM BERT-routed inference
  python cli.py inference toughness         Score problems with BERT router
  python cli.py sae train                   Train a Sparse Autoencoder
  python cli.py sae extract                 Extract sparse feature vectors
  python cli.py mlp train                   Train the MLP router
  python cli.py mlp eval                    Evaluate the MLP router
  python cli.py route-llm calculate-threshold  Compute routing threshold
  python cli.py route-llm batch             Batch RouteLLM routing decisions
  python cli.py router batch                Batch SAE+MLP routing decisions
  python cli.py test run                    Run Docker-based code tests
  python cli.py stats calculate             Pareto, McNemar, and Δ accuracy comparison
  python cli.py file to-csv                 Convert JSONL to CSV
  python cli.py file filter                 Filter JSONL by split task IDs
  python cli.py login                       HuggingFace login
"""

from dotenv import load_dotenv
load_dotenv()  # must run before any module reads os.environ

from pathlib import Path
from typing import Annotated, List, Optional
import typer

app = typer.Typer(help="CS5966 Interpretable Routing CLI.", no_args_is_help=True)

# Sub-apps
db_app        = typer.Typer(help="Database commands.", no_args_is_help=True)
server_app    = typer.Typer(help="Server lifecycle commands.", no_args_is_help=True)
inference_app = typer.Typer(help="Inference commands.", no_args_is_help=True)
sae_app       = typer.Typer(help="SAE training and extraction commands.", no_args_is_help=True)
mlp_app       = typer.Typer(help="MLP router commands.", no_args_is_help=True)
route_llm_app = typer.Typer(help="RouteLLM commands.", no_args_is_help=True)
router_app    = typer.Typer(help="SAE+MLP router commands.", no_args_is_help=True)
test_app      = typer.Typer(help="Code testing commands.", no_args_is_help=True)
stats_app     = typer.Typer(help="Result statistics commands.", no_args_is_help=True)
file_app      = typer.Typer(help="File utility commands.", no_args_is_help=True)

app.add_typer(db_app,        name="db")
app.add_typer(server_app,    name="server")
app.add_typer(inference_app, name="inference")
app.add_typer(sae_app,       name="sae")
app.add_typer(mlp_app,       name="mlp")
app.add_typer(route_llm_app, name="route-llm")
app.add_typer(router_app,    name="router")
app.add_typer(test_app,      name="test")
app.add_typer(stats_app,     name="stats")
app.add_typer(file_app,      name="file")


# =============================================================================
# db
# =============================================================================

@db_app.command("init")
def db_init() -> None:
    """Initialize the SQLite database, seed tasks, and create the default split."""
    from util.database_util import init_db
    init_db()


# =============================================================================
# server
# =============================================================================

@server_app.command("up")
def server_up(
    models: Annotated[List[str], typer.Option("--model", help="model_id:gpu_id (repeatable)")] = [],
    detach: Annotated[bool,      typer.Option("--detach", help="Run servers in background")]    = False,
) -> None:
    """Start vLLM inference servers and litellm proxy."""
    from util.model_util import up
    pairs = []
    for entry in models:
        model_id, _, gpu_id = entry.partition(":")
        pairs.append((model_id.strip(), gpu_id.strip() or "0"))
    up(models=pairs, detach=detach)


@server_app.command("down")
def server_down() -> None:
    """Stop all inference servers."""
    from util.model_util import down
    down()


@server_app.command("status")
def server_status() -> None:
    """Show server health and PIDs."""
    from util.model_util import status
    status()


# =============================================================================
# inference
# =============================================================================

@inference_app.command("run")
def inference_run(
    model_name: Annotated[str,  typer.Option("--model-name", help="HuggingFace model ID")] = "",
    split_id:   Annotated[int,  typer.Option("--split-id",   help="DB split id")]           = 1,
    workers:    Annotated[int,  typer.Option("--workers",    help="Concurrent inference requests")] = 1,
    is_test:    Annotated[bool, typer.Option("--test",       help="Run on test partition instead of train")] = False,
) -> None:
    """Run baseline inference (no routing) for a single model."""
    from daos import split_dao, task_split_dao

    if split_dao.get_by_id(split_id) is None:
        typer.echo(f"Error: split_id={split_id} does not exist in the database.", err=True)
        raise typer.Exit(code=1)
    if task_split_dao.count_for_split(split_id, is_test=is_test) == 0:
        partition = "test" if is_test else "train"
        typer.echo(f"Error: no tasks found for split_id={split_id} ({partition} partition).", err=True)
        raise typer.Exit(code=1)

    from util.model_util import require_up
    from util.inference_util import run_inference, get_openai_client
    from daos import tasks_dao

    require_up()
    client = get_openai_client()
    tasks  = tasks_dao.get_all_for_split(split_id, is_test=is_test)
    run_inference(
        problems=tasks,
        create_fn=client.completions.create,
        model_str=f"openai/{model_name}",
        model_name=model_name,
        max_workers=workers,
    )


# =============================================================================
# sae
# =============================================================================

@sae_app.command("train")
def sae_train(
    model_name: Annotated[str, typer.Option("--model-name", help="HuggingFace model ID")]      = "",
    hook_name:  Annotated[str, typer.Option("--hook-name",  help="TransformerLens hook point")] = "",
    d_model:    Annotated[int, typer.Option("--d-model",    help="Model hidden dimension")]     = 0,
    split_id:   Annotated[int, typer.Option("--split-id",   help="DB split id")]                = 1,
) -> None:
    """Train a Sparse Autoencoder on the given model's residual stream."""
    from util.smart_file_util import activations_path

    p = activations_path(split_id, model_name)
    if not p.exists():
        typer.echo(
            f"Error: activations not found at '{p}'. "
            f"Run activation extraction for model='{model_name}', split_id={split_id} first.",
            err=True,
        )
        raise typer.Exit(code=1)

    from sae.train_sae import train_sae
    train_sae(model_name=model_name, hook_name=hook_name, d_model=d_model, split_id=split_id)


@sae_app.command("extract")
def sae_extract(
    model_name: Annotated[str,  typer.Option("--model-name", help="HuggingFace model ID")]      = "",
    split_id:   Annotated[int,  typer.Option("--split-id",   help="DB split id")]                = 1,
    is_test:    Annotated[bool, typer.Option("--test",        help="Extract test partition")]    = False,
    sae_path:   Annotated[Optional[str], typer.Option("--sae-path", help="SAE checkpoint dir")] = None,
) -> None:
    """Extract dense activations and encode through SAE to produce sparse feature vectors."""
    if sae_path is None:
        from util.smart_file_util import sae_weights_path, sae_cfg_path

        for p, label in [
            (sae_weights_path(split_id, model_name), "SAE weights"),
            (sae_cfg_path(split_id, model_name),     "SAE config"),
        ]:
            if not p.exists():
                typer.echo(
                    f"Error: {label} not found at '{p}'. "
                    f"Run 'sae train' for model='{model_name}', split_id={split_id} first.",
                    err=True,
                )
                raise typer.Exit(code=1)

    from sae.extract_spv import run

    run(model_name=model_name, split_id=split_id, is_test=is_test, sae_path=sae_path)


# =============================================================================
# mlp
# =============================================================================

@mlp_app.command("train")
def mlp_train(
    model_name:  Annotated[str,  typer.Option("--model-name",  help="HuggingFace model ID")]                     = "",
    split_id:    Annotated[int,  typer.Option("--split-id",    help="DB split id")]                              = 1,
    grid_search: Annotated[bool, typer.Option("--grid-search", help="Run k-fold CV to find best hyperparams")] = False,
) -> None:
    """Train the MLP router on SAE sparse features and test results from the DB."""
    from util.smart_file_util import sparse_features_path
    from daos import model_task_result_dao

    if not sparse_features_path(split_id, model_name).exists():
        typer.echo(
            f"Error: sparse features not found for model='{model_name}', split_id={split_id}. "
            f"Run 'sae extract' first.",
            err=True,
        )
        raise typer.Exit(code=1)
    if model_task_result_dao.count_with_pass_labels_for_model_split(model_name, split_id, is_test=False) == 0:
        typer.echo(
            f"Error: no pass/fail labels for model='{model_name}', split_id={split_id} (train partition). "
            f"Run 'test run' first.",
            err=True,
        )
        raise typer.Exit(code=1)

    from mlp.mlp_train import train_mlp

    train_mlp(split_id=split_id, model_name=model_name, grid_search=grid_search)


@mlp_app.command("eval")
def mlp_eval(
    model_name: Annotated[str, typer.Option("--model-name", help="HuggingFace model ID")] = "",
    split_id:   Annotated[int, typer.Option("--split-id",   help="DB split id")]           = 1,
) -> None:
    """Evaluate the trained MLP router on the test split."""
    from util.smart_file_util import mlp_path
    from daos import model_task_result_dao

    if not mlp_path(split_id, model_name).exists():
        typer.echo(
            f"Error: MLP weights not found for model='{model_name}', split_id={split_id}. "
            f"Run 'mlp train' first.",
            err=True,
        )
        raise typer.Exit(code=1)
    if model_task_result_dao.count_with_pass_labels_for_model_split(model_name, split_id, is_test=True) == 0:
        typer.echo(
            f"Error: no pass/fail labels for model='{model_name}', split_id={split_id} (test partition). "
            f"Run 'test run --test' first.",
            err=True,
        )
        raise typer.Exit(code=1)

    from mlp.eval_mlp import evaluate_mlp

    evaluate_mlp(split_id=split_id, model_name=model_name)


# =============================================================================
# route-llm
# =============================================================================

@route_llm_app.command("calculate-threshold")
def route_llm_threshold(
    split_id:          Annotated[int,  typer.Option("--split-id",     help="DB split id")] = 1,
    weak_model_name:   Annotated[str,  typer.Option("--weak-model",   help="Weak model name")] = "",
    strong_model_name: Annotated[str,  typer.Option("--strong-model", help="Strong model name")] = "",
    is_test:           Annotated[bool, typer.Option("--is-test",      help="Use test partition (default: train)")] = False,
) -> None:
    """Compute the RouteLLM routing threshold via Pareto frontier + geometric elbow."""
    from route_llm.calculate_threshold import calculate_threshold
    import daos.runs_dao as runs_dao

    if not weak_model_name or not strong_model_name:
        typer.echo("Error: --weak-model and --strong-model are required.", err=True)
        raise typer.Exit(code=1)

    import daos.model_task_result_dao as model_task_result_dao

    for mname in (weak_model_name, strong_model_name):
        if model_task_result_dao.count_for_model_split(mname, split_id, is_test) == 0:
            partition = "test" if is_test else "train"
            typer.echo(
                f"Error: no inference results for model='{mname}', split_id={split_id}, "
                f"partition={partition}. Run 'inference run' for that model first.",
                err=True,
            )
            raise typer.Exit(code=1)

    run = runs_dao.get_by_models_and_split(weak_model_name, strong_model_name, split_id)

    if not (run is None or run.route_llm_threshold is None):
        typer.echo(f"Threshold has already been calcualted to be: {run.route_llm_threshold}")
        typer.echo("If you need this to be recalculated, please clear this record and/or create a new run.")
        return

    threshold = calculate_threshold(
        split_id=split_id,
        weak_model_name=weak_model_name,
        strong_model_name=strong_model_name,
        is_test=is_test,
    )

    if run is None:
        runs_dao.create(weak_model_name, strong_model_name,split_id, threshold)
    else:
        runs_dao.update_threshold(run.id, threshold)

    new_run = runs_dao.get_by_models_and_split(weak_model_name, strong_model_name, split_id)

    if new_run:
        typer.echo(f"Threshold saved to run {new_run.id}.")
    else:
        typer.echo("Warning: no matching run found in DB — threshold not persisted.", err=True)


@route_llm_app.command("batch")
def route_llm_batch(
    weak_model_name:   Annotated[str,  typer.Option("--weak-model",   help="Weak model name")]   = "",
    strong_model_name: Annotated[str,  typer.Option("--strong-model", help="Strong model name")] = "",
    split_id:          Annotated[int,  typer.Option("--split-id",     help="DB split id")]       = 1,
    is_test:           Annotated[bool, typer.Option("--test",         help="Use test partition")] = True,
    output:            Annotated[str,  typer.Option("--output",       help="Output .jsonl path")] = "route_llm_decisions.jsonl",
) -> None:
    """Generate RouteLLM routing decisions using toughness scores vs saved threshold."""
    if not weak_model_name or not strong_model_name:
        typer.echo("Error: --weak-model and --strong-model are required.", err=True)
        raise typer.Exit(code=1)

    if Path(output).exists():
        typer.echo(f"Output already exists at {output}, skipping.")
        return

    import daos.runs_dao as runs_dao
    import daos.tasks_dao as tasks_dao
    import daos.model_task_result_dao as model_task_result_dao
    from util.smart_file_util import write_jsonl

    run = runs_dao.get_by_models_and_split(weak_model_name, strong_model_name, split_id)
    if run is None or run.route_llm_threshold is None:
        typer.echo(
            "Error: no threshold found. Run 'route-llm calculate-threshold' first.",
            err=True,
        )
        raise typer.Exit(code=1)

    threshold = run.route_llm_threshold
    tasks     = tasks_dao.get_all_for_split(split_id, is_test=is_test)

    results   = model_task_result_dao.get_all_for_model_split(weak_model_name, split_id, is_test=is_test)
    label_map = {r.task_id: r.passed for r in results}

    decisions = []
    for task in tasks:
        if task.toughness_score is None:
            continue
        route    = "strong" if task.toughness_score >= threshold else "weak"
        weak_pass = label_map.get(task.id)
        correct  = (route == "weak" and weak_pass) or (route == "strong" and not weak_pass)
        decisions.append({
            "task_id":         task.id,
            "route":           route,
            "toughness_score": round(task.toughness_score, 5),
            "correct":         correct,
            "weak_pass":       bool(weak_pass) if weak_pass is not None else None,
        })

    write_jsonl(Path(output), decisions)
    typer.echo(f"RouteLLM decisions saved to {output}  ({len(decisions)} problems, threshold={threshold:.5f})")


# =============================================================================
# router (SAE+MLP batch decisions)
# =============================================================================

@router_app.command("batch")
def router_batch(
    model_name: Annotated[str,  typer.Option("--model-name", help="HuggingFace model ID")]    = "",
    split_id:   Annotated[int,  typer.Option("--split-id",   help="DB split id")]              = 1,
    is_test:    Annotated[bool, typer.Option("--test",        help="Use test partition")]      = True,
    output:     Annotated[str,  typer.Option("--output",      help="Output .jsonl path")]      = "routing_decisions.jsonl",
) -> None:
    """Generate SAE+MLP routing decisions for a split and write to JSONL."""
    if Path(output).exists():
        typer.echo(f"Output already exists at {output}, skipping.")
        return

    from util.smart_file_util import mlp_path, sparse_features_path

    if not mlp_path(split_id, model_name).exists():
        typer.echo(
            f"Error: MLP weights not found for model='{model_name}', split_id={split_id}. "
            f"Run 'mlp train' first.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not sparse_features_path(split_id, model_name).exists():
        typer.echo(
            f"Error: sparse features not found for model='{model_name}', split_id={split_id}. "
            f"Run 'sae extract' first.",
            err=True,
        )
        raise typer.Exit(code=1)

    import torch
    from mlp.model import MLP, HIDDEN_DIM
    from util import tensor_util
    from util.smart_file_util import mlp_path, write_jsonl
    from daos import model_task_result_dao

    device = "cuda" if torch.cuda.is_available() else "cpu"

    features_dict = tensor_util.load_features(split_id, model_name)
    features      = features_dict["features"].to(device)
    task_ids      = features_dict["task_ids"]

    weights = mlp_path(split_id, model_name)
    model   = MLP(d_in=features.shape[1], hidden=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(features).cpu()

    # Load existing pass/fail labels for correctness annotation
    results   = model_task_result_dao.get_all_for_model_split(model_name, split_id, is_test=is_test)
    label_map = {r.task_id: r.passed for r in results}

    decisions = []
    for task_id, logit in zip(task_ids, logits.tolist()):
        route     = "weak" if logit > 0 else "strong"
        weak_pass = label_map.get(task_id)
        correct   = (route == "weak" and weak_pass) or (route == "strong" and not weak_pass)
        decisions.append({
            "task_id":   task_id,
            "route":     route,
            "logit":     round(logit, 5),
            "correct":   correct,
            "weak_pass": bool(weak_pass) if weak_pass is not None else None,
        })

    write_jsonl(Path(output), decisions)
    typer.echo(f"Routing decisions saved to {output}  ({len(decisions)} problems)")


# =============================================================================
# test
# =============================================================================

@test_app.command("run")
def test_run(
    model_name: Annotated[str,  typer.Option("--model-name", help="HuggingFace model ID")],
    split_id:   Annotated[int,  typer.Option("--split-id",   help="DB split id")]           = 1,
    is_test:    Annotated[bool, typer.Option("--test",       help="Use test partition")]    = False,
) -> None:
    """Evaluate inference results against Docker test cases for all languages."""
    from daos import model_task_result_dao

    if model_task_result_dao.count_for_model_split(model_name, split_id, is_test) == 0:
        partition = "test" if is_test else "train"
        typer.echo(
            f"Error: no inference results for model='{model_name}', split_id={split_id}, "
            f"partition={partition}. Run 'inference run' first.",
            err=True,
        )
        raise typer.Exit(code=1)

    from util.unit_test_util import run_tests
    run_tests(model_name=model_name, split_id=split_id, is_test=is_test)


# =============================================================================
# stats
# =============================================================================

@stats_app.command("calculate")
def stats_calculate(
    sae_router: Annotated[str, typer.Option("--sae-router", help="SAE+MLP decisions JSONL")]  = "routing_decisions.jsonl",
    route_llm:  Annotated[str, typer.Option("--route-llm",  help="RouteLLM decisions JSONL")] = "route_llm_decisions.jsonl",
    output:     Annotated[str, typer.Option("--output",     help="Output plot path")]          = "visuals/routing_stats.png",
    n_boot:     Annotated[int, typer.Option("--n-boot",     help="Bootstrap iterations")]      = 10_000,
) -> None:
    """Compare SAE+MLP and RouteLLM routers: Pareto curves, McNemar test, and Δ accuracy with CI."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import chi2
    from util.smart_file_util import load_jsonl

    sae_path = Path(sae_router)
    llm_path = Path(route_llm)
    for p in (sae_path, llm_path):
        if not p.exists():
            typer.echo(f"Error: missing file {p}. Run the corresponding batch command first.", err=True)
            raise typer.Exit(code=1)

    sae_records = load_jsonl(sae_path)
    llm_records = load_jsonl(llm_path)

    # ── Pareto sweep ──────────────────────────────────────────────────────────
    def _pareto_curve(records: list[dict], score_key: str, strong_if_gte: bool):
        """Sweep score thresholds and return (costs, accuracies, chosen_cost) lists."""
        valid = [r for r in records if score_key in r]
        if not valid:
            return [], [], None
        scores     = [r[score_key] for r in valid]
        thresholds = sorted(set(scores))
        costs, accs = [], []
        for t in thresholds:
            if strong_if_gte:
                strong = [r for r in valid if r[score_key] >= t]
                weak   = [r for r in valid if r[score_key] <  t]
            else:  # strong if logit <= t (lower logit → more confident it's hard)
                strong = [r for r in valid if r[score_key] <= t]
                weak   = [r for r in valid if r[score_key] >  t]
            cost    = len(strong) / len(valid)
            correct = sum(r["correct"] for r in strong) + sum(r["correct"] for r in weak)
            costs.append(cost)
            accs.append(correct / len(valid))
        chosen_cost = sum(1 for r in valid if r["route"] == "strong") / len(valid)
        return costs, accs, chosen_cost

    llm_costs, llm_accs, llm_op = _pareto_curve(llm_records, "toughness_score", strong_if_gte=True)
    sae_costs, sae_accs, sae_op = _pareto_curve(sae_records, "logit",           strong_if_gte=False)

    # ── McNemar ───────────────────────────────────────────────────────────────
    sae_map = {r["task_id"]: r["correct"] for r in sae_records}
    llm_map = {r["task_id"]: r["correct"] for r in llm_records}
    shared  = sorted(set(sae_map) & set(llm_map))

    sae_correct_shared = np.array([sae_map[t] for t in shared], dtype=float)
    llm_correct_shared = np.array([llm_map[t] for t in shared], dtype=float)

    B = int(sum((sae_correct_shared == 1) & (llm_correct_shared == 0)))  # SAE right, LLM wrong
    C = int(sum((sae_correct_shared == 0) & (llm_correct_shared == 1)))  # SAE wrong, LLM right
    if B + C > 0:
        mcnemar_stat = (abs(B - C) - 1) ** 2 / (B + C)
        mcnemar_p    = chi2.sf(mcnemar_stat, df=1)
    else:
        mcnemar_stat, mcnemar_p = 0.0, 1.0

    # ── Δ accuracy with bootstrap CI ──────────────────────────────────────────
    sae_all = np.array([r["correct"] for r in sae_records], dtype=float)
    llm_all = np.array([r["correct"] for r in llm_records], dtype=float)

    acc_sae = sae_all.mean()
    acc_llm = llm_all.mean()

    # Pair by shared task_ids for bootstrap
    n = len(shared)
    sae_s = sae_correct_shared
    llm_s = llm_correct_shared
    rng = np.random.default_rng(42)
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = sae_s[idx].mean() - llm_s[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    delta = acc_sae - acc_llm

    # ── Terminal output ────────────────────────────────────────────────────────
    typer.echo("\n── Router comparison ────────────────────────────────")
    typer.echo(f"{'':20s} {'SAE+MLP':>10} {'RouteLLM':>10}")
    typer.echo(f"{'Accuracy':20s} {acc_sae:>9.1%} {acc_llm:>9.1%}")
    typer.echo(f"{'Strong %':20s} {sum(1 for r in sae_records if r['route']=='strong')/len(sae_records):>9.1%} "
               f"{sum(1 for r in llm_records if r['route']=='strong')/len(llm_records):>9.1%}")
    typer.echo(f"{'N tasks':20s} {len(sae_records):>10} {len(llm_records):>10}")
    typer.echo(f"\nMcNemar (shared N={n}): B={B}  C={C}  χ²={mcnemar_stat:.2f}  p={mcnemar_p:.4f}")
    sign = "+" if delta >= 0 else ""
    typer.echo(f"Δ accuracy (SAE−LLM):  {sign}{delta:.1%}  95% CI [{ci_lo:.1%}, {ci_hi:.1%}]")

    # ── Figure ────────────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: Pareto curves
    ax1 = fig.add_subplot(gs[0])
    if llm_costs:
        ax1.plot(llm_costs, llm_accs, color="tab:blue",  linewidth=1.5, label="RouteLLM")
        ax1.axvline(llm_op, color="tab:blue",  linestyle="--", alpha=0.6, linewidth=1)
    if sae_costs:
        ax1.plot(sae_costs, sae_accs, color="tab:orange", linewidth=1.5, label="SAE+MLP")
        ax1.axvline(sae_op, color="tab:orange", linestyle="--", alpha=0.6, linewidth=1)
    ax1.set_xlabel("Cost (fraction routed to strong)")
    ax1.set_ylabel("Routing accuracy")
    ax1.set_title("Pareto Curves")
    if llm_costs or sae_costs:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: McNemar 2×2 heatmap
    ax2 = fig.add_subplot(gs[1])
    A = int(sum((sae_correct_shared == 1) & (llm_correct_shared == 1)))
    D = int(sum((sae_correct_shared == 0) & (llm_correct_shared == 0)))
    table = np.array([[A, B], [C, D]])
    im = ax2.imshow(table, cmap="Blues")
    for (r, c_), val in np.ndenumerate(table):
        ax2.text(c_, r, str(val), ha="center", va="center", fontsize=14)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["LLM correct", "LLM wrong"])
    ax2.set_yticklabels(["SAE correct", "SAE wrong"])
    ax2.set_title(f"McNemar  χ²={mcnemar_stat:.2f}  p={mcnemar_p:.4f}")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: Δ bar with CI
    ax3 = fig.add_subplot(gs[2])
    color = "tab:green" if delta >= 0 else "tab:red"
    ax3.bar([0], [delta], color=color, alpha=0.7, width=0.4)
    ax3.errorbar([0], [delta], yerr=[[delta - ci_lo], [ci_hi - delta]],
                 fmt="none", color="black", capsize=8, linewidth=2)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks([0])
    ax3.set_xticklabels(["SAE+MLP − RouteLLM"])
    ax3.set_ylabel("Δ Accuracy")
    ax3.set_title("Accuracy Difference\n(95% bootstrap CI)")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

    fig.suptitle("Router Comparison", fontsize=14)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    typer.echo(f"\nPlot saved to {out_path}")


# =============================================================================
# file
# =============================================================================

@file_app.command("to-csv")
def file_to_csv(
    input_file:  Annotated[Path, typer.Argument(help="Path to the .jsonl file")],
    output_file: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output .csv path")] = None,
    columns:     Annotated[Optional[str],  typer.Option("--columns", "-c", help="Comma-separated columns")] = None,
) -> None:
    """Convert a JSONL file to CSV."""
    from util.smart_file_util import load_jsonl, export_csv

    records = load_jsonl(input_file)
    cols    = [c.strip() for c in columns.split(",")] if columns else None
    out     = export_csv(input_file, records, columns=cols, output_path=output_file)
    typer.echo(f"Wrote {out}")


@file_app.command("filter")
def file_filter(
    source_file: Annotated[Path, typer.Argument(help="Source .jsonl file")],
    split_id:    Annotated[int,  typer.Option("--split-id", help="DB split id")]                         = 1,
    is_test:     Annotated[bool, typer.Option("--test",     help="Filter to test partition (default: train)")] = False,
    output_dir:  Annotated[Path, typer.Option("--output-dir", "-o", help="Directory for output file")]   = Path("route_llm_results"),
) -> None:
    """Filter a JSONL file to records whose task_id belongs to the specified split partition."""
    from util.smart_file_util import load_jsonl, write_jsonl, filter_by_task_ids
    from daos import task_split_dao

    task_ids = set(task_split_dao.get_task_ids_for_split(split_id, is_test=is_test))

    records  = load_jsonl(source_file)
    filtered = filter_by_task_ids(records, task_ids)

    partition   = "test" if is_test else "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_file.stem}_{partition}.jsonl"
    write_jsonl(output_path, filtered)
    typer.echo(f"Kept {len(filtered)}/{len(records)} records → {output_path}")


# =============================================================================
# login
# =============================================================================

@app.command()
def login() -> None:
    """Authenticate with HuggingFace to access gated models (Llama)."""
    from huggingface_hub import login as hf_login
    hf_login()


if __name__ == "__main__":
    app()
