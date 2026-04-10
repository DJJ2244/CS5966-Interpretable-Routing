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
  python cli.py router batch                Batch SAE+MLP routing decisions
  python cli.py test run                    Run Docker-based code tests
  python cli.py stats calculate             Compute result statistics
  python cli.py file to-csv                 Convert JSONL to CSV
  python cli.py file filter                 Filter JSONL by split task IDs
  python cli.py login                       HuggingFace login
"""

from dotenv import load_dotenv
load_dotenv()  # must run before any module reads os.environ

from pathlib import Path
from typing import Annotated, Optional
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
    weak_gpu:   Annotated[str,  typer.Option("--weak-gpu",   help="CUDA device for weak model")]  = "0",
    strong_gpu: Annotated[str,  typer.Option("--strong-gpu", help="CUDA device for strong model")] = "1",
    single_gpu: Annotated[bool, typer.Option("--single-gpu", help="Run both models on GPU 0")]      = False,
    detach:     Annotated[bool, typer.Option("--detach",     help="Run servers in background")]      = False,
) -> None:
    """Start vLLM inference servers and litellm proxy."""
    from util.model_util import up
    if single_gpu:
        strong_gpu = weak_gpu
    up(weak_gpu=weak_gpu, strong_gpu=strong_gpu, detach=detach)


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
    model:      Annotated[str, typer.Option("--model",      help="weak | strong | all")]           = "all",
    split_id:   Annotated[int, typer.Option("--split-id",   help="DB split id")]                    = 1,
    output_dir: Annotated[str, typer.Option("--output-dir", help="Directory for result .jsonl")]    = "route_llm_results",
    workers:    Annotated[int, typer.Option("--workers",    help="Concurrent inference requests")]  = 8,
) -> None:
    """Run baseline inference (no routing) for weak, strong, or both models."""
    import os
    from openai import OpenAI
    from util.model_util import require_up
    from util.inference_util import run_inference, get_openai_client
    from daos import tasks_dao, model_dao
    from util.database_connection_util import get_connection

    require_up()
    conn = get_connection()

    weak_name   = os.environ["WEAK_MODEL"].removeprefix("openai/")
    strong_name = os.environ["STRONG_MODEL"].removeprefix("openai/")
    client      = get_openai_client()

    tasks = tasks_dao.get_all_for_split(conn, split_id, is_test=False)

    targets = []
    if model in ("weak", "all"):
        targets.append((weak_name,   Path(output_dir) / "results_weak.jsonl"))
    if model in ("strong", "all"):
        targets.append((strong_name, Path(output_dir) / "results_strong.jsonl"))
    if not targets:
        raise typer.BadParameter(f"Unknown model '{model}'. Choose weak, strong, or all.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for model_str, output_path in targets:
        db_model = model_dao.get_or_create(conn, model_str)
        typer.echo(f"\n=== Running {model_str} ===")
        run_inference(
            problems=tasks,
            create_fn=client.completions.create,
            model_str=model_str,
            output_path=str(output_path),
            conn=conn,
            model_id=db_model.id,
            total=len(tasks),
            max_workers=workers,
        )
    conn.close()


@inference_app.command("route")
def inference_route(
    split_id:   Annotated[int, typer.Option("--split-id",   help="DB split id")]                   = 1,
    output_dir: Annotated[str, typer.Option("--output-dir", help="Directory for result .jsonl")]   = "route_llm_results",
    workers:    Annotated[int, typer.Option("--workers",    help="Concurrent inference requests")] = 8,
) -> None:
    """Route each problem to weak or strong model via the BERT router."""
    from util.model_util import require_up
    from util.inference_util import run_inference, get_router_client, ROUTER, THRESHOLD
    from daos import tasks_dao
    from util.database_connection_util import get_connection

    require_up()
    conn  = get_connection()
    tasks = tasks_dao.get_all_for_split(conn, split_id, is_test=False)

    client     = get_router_client()
    model_str  = f"router-{ROUTER}-{THRESHOLD}"
    output_path = Path(output_dir) / "router_results.jsonl"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_inference(
        problems=tasks,
        create_fn=client.completions.create,
        model_str=model_str,
        output_path=str(output_path),
        total=len(tasks),
        max_workers=workers,
    )
    conn.close()


@inference_app.command("toughness")
def inference_toughness(
    split_id:   Annotated[int, typer.Option("--split-id",   help="DB split id")]                  = 1,
    is_test:    Annotated[bool, typer.Option("--test",      help="Score test partition")]          = False,
    output_dir: Annotated[str, typer.Option("--output-dir", help="Directory for toughness.jsonl")] = "route_llm_results",
) -> None:
    """Score all problems with the BERT router. No model inference required."""
    from route_llm.toughness import record_toughness
    from util.database_connection_util import get_connection

    conn = get_connection()
    record_toughness(split_id=split_id, is_test=is_test, output_dir=output_dir, conn=conn)
    conn.close()


# =============================================================================
# sae
# =============================================================================

@sae_app.command("train")
def sae_train(
    model:    Annotated[str,          typer.Option("--model",    help="weak | strong")]          = "weak",
    split_id: Annotated[Optional[int], typer.Option("--split-id", help="DB split id for naming")] = None,
    model_id: Annotated[Optional[int], typer.Option("--model-id", help="DB model id for naming")] = None,
) -> None:
    """Train a Sparse Autoencoder on the given model's residual stream."""
    from sae.train_sae import train_sae
    train_sae(model_key=model, split_id=split_id, model_id=model_id)


@sae_app.command("extract")
def sae_extract(
    model_key: Annotated[str, typer.Option("--model-key", help="weak | strong")]            = "weak",
    split_id:  Annotated[int, typer.Option("--split-id",  help="DB split id")]               = 1,
    model_id:  Annotated[int, typer.Option("--model-id",  help="DB model id")]               = 1,
    is_test:   Annotated[bool, typer.Option("--test",     help="Extract test partition")]    = False,
    sae_path:  Annotated[Optional[str], typer.Option("--sae-path", help="SAE checkpoint dir")] = None,
) -> None:
    """Extract dense activations and encode through SAE to produce sparse feature vectors."""
    from sae.extract_spv import run
    from util.database_connection_util import get_connection

    conn = get_connection()
    run(model_key=model_key, split_id=split_id, model_id=model_id,
        is_test=is_test, conn=conn, sae_path=sae_path)
    conn.close()


# =============================================================================
# mlp
# =============================================================================

@mlp_app.command("train")
def mlp_train(
    split_id: Annotated[int, typer.Option("--split-id", help="DB split id")] = 1,
    model_id: Annotated[int, typer.Option("--model-id", help="DB model id")] = 1,
) -> None:
    """Train the MLP router on SAE sparse features and test results from the DB."""
    from mlp.mlp_train import train_mlp
    from util.database_connection_util import get_connection

    conn = get_connection()
    train_mlp(split_id=split_id, model_id=model_id, conn=conn)
    conn.close()


@mlp_app.command("eval")
def mlp_eval(
    split_id: Annotated[int, typer.Option("--split-id", help="DB split id")] = 1,
    model_id: Annotated[int, typer.Option("--model-id", help="DB model id")] = 1,
) -> None:
    """Evaluate the trained MLP router on the test split."""
    from mlp.eval_mlp import evaluate_mlp
    from util.database_connection_util import get_connection

    conn = get_connection()
    evaluate_mlp(split_id=split_id, model_id=model_id, conn=conn)
    conn.close()


# =============================================================================
# route-llm
# =============================================================================

@route_llm_app.command("calculate-threshold")
def route_llm_threshold(
    toughness_path:    Annotated[str,   typer.Option("--toughness-path",   help="Path to toughness.jsonl")] = "route_llm_results/toughness.jsonl",
    target_strong_rate: Annotated[float, typer.Option("--target-strong-rate", help="Fraction to route to strong")] = 0.5,
    save_path:         Annotated[Optional[str], typer.Option("--save",     help="Save threshold to this path")] = None,
) -> None:
    """Compute the RouteLLM routing threshold from toughness scores."""
    from route_llm.calculate_threshold import calculate_threshold, save_threshold

    threshold = calculate_threshold(
        toughness_path=Path(toughness_path),
        target_strong_rate=target_strong_rate,
    )
    if save_path:
        save_threshold(Path(save_path), threshold)


# =============================================================================
# router (SAE+MLP batch decisions)
# =============================================================================

@router_app.command("batch")
def router_batch(
    split_id: Annotated[int, typer.Option("--split-id", help="DB split id")]          = 1,
    model_id: Annotated[int, typer.Option("--model-id", help="DB model id")]          = 1,
    is_test:  Annotated[bool, typer.Option("--test",    help="Use test partition")]   = True,
    output:   Annotated[str, typer.Option("--output",   help="Output .jsonl path")]   = "routing_decisions.jsonl",
) -> None:
    """Generate SAE+MLP routing decisions for a split and write to DB + JSONL."""
    import torch
    import json
    from mlp.model import MLP, HIDDEN_DIM
    from util import tensor_util
    from util.smart_file_util import mlp_path, write_jsonl
    from daos import model_task_result_dao
    from util.database_connection_util import get_connection

    conn    = get_connection()
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    features_dict = tensor_util.load_features(split_id, model_id)
    features      = features_dict["features"].to(device)
    task_ids      = features_dict["task_ids"]

    weights = mlp_path(split_id, model_id)
    model   = MLP(d_in=features.shape[1], hidden=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(features).cpu()

    # Load existing pass/fail labels for correctness annotation
    results = model_task_result_dao.get_all_for_model_split(conn, model_id, split_id, is_test=is_test)
    label_map = {r.task_id: r.passed for r in results}

    decisions = []
    for task_id, logit in zip(task_ids, logits.tolist()):
        route     = "weak" if logit > 0 else "strong"
        weak_pass = label_map.get(task_id)
        correct   = (route == "weak" and weak_pass) or (route == "strong" and not weak_pass)
        decisions.append({
            "task_id":  task_id,
            "route":    route,
            "logit":    round(logit, 5),
            "correct":  correct,
        })

    write_jsonl(Path(output), decisions)
    typer.echo(f"Routing decisions saved to {output}  ({len(decisions)} problems)")
    conn.close()


# =============================================================================
# test
# =============================================================================

@test_app.command("run")
def test_run(
    results: Annotated[Path, typer.Option("--results", help="Path to inference results .jsonl")],
    model:   Annotated[str,  typer.Option("--model",   help="Model name for output filename")],
) -> None:
    """Evaluate inference results against Docker test cases for all languages."""
    from util.unit_test_util import run_tests
    run_tests(results, model)


# =============================================================================
# stats
# =============================================================================

@stats_app.command("calculate")
def stats_calculate(
    split_id: Annotated[int, typer.Option("--split-id", help="DB split id")] = 1,
    run_id:   Annotated[Optional[int], typer.Option("--run-id", help="DB run id")] = None,
) -> None:
    """Print summary statistics: pass rates, routing breakdown, accuracy."""
    from daos import model_task_result_dao
    from util.database_connection_util import get_connection

    conn  = get_connection()
    rates = model_task_result_dao.get_pass_rates_for_split(conn, split_id, is_test=True)

    typer.echo(f"\n── Results for split {split_id} (test partition) ─────")
    typer.echo(f"{'MODEL':<40} {'PASS':>6} {'TOTAL':>6} {'ACC':>7}")
    typer.echo("-" * 62)
    for row in rates:
        total  = row.total
        passed = row.passed
        acc    = passed / total if total else 0.0
        typer.echo(f"{row.model_name:<40} {passed:>6} {total:>6} {acc:>6.1%}")

    conn.close()


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
    from util.database_connection_util import get_connection

    conn     = get_connection()
    task_ids = set(task_split_dao.get_task_ids_for_split(conn, split_id, is_test=is_test))
    conn.close()

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
