"""
experiment.py - CS5966 Interpretable Routing experiment CLI.

Session commands (run once per working session):
  python experiment.py up        Start vLLM servers + litellm proxy
  python experiment.py down      Stop all servers
  python experiment.py status    Show server health

Experiment commands (require servers to be up):
  python experiment.py inference --model weak|strong|all
  python experiment.py route
  python experiment.py toughness
  python experiment.py test --results <path>
  python experiment.py run-all

Utility:
  python experiment.py split     Regenerate train/test data split
"""

from dotenv import load_dotenv
load_dotenv()  # must run before any route_llm_inference imports read os.environ

from pathlib import Path
from typing import Annotated
import typer

app = typer.Typer(
    help="CS5966 Interpretable Routing — experiment CLI.",
    no_args_is_help=True,
)

# ── Type aliases ──────────────────────────────────────────────────────────────

_Model  = Annotated[str,  typer.Option("--model",      help="weak | strong | all")]
_Split  = Annotated[str,  typer.Option("--split",      help="train | test")]
_OutDir = Annotated[str,  typer.Option("--output-dir", help="Directory for result .jsonl files")]


# ── Session management ────────────────────────────────────────────────────────

@app.command()
def up(
    weak_gpu:   Annotated[str,  typer.Option("--weak-gpu",   help="CUDA device for weak model")]              = "0",
    strong_gpu: Annotated[str,  typer.Option("--strong-gpu", help="CUDA device for strong model")]            = "1",
    single_gpu: Annotated[bool, typer.Option("--single-gpu", help="Run both models on GPU 0")]                = False,
    detach:     Annotated[bool, typer.Option("--detach",     help="Run servers in background (no blocking)")] = False,
) -> None:
    """Start vLLM inference servers and litellm proxy."""
    from servers.manager import up as _up
    if single_gpu:
        strong_gpu = weak_gpu
    _up(weak_gpu=weak_gpu, strong_gpu=strong_gpu, detach=detach)


@app.command()
def down() -> None:
    """Stop all inference servers."""
    from servers.manager import down as _down
    _down()


@app.command()
def status() -> None:
    """Show server health and PIDs."""
    from servers.manager import status as _status
    _status()


# ── Experiment commands ───────────────────────────────────────────────────────

@app.command()
def inference(
    model:      _Model  = "all",
    split:      _Split  = "train",
    output_dir: _OutDir = "route_llm_results",
    workers:    Annotated[int, typer.Option("--workers", help="Concurrent inference requests")] = 8,
) -> None:
    """Run baseline inference (no routing) for weak, strong, or both models."""
    from servers.manager import require_up
    from route_llm_inference.baseline import run_baseline
    require_up()
    run_baseline(model=model, split=split, output_dir=output_dir, max_workers=workers)


@app.command()
def route(
    split:      _Split  = "train",
    output_dir: _OutDir = "route_llm_results",
) -> None:
    """Route each problem to weak or strong model via the BERT router."""
    from servers.manager import require_up
    from route_llm_inference.routing import run_routing
    require_up()
    run_routing(split=split, output_dir=output_dir)


@app.command()
def toughness(
    split:      _Split  = "train",
    output_dir: _OutDir = "route_llm_results",
) -> None:
    """Score all problems with the BERT router. No model inference required."""
    from route_llm_inference.toughness import record_toughness
    record_toughness(split=split, output_dir=output_dir)


@app.command()
def test(
    results: Annotated[Path, typer.Option("--results", help="Path to inference results .jsonl")],
    result_name: Annotated[str, typer.Option("--name", help="Value to append to the results file")]
) -> None:
    if result_name == None:
        result_name = ""
    """Evaluate inference results against test cases using Docker."""
    from testing.runner import run_tests
    run_tests(results, result_name)


@app.command()
def run_all(
    split:      _Split  = "train",
    output_dir: _OutDir = "route_llm_results",
) -> None:
    """Run the full pipeline: inference (both) → route → toughness → test."""
    from servers.manager import require_up
    from route_llm_inference.baseline import run_baseline
    from route_llm_inference.routing import run_routing
    from route_llm_inference.toughness import record_toughness
    from testing.runner import run_tests

    require_up()

    run_baseline(model="all",  split=split, output_dir=output_dir)
    run_routing(               split=split, output_dir=output_dir)
    record_toughness(          split=split, output_dir=output_dir)

    for name in ("results_weak", "results_strong", "router_results"):
        p = Path(output_dir) / f"{name}.jsonl"
        if p.exists():
            run_tests(p)


# ── Utility ───────────────────────────────────────────────────────────────────

@app.command()
def login() -> None:
    """Authenticate with HuggingFace to access gated models (Llama)."""
    from huggingface_hub import login as hf_login
    hf_login()


@app.command()
def split() -> None:
    """Regenerate the stratified train/test data split."""
    from util.split import split as _split
    _split()


if __name__ == "__main__":
    app()
