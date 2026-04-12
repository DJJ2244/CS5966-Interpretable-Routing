"""
model_util.py - Lifecycle management for vLLM inference servers and litellm proxy.

Functions
---------
up(models, detach)   Start vLLM servers for each (model_id, gpu_id) pair + litellm proxy.
down()               Kill servers tracked in .session.json.
status()             Print a health table for each tracked server.
require_up()         Exit(1) with a helpful message if servers are not ready.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import typer
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

SESSION_FILE   = Path(".session.json")
LOG_DIR        = Path("logs/servers")
MAX_WAIT_SEC   = 300
BASE_VLLM_PORT = 8001
PROXY_PORT     = 4000
LITELLM_CONFIG = Path("litellm_config.yaml")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_model(model: str) -> None:
    """Download model weights if not cached; no-op if already present."""
    import warnings
    warnings.filterwarnings("ignore", message=".*symlinks.*", category=UserWarning)
    try:
        snapshot_download(repo_id=model, local_files_only=True)
        typer.echo(f"  {model}: cached")
    except LocalEntryNotFoundError:
        typer.echo(f"  {model}: downloading...")
        snapshot_download(repo_id=model)


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2):
            return True
    except Exception:
        return False


def _wait_for(name: str, port: int) -> None:
    typer.echo(f"  Waiting for {name} on port {port} ", nl=False)
    deadline = time.time() + MAX_WAIT_SEC
    while time.time() < deadline:
        if health_ok(port):
            typer.echo(" ready.")
            return
        typer.echo(".", nl=False)
        time.sleep(5)
    typer.echo("")
    typer.echo(f"ERROR: {name} did not become healthy within {MAX_WAIT_SEC}s.", err=True)
    raise typer.Exit(1)


def _spawn(cmd: list, env: dict = None, log_name: str = "server", detach: bool = False) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"{log_name}.log", "w")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT}
    if detach:
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, env=env, **kwargs)


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def _vllm_cmd(model: str, port: int) -> list:
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",         model,
        "--dtype",         "float16",
        "--port",          str(port),
        "--max-model-len", "4096",
        "--enforce-eager",
        "--max-num-seqs",  "1",
    ]


def _litellm_cmd() -> list:
    return ["litellm", "--config", str(LITELLM_CONFIG), "--port", str(PROXY_PORT)]


def _write_litellm_config(entries: list[tuple[str, int]]) -> None:
    """Write litellm_config.yaml from (model_id, port) pairs."""
    lines = ["model_list:"]
    for model_id, port in entries:
        lines += [
            f"  - model_name: {model_id}",
            f"    litellm_params:",
            f"      model: openai/{model_id}",
            f"      api_base: http://localhost:{port}/v1",
            f"      api_key: dummy",
            f"",
        ]
    LITELLM_CONFIG.write_text("\n".join(lines))


# ── Public API ────────────────────────────────────────────────────────────────

def up(models: list[tuple[str, str]], detach: bool = False) -> None:
    """Start one vLLM server per (model_id, gpu_id) pair, then start the litellm proxy.

    Args:
        models: List of (model_id, gpu_id) tuples. Ports are assigned sequentially
                from BASE_VLLM_PORT (8001, 8002, ...).
        detach: If True, run all servers in the background and return immediately.
    """
    if SESSION_FILE.exists():
        session = json.loads(SESSION_FILE.read_text())
        if all(health_ok(v["port"]) for v in session.values()):
            typer.echo("Servers already running. Use 'status' to see details.")
            return
        SESSION_FILE.unlink()  # stale session

    typer.echo("Checking model weights...")
    for model_id, _ in models:
        _ensure_model(model_id)

    session = {}
    procs   = []
    entries = []  # (model_id, port) for litellm config

    for i, (model_id, gpu_id) in enumerate(models):
        port     = BASE_VLLM_PORT + i
        key      = f"vllm_{i}"
        log_name = f"vllm_{i}"
        typer.echo(f"Starting {model_id} on port {port} [GPU {gpu_id}]")
        proc = _spawn(
            _vllm_cmd(model_id, port),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id},
            log_name=log_name,
            detach=detach,
        )
        session[key] = {"pid": proc.pid, "port": port, "model": model_id}
        procs.append(proc)
        entries.append((model_id, port))

    SESSION_FILE.write_text(json.dumps(session, indent=2))

    for i, (model_id, _) in enumerate(models):
        _wait_for(model_id, BASE_VLLM_PORT + i)

    _write_litellm_config(entries)
    typer.echo(f"Starting litellm proxy on port {PROXY_PORT}")
    proxy_proc = _spawn(_litellm_cmd(), log_name="proxy", detach=detach)
    session["proxy"] = {"pid": proxy_proc.pid, "port": PROXY_PORT}
    SESSION_FILE.write_text(json.dumps(session, indent=2))
    procs.append(proxy_proc)

    _wait_for("litellm proxy", PROXY_PORT)
    typer.echo(f"\nAll servers ready. Logs: {LOG_DIR}/")

    if detach:
        typer.echo("Running in background. Use 'down' to stop.")
        return

    def _shutdown(_sig=None, _frame=None):
        typer.echo("\nShutting down servers...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
        typer.echo("All servers stopped.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _shutdown)

    typer.echo("Press Ctrl+C to stop.\n")
    while True:
        for p in procs:
            if p.poll() is not None:
                typer.echo(f"ERROR: server (pid {p.pid}) exited unexpectedly.", err=True)
                _shutdown()
        time.sleep(5)


def down() -> None:
    if not SESSION_FILE.exists():
        typer.echo("No active session.")
        return
    session = json.loads(SESSION_FILE.read_text())
    for name, info in session.items():
        _kill(info["pid"])
        typer.echo(f"  Stopped {name} (pid {info['pid']})")
    SESSION_FILE.unlink()
    typer.echo("All servers stopped.")


def status() -> None:
    if not SESSION_FILE.exists():
        typer.echo("No active session. Run: python cli.py server up")
        return
    session = json.loads(SESSION_FILE.read_text())
    typer.echo(f"\n{'SERVICE':<14} {'PID':>7} {'PORT':>6}  {'HEALTHY'}")
    typer.echo("-" * 40)
    for name, info in session.items():
        healthy = "yes" if health_ok(info["port"]) else "NO"
        typer.echo(f"{name:<14} {info['pid']:>7} {info['port']:>6}  {healthy}")
    typer.echo("")


def require_up() -> None:
    if not SESSION_FILE.exists():
        typer.echo("No active session. Run: python cli.py server up", err=True)
        raise typer.Exit(1)
    session = json.loads(SESSION_FILE.read_text())
    failing = [name for name, info in session.items() if not health_ok(info["port"])]
    if failing:
        typer.echo(f"Servers not ready: {', '.join(failing)}", err=True)
        typer.echo("Run: python cli.py server up", err=True)
        raise typer.Exit(1)
