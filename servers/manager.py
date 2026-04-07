"""
servers/manager.py - Lifecycle management for vLLM inference servers and litellm proxy.

Functions
---------
up(weak_gpu, strong_gpu)  Start all three servers, block until Ctrl+C, then clean up.
down()                     Kill servers tracked in .session.json (useful from another terminal).
status()                   Print a health table for each tracked server.
require_up()               Raise typer.Exit(1) with a helpful message if servers are not ready.
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

SESSION_FILE = Path(".session.json")
LOG_DIR      = Path("logs/servers")
MAX_WAIT_SEC = 300

WEAK_MODEL   = "meta-llama/Llama-3.2-1B"
STRONG_MODEL = "meta-llama/Meta-Llama-3-8B"
WEAK_PORT    = 8001
STRONG_PORT  = 8002
PROXY_PORT   = 4000


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
    # On Windows os.kill(SIGTERM) maps to TerminateProcess (immediate).
    # On Unix it sends SIGTERM for graceful shutdown.
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
    ]


def _litellm_cmd() -> list:
    return [sys.executable, "-m", "litellm", "--config", "litellm_config.yaml", "--port", str(PROXY_PORT)]


# ── Public API ────────────────────────────────────────────────────────────────

def up(weak_gpu: str = "0", strong_gpu: str = "1", detach: bool = False) -> None:
    if SESSION_FILE.exists():
        session = json.loads(SESSION_FILE.read_text())
        if all(health_ok(v["port"]) for v in session.values()):
            typer.echo("Servers already running. Use 'status' to see details.")
            return
        SESSION_FILE.unlink()  # stale session

    typer.echo("Checking model weights...")
    _ensure_model(WEAK_MODEL)
    _ensure_model(STRONG_MODEL)

    typer.echo(f"Starting weak model  ({WEAK_MODEL}) on port {WEAK_PORT}  [GPU {weak_gpu}]")
    weak_proc = _spawn(
        _vllm_cmd(WEAK_MODEL, WEAK_PORT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": weak_gpu},
        log_name="weak_vllm",
        detach=detach,
    )

    typer.echo(f"Starting strong model ({STRONG_MODEL}) on port {STRONG_PORT} [GPU {strong_gpu}]")
    strong_proc = _spawn(
        _vllm_cmd(STRONG_MODEL, STRONG_PORT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": strong_gpu},
        log_name="strong_vllm",
        detach=detach,
    )

    # Write session immediately so down() can clean up if Ctrl-C during the wait
    session = {
        "weak_vllm":   {"pid": weak_proc.pid,   "port": WEAK_PORT,   "model": WEAK_MODEL},
        "strong_vllm": {"pid": strong_proc.pid, "port": STRONG_PORT, "model": STRONG_MODEL},
    }
    SESSION_FILE.write_text(json.dumps(session, indent=2))

    _wait_for("weak vLLM",   WEAK_PORT)
    _wait_for("strong vLLM", STRONG_PORT)

    typer.echo(f"Starting litellm proxy on port {PROXY_PORT}")
    proxy_proc = _spawn(_litellm_cmd(), log_name="proxy", detach=detach)
    session["proxy"] = {"pid": proxy_proc.pid, "port": PROXY_PORT}
    SESSION_FILE.write_text(json.dumps(session, indent=2))

    _wait_for("litellm proxy", PROXY_PORT)
    typer.echo(f"\nAll servers ready. Logs: {LOG_DIR}/")

    if detach:
        typer.echo("Running in background. Use 'down' to stop.")
        return

    # Blocking mode — clean up on Ctrl-C
    procs = [weak_proc, strong_proc, proxy_proc]

    def _shutdown(sig=None, frame=None):
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
    if sys.platform != "win32":  # SIGTERM is not delivered on Windows
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
        typer.echo("No active session. Run: python experiment.py up")
        return
    session = json.loads(SESSION_FILE.read_text())
    typer.echo(f"\n{'SERVICE':<14} {'PID':>7} {'PORT':>6}  {'HEALTHY'}")
    typer.echo("-" * 40)
    for name, info in session.items():
        healthy = "yes" if health_ok(info["port"]) else "NO"
        typer.echo(f"{name:<14} {info['pid']:>7} {info['port']:>6}  {healthy}")
    typer.echo("")


def require_up() -> None:
    checks = [
        ("weak vLLM",    WEAK_PORT),
        ("strong vLLM",  STRONG_PORT),
        ("litellm proxy", PROXY_PORT),
    ]
    failing = [name for name, port in checks if not health_ok(port)]
    if failing:
        typer.echo(f"Servers not ready: {', '.join(failing)}", err=True)
        typer.echo("Run: python experiment.py up", err=True)
        raise typer.Exit(1)
