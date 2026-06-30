"""
main.py — CLI Entry Point for the Agentic Codebase Intelligence Platform.

Usage:
    # Point at ANY folder on your machine — no copying needed
    python main.py ingest --repo /path/to/your/project
    python main.py query  --repo /path/to/your/project

    # Or set once as env var and omit --repo everywhere
    export ACIP_REPO=/path/to/your/project
    python main.py ingest
    python main.py query

    # One-shot query
    python main.py query --repo ~/myproject --ask "Explain the architecture"

    # Start the REST API
    python main.py serve --repo ~/myproject

Run `python main.py --help` for full options.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

for _noisy in ("httpx", "openai", "httpcore", "faiss"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _resolve_repo(args: argparse.Namespace) -> Path:
    """
    Resolve the target repo directory from (in priority order):
      1. --repo CLI flag
      2. ACIP_REPO environment variable
      3. data/repo/ subfolder next to main.py  (original default)
    """
    if getattr(args, "repo", None):
        p = Path(args.repo).expanduser().resolve()
    elif os.environ.get("ACIP_REPO"):
        p = Path(os.environ["ACIP_REPO"]).expanduser().resolve()
    else:
        p = Path(__file__).parent / "data" / "repo"

    if not p.exists():
        print(f"[ERROR] Repo path does not exist: {p}")
        print("  Pass a valid path with --repo /your/project")
        sys.exit(1)

    return p


def _apply_repo_to_config(repo_path: Path) -> None:
    """Override DATA_DIR in config at runtime so all modules see the right path."""
    import config
    config.DATA_DIR = repo_path


# ── Ingestion ─────────────────────────────────────────────────────────────────

def run_ingestion(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    console = Console()

    repo_path = _resolve_repo(args)
    _apply_repo_to_config(repo_path)

    console.rule(f"[bold cyan]INGESTION PIPELINE")
    console.print(f"  Repo: [bold]{repo_path}[/bold]")

    from ingestion.indexer import get_index
    from ingestion.pipeline import ingest_repository
    from agents.graph_agent import GraphAgent
    import config

    index = get_index()
    graph = GraphAgent()

    async def _run() -> object:
        manifest_path = config.VECTOR_STORE_DIR / "manifest.json"
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()) as prog:
            task = prog.add_task("Starting ingestion...")

            async def _progress(phase: str, message: str, _pct: int, extra: dict) -> None:
                prog.update(task, description=f"{phase}: {message}")

            return await ingest_repository(
                repo_path=repo_path,
                index=index,
                graph=graph,
                manifest_path=manifest_path,
                progress=_progress,
            )

    result = asyncio.run(_run())
    files = result.files
    if not files:
        console.print(
            f"[red]No supported source files found in:[/red] {repo_path}\n"
            f"Supported extensions: {config.SUPPORTED_EXTENSIONS}\n"
            "Check the path or add extensions in config.py."
        )
        sys.exit(1)

    console.print(f"[cyan]Changed files:[/cyan] {len(result.changed_files)} | [cyan]Reused:[/cyan] {len(result.reused_files)}")
    if result.changed_files:
        console.print("  -> " + ", ".join(result.changed_files[:8]) + (" ..." if len(result.changed_files) > 8 else ""))
    console.print(f"[cyan]Chunking/index complete:[/cyan] {len(result.chunks)} chunks, {index.num_chunks} vectors")
    stats = result.graph_stats
    console.print(f"[cyan]Dependency graph:[/cyan] {stats['nodes']} nodes, {stats['edges']} edges")

    console.rule("[bold green]Ingestion Complete")
    console.print(
        f"  Repo:   [bold]{repo_path}[/bold]\n"
        f"  Files:  [bold]{len(files)}[/bold]  "
        f"Chunks: [bold]{len(result.chunks)}[/bold]  "
        f"Graph nodes: [bold]{stats['nodes']}[/bold]"
    )
    console.print(f"\n  Now run:  [bold]python main.py query --repo {repo_path}[/bold]")


# ── Query ─────────────────────────────────────────────────────────────────────

def run_query(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    repo_path = _resolve_repo(args)
    _apply_repo_to_config(repo_path)

    console.print(f"[cyan]Loading index and graph ...[/cyan]")
    from ingestion.indexer import get_index
    from agents.graph_agent import GraphAgent
    from agents.orchestrator import Orchestrator

    index = get_index()
    if not index.load():
        console.print(
            "[red]No index found.[/red] Run ingestion first:\n"
            f"  [bold]python main.py ingest --repo {repo_path}[/bold]"
        )
        sys.exit(1)

    graph = GraphAgent()
    graph.load()

    orchestrator = Orchestrator(vector_index=index, graph_agent=graph)
    console.print(f"[green]System ready.[/green]  Repo: [bold]{repo_path}[/bold]\n")

    if args.ask:
        _execute_single_query(args.ask, orchestrator, console)
        return

    console.print(Panel(
        "[bold]Agentic Codebase Intelligence Platform[/bold]\n"
        f"Repo: {repo_path}\n"
        "Commands: [bold]/history[/bold]  [bold]/files[/bold]  [bold]/quit[/bold]",
        title="ACIP",
        style="cyan",
    ))

    while True:
        try:
            user_input = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        elif user_input.lower() == "/history":
            for e in orchestrator.memory.all_entries()[-10:]:
                ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
                console.print(f"  [{ts}] {e.query[:80]}")
            continue
        elif user_input.lower() == "/files":
            for f in orchestrator.tools.list_files():
                console.print(f"  {f}")
            continue

        _execute_single_query(user_input, orchestrator, console)


def _execute_single_query(query: str, orchestrator, console) -> None:
    from rich.markdown import Markdown
    from rich.panel import Panel

    console.print("\n[dim]Planning and executing ...[/dim]")
    t0 = time.perf_counter()
    resp = orchestrator.query(query)
    elapsed = time.perf_counter() - t0

    console.print(Panel(Markdown(resp.answer), title="Answer", style="green"))

    meta = [f"Plan: {' -> '.join(resp.plan)}"]
    if resp.evaluation:
        meta.append(f"Score: {resp.evaluation.overall_score:.2f}")
    meta.append(f"Time: {elapsed:.1f}s")
    meta.append(f"Chunks: {resp.metrics.get('chunks_searched', '?')}")
    console.print("[dim]" + " | ".join(meta) + "[/dim]")

    if resp.evaluation and resp.evaluation.issues:
        console.print(f"[yellow]Issues: {', '.join(resp.evaluation.issues)}[/yellow]")


# ── Serve ─────────────────────────────────────────────────────────────────────

def run_serve(args: argparse.Namespace) -> None:
    import uvicorn
    import config

    repo_path = _resolve_repo(args)
    _apply_repo_to_config(repo_path)
    os.environ["ACIP_REPO"] = str(repo_path)

    logger.info("Starting API server on %s:%d  repo=%s", config.API_HOST, config.API_PORT, repo_path)
    uvicorn.run("api.server:app", host=config.API_HOST, port=config.API_PORT,
                reload=args.reload, log_level="info")


# ── CLI parser ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acip",
        description="Agentic Codebase Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest --repo ~/projects/myapp
  python main.py query  --repo ~/projects/myapp
  python main.py query  --repo ~/projects/myapp --ask "Explain the architecture"
  python main.py serve  --repo ~/projects/myapp

  # Set once, skip --repo everywhere:
  export ACIP_REPO=~/projects/myapp
  python main.py ingest && python main.py query
        """,
    )

    repo_kwargs = dict(
        metavar="PATH",
        default=None,
        help="Path to the codebase to analyse. Defaults to $ACIP_REPO, then data/repo/.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Parse, embed, and index a repository.")
    ingest_p.add_argument("--repo", **repo_kwargs)
    ingest_p.set_defaults(func=run_ingestion)

    query_p = sub.add_parser("query", help="Ask questions about an indexed repository.")
    query_p.add_argument("--repo", **repo_kwargs)
    query_p.add_argument("--ask", default="", metavar="QUESTION", help="Single non-interactive question.")
    query_p.set_defaults(func=run_query)

    serve_p = sub.add_parser("serve", help="Start the FastAPI REST server.")
    serve_p.add_argument("--repo", **repo_kwargs)
    serve_p.add_argument("--reload", action="store_true")
    serve_p.set_defaults(func=run_serve)

    return parser


def run_web(args: argparse.Namespace) -> None:
    """Start the web UI server (opens browser automatically)."""
    import uvicorn, webbrowser, threading
    import config

    repo_path = _resolve_repo(args) if getattr(args, 'repo', None) or os.environ.get('ACIP_REPO') else None
    if repo_path:
        _apply_repo_to_config(repo_path)
        os.environ["ACIP_REPO"] = str(repo_path)

    port = int(os.environ.get("PORT", config.API_PORT))
    url = f"http://localhost:{port}"
    print(f"\n  ACIP Web UI → {url}\n  Press Ctrl+C to stop.\n")
    logger.info("Web server ready at %s. Waiting for browser requests.", url)

    def _open():
        import time; time.sleep(1.2)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

    uvicorn.run("api.server:app", host="0.0.0.0", port=port,
                reload=getattr(args, 'reload', False), log_level="warning")


def build_full_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    sub = parser._subparsers._group_actions[0]
    web_p = sub.add_parser("web", help="Start the web UI (recommended entry point).")
    web_p.add_argument("--repo", metavar="PATH", default=None,
                       help="Optional: pre-load a repo at startup.")
    web_p.add_argument("--reload", action="store_true")
    web_p.set_defaults(func=run_web)
    return parser


def main() -> None:
    build_full_parser().parse_args().func(build_full_parser().parse_args())


if __name__ == "__main__":
    main()
