""" main.py — Interactive RAG chat interface. """
 
import argparse
import sys
 
from rich.console   import Console
from rich.panel     import Panel
from rich.markdown  import Markdown
from rich.rule      import Rule
 
from src.rag_chain   import ask
from src.vectorstore import collection_size
 
console = Console()
 
COMMANDS = {
    "/exit":    "Quit the chat",
    "/quit":    "Quit the chat",
    "/sources": "Toggle showing retrieved source chunks",
    "/help":    "Show this help message",
    "/stats":   "Show vector store stats",
}
 
 
def print_sources(sources: list[dict]) -> None:
    console.print("\n[dim]── Retrieved context ──────────────────────────────[/dim]")
    for i, src in enumerate(sources, 1):
        import os
        fname = os.path.basename(src["source"])
        score = src["score"]
        console.print(f"[dim][{i}] {fname} · page {src['page']} · score {score}[/dim]")
        console.print(f"[dim]{src['text'][:200].strip()}…[/dim]\n")
 
 
def print_help() -> None:
    console.print("\n[bold]Commands:[/bold]")
    for cmd, desc in COMMANDS.items():
        console.print(f"  [cyan]{cmd:<14}[/cyan] {desc}")
    console.print()
 
 
def run_chat(stream: bool, top_k: int, show_sources: bool, single_query: str | None) -> None:
    console.print(Panel.fit(
        "[bold blue]RAG Chat[/bold blue]\n"
        f"[dim]Vector store: {collection_size()} chunks  |  "
        f"top-k: {top_k}  |  streaming: {stream}[/dim]\n"
        "[dim]Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit[/dim]",
    ))
 
    
    if single_query:                                                    # Single-query mode (non-interactive)
        console.print(f"\n[bold]Q:[/bold] {single_query}\n")
        result = ask(single_query, top_k=top_k, stream=stream)
        if not stream:
            console.print(Markdown(result["answer"]))
        if show_sources:
            print_sources(result["sources"])
        return
 
    
    while True:                                                         # Interactive loop
        try:
            query = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
 
        if not query:
            continue
 
        
        if query.startswith("/"):                                       # Handle commands
            cmd = query.lower()
            if cmd in ("/exit", "/quit"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/sources":
                show_sources = not show_sources
                console.print(f"[dim]Source display: {'ON' if show_sources else 'OFF'}[/dim]")
            elif cmd == "/help":
                print_help()
            elif cmd == "/stats":
                console.print(f"[dim]Chunks in store: {collection_size()}[/dim]")
            else:
                console.print(f"[red]Unknown command.[/red] Type /help for options.")
            continue
 
        
        try:                                                            # Run RAG pipeline
            console.print(Rule(style="dim"))
            console.print("[bold blue]Assistant:[/bold blue] ", end="")
 
            result = ask(query, top_k=top_k, stream=stream)
 
            if not stream:
                console.print(Markdown(result["answer"]))
 
            if show_sources:
                print_sources(result["sources"])
 
        except RuntimeError as e:
            console.print(f"\n[red]Error:[/red] {e}")
            console.print("[yellow]Hint:[/yellow] Run [bold]python ingest.py[/bold] first.")
 
 
def main():
    parser = argparse.ArgumentParser(description="RAG chat interface.")
    parser.add_argument("--no-stream",    dest="stream",       action="store_false", default=True,
                        help="Disable streaming output")
    parser.add_argument("--top-k",        dest="top_k",        type=int, default=None,
                        help="Number of chunks to retrieve (overrides config)")
    parser.add_argument("--show-sources", dest="show_sources", action="store_true",  default=False,
                        help="Print retrieved source chunks after each answer")
    parser.add_argument("-q", "--query",  dest="query",        default=None,
                        help="Single query to answer, then exit")
    args = parser.parse_args()
 
    from config import TOP_K_RESULTS
    top_k = args.top_k or TOP_K_RESULTS
 
    try:
        run_chat(
            stream       = args.stream,
            top_k        = top_k,
            show_sources = args.show_sources,
            single_query = args.query,
        )
    except EnvironmentError as e:
        console.print(f"\n[red]Config error:[/red] {e}")
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()