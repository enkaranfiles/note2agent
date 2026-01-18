"""
CLI Interface

Command-line interface for the RAG system.
Provides commands: query, refresh, status, clear.
"""

import os
import sys
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from core.vector_store import VectorStore
from core.knowledge_base import KnowledgeBase


# Load environment variables
load_dotenv()

console = Console()


def get_vector_store():
    """Initialize and return vector store instance"""
    voyage_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_key:
        console.print("\n[red]❌ ERROR: VOYAGE_API_KEY not found in .env file[/red]")
        console.print("[yellow]Please add your Voyage AI API key to .env file[/yellow]\n")
        sys.exit(1)

    try:
        vector_store = VectorStore(
            collection_name="note2agent_docs",
            chromadb_host="localhost",
            chromadb_port=8000
        )

        # Health check
        if not vector_store.health_check():
            console.print("\n[red]❌ ERROR: Cannot connect to ChromaDB[/red]")
            console.print("[yellow]Make sure ChromaDB is running:[/yellow]")
            console.print("[cyan]docker-compose -f docker-compose.chromadb.yml up -d[/cyan]\n")
            sys.exit(1)

        return vector_store

    except Exception as e:
        console.print(f"\n[red]❌ ERROR: {str(e)}[/red]\n")
        sys.exit(1)


def get_knowledge_base():
    """Initialize and return knowledge base instance"""
    vector_store = get_vector_store()
    return KnowledgeBase(vector_store)


@click.group()
def cli():
    """Note2Agent - RAG Framework with LangGraph"""
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='Force re-index all files')
@click.option('--full', is_flag=True, help='Full refresh (not incremental)')
def refresh(path, force, full):
    """
    Refresh knowledge base from documents directory.

    PATH: Directory containing your documents (PDF, Markdown, TXT)

    Examples:
        note2agent refresh ./data/documents
        note2agent refresh ./data/documents --force
    """
    console.print("\n[bold cyan]🔄 Refreshing Knowledge Base[/bold cyan]\n")

    kb = get_knowledge_base()

    # Run refresh
    incremental = not full
    stats = kb.refresh(path, incremental=incremental, force=force)

    # Display summary with Rich
    if stats['errors']:
        console.print(f"\n[yellow]⚠️  Completed with {len(stats['errors'])} error(s)[/yellow]")
        for error in stats['errors']:
            console.print(f"   [red]{error}[/red]")
    else:
        console.print("\n[green]✅ Refresh completed successfully![/green]")


@cli.command()
def status():
    """Show knowledge base status and statistics"""
    console.print("\n[bold cyan]📊 Knowledge Base Status[/bold cyan]\n")

    kb = get_knowledge_base()
    status_info = kb.get_status()

    # Create status table
    table = Table(title="System Status", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Collection Name", status_info['vector_store']['collection_name'])
    table.add_row("Total Chunks", str(status_info['vector_store']['total_chunks']))
    table.add_row("Indexed Files", str(status_info['indexed_files']))
    table.add_row("Embedding Model", status_info['vector_store']['model'])
    table.add_row("Last Updated", status_info['last_updated'] or "Never")
    table.add_row(
        "ChromaDB Status",
        "[green]✓ Healthy[/green]" if status_info['chromadb_healthy'] else "[red]✗ Unhealthy[/red]"
    )

    console.print(table)

    # List indexed files
    files = kb.list_indexed_files()
    if files:
        console.print("\n[bold]Indexed Files:[/bold]")
        for file_info in files:
            file_path = file_info['path']
            file_name = os.path.basename(file_path)
            console.print(f"  • [cyan]{file_name}[/cyan] ({file_info['chunks']} chunks)")

    console.print()


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear the entire knowledge base?')
def clear():
    """Clear the entire knowledge base"""
    console.print("\n[bold red]🗑️  Clearing Knowledge Base[/bold red]\n")

    kb = get_knowledge_base()
    kb.clear()

    console.print("[green]✅ Knowledge base cleared successfully[/green]\n")

if __name__ == '__main__':
    cli()
