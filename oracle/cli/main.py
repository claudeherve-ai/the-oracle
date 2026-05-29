"""The Oracle CLI — beautiful terminal interface."""

import time
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
import httpx

app = typer.Typer(name="oracle", help="The Oracle — Predictive Intelligence Engine", add_completion=False)
console = Console()
API_URL = "http://localhost:8001/v1"

GREEN = "green"
CYAN = "cyan"
DIM = "dim"
RED = "red"
YELLOW = "yellow"
BOLD = "bold"

CATEGORY_COLORS = {
    "tech_trend": "cyan", "product_launch": "magenta", "market_move": "green",
    "regulatory": "yellow", "startup_success": "blue", "culture": "red", "github_trend": "cyan",
}


@app.command()
def predict(
    question: str = typer.Argument(..., help="Question to generate predictions for"),
    count: int = typer.Option(5, help="Max predictions"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="API endpoint"),
):
    """Generate predictions for a specific question."""
    client = httpx.Client(base_url=endpoint, timeout=120)
    console.print()
    console.print(Panel.fit("[bold green]THE ORACLE[/bold green]\n[dim]Predictive Intelligence Engine[/dim]", border_style="green"))
    console.print(f"[dim]Question:[/dim] {question}")
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[green]Consulting the oracle...", total=None)
        try:
            r = client.post("/predict/query", json={"question": question, "max_predictions": count})
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1)

    predictions = data.get("predictions", [])
    _display_predictions(predictions)
    client.close()


@app.command()
def scan(
    count: int = typer.Option(5, help="Max predictions"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="API endpoint"),
):
    """Auto-generate predictions from current signals."""
    client = httpx.Client(base_url=endpoint, timeout=120)
    console.print()
    console.print(Panel.fit("[bold green]THE ORACLE[/bold green]\n[dim]Scanning signals for predictions...[/dim]", border_style="green"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("[green]Ingesting signals...", total=None)
        try:
            r = client.post("/ingest")
            r.raise_for_status()
            data = r.json()
            console.print(f"[dim]  Ingested {data['ingested']} signals[/dim]")
        except httpx.HTTPError:
            pass

        task2 = progress.add_task("[green]Generating predictions...", total=None)
        try:
            r = client.post("/predict", json={"question": "", "max_predictions": count})
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1)

    predictions = data.get("predictions", [])
    _display_predictions(predictions)
    client.close()


@app.command()
def predictions(
    category: Optional[str] = typer.Option(None, help="Filter by category"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="API endpoint"),
):
    """List predictions."""
    client = httpx.Client(base_url=endpoint)
    params = {"limit": 50}
    if category:
        params["category"] = category
    if status:
        params["status"] = status
    try:
        r = client.get("/predictions", params=params)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)

    items = data.get("items", [])
    if not items:
        console.print("[dim]No predictions yet.[/dim]")
        return

    _display_predictions(items)
    client.close()


@app.command()
def resolve(
    prediction_id: str = typer.Argument(..., help="Prediction ID"),
    outcome: str = typer.Option(..., help="correct or incorrect"),
    note: Optional[str] = typer.Option(None, help="Resolution note"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="API endpoint"),
):
    """Resolve a prediction."""
    client = httpx.Client(base_url=endpoint)
    try:
        r = client.post(f"/predictions/{prediction_id}/resolve", json={
            "outcome": outcome, "resolution": note,
        })
        r.raise_for_status()
        console.print(f"[green]✓ Prediction resolved as {outcome}[/green]")
    except httpx.HTTPError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)
    client.close()


@app.command()
def calibration(
    category: Optional[str] = typer.Option(None, help="Filter by category"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="API endpoint"),
):
    """Show calibration report."""
    client = httpx.Client(base_url=endpoint)
    params = {}
    if category:
        params["category"] = category
    try:
        r = client.get("/calibration", params=params)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel.fit("[bold green]CALIBRATION REPORT[/bold green]", border_style="green"))
    accuracy = data.get("overall_accuracy", 0)
    color = "green" if accuracy >= 0.7 else "yellow" if accuracy >= 0.5 else "red"
    console.print(f"Overall Accuracy: [{color}]{accuracy:.1%}[/{color}]")
    console.print(f"Total resolved: {data.get('overall_total', 0)}")
    console.print()

    buckets = data.get("buckets", [])
    if buckets:
        table = Table(title="By Confidence Bucket", border_style="green")
        table.add_column("Confidence", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Correct", justify="right")
        table.add_column("Accuracy", justify="right")
        for b in buckets:
            acc = b.get("accuracy", 0)
            c = "green" if acc >= 0.7 else "yellow"
            table.add_row(b["confidence_range"], str(b["total"]), str(b["correct"]), f"[{c}]{acc:.1%}[/{c}]")
        console.print(table)
    client.close()


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8001, help="Port"),
):
    """Start the API server."""
    import os
    os.environ["ORACLE_HOST"] = host
    os.environ["ORACLE_PORT"] = str(port)
    console.print(f"[bold green]The Oracle[/bold green] on [cyan]{host}:{port}[/cyan]")
    from oracle.api.app import main
    main()


@app.command()
def dashboard(
    endpoint: str = typer.Option("http://localhost:8001", help="API endpoint"),
):
    """Open the calibration dashboard."""
    import webbrowser
    url = f"{endpoint}/dashboard/index.html"
    console.print(f"[green]Opening dashboard: {url}[/green]")
    webbrowser.open(url)


def _display_predictions(items):
    console.print()
    for i, p in enumerate(items):
        conf = p["confidence"]
        c = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "red"
        bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        cat_color = CATEGORY_COLORS.get(p["category"], "dim")
        console.print(f"[bold]{i+1}.[/bold] {p['statement']}")
        console.print(f"   Confidence: [{c}]{bar}[/{c}] [{c}]{conf:.0%}[/{c}]  |  [{cat_color}]{p['category']}[/{cat_color}]")
        if p.get("deadline"):
            console.print(f"   Deadline: [dim]{p['deadline'][:10]}[/dim]")
        if p.get("reasoning"):
            console.print(f"   [dim]{p['reasoning'][:120]}...[/dim]")
        console.print()


if __name__ == "__main__":
    app()
