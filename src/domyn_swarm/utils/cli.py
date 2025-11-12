from rich.console import Console
from rich.table import Table
import typer


def _pick_one(names: list[str], console: Console) -> str:
    table = Table(title="Matching swarms")
    table.add_column("#", justify="right")
    table.add_column("deployment_name")
    for i, n in enumerate(names, 1):
        table.add_row(str(i), n)
    console.print(table)
    choice = typer.prompt("Select a swarm to cancel (number)")
    try:
        idx = int(choice)
        if not (1 <= idx <= len(names)):
            raise ValueError
        return names[idx - 1]
    except Exception as exc:
        raise typer.BadParameter("Invalid selection.") from exc
