"""Torq CLI — stub entry point.

The full CLI (tq ingest, tq list, tq info, tq export) is implemented
in later stories. Running `tq` before then shows a status message.
"""

import typer

app = typer.Typer(
    name="tq",
    help="Torq — Robot Learning Data Infrastructure SDK",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Torq CLI — commands not yet implemented."""
    if ctx.invoked_subcommand is None:
        typer.echo("torq v0.1.0-alpha — CLI commands not yet implemented.")
        typer.echo("Use the Python SDK: import torq as tq")
        raise typer.Exit(0)
