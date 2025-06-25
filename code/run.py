import typer

import cmd_pipeline.pipeline as pipeline
import cmd_pipeline.stars as stars
from pathlib import Path

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(stars.app, name="star-list")
app.add_typer(pipeline.app, name="pipeline")

"""For accessing the command line tools."""

if __name__ == "__main__":
    app()
