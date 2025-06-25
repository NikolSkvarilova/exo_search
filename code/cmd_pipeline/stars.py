import typer
import cmd_pipeline.tess as tess

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Creating star lists.")
app.add_typer(tess.app, name="tess", help="Commands for working with TESS data.")

"""For accessing the command line tools regarding star lists."""

if __name__ == "__main__":
    app()
