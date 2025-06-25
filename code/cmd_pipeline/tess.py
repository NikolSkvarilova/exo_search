import typer
from pathlib import Path
from typing_extensions import Annotated

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def from_project_candidates(
    filepath: Annotated[Path, typer.Argument(help="File path to TESS project candidates CSV.", show_default=False)], 
    dst_filepath: Annotated[Path, typer.Argument(help="File path for the new star list.", show_default=False)], 
    disposition: str = typer.Option(None, "--disposition", help="Only stars with at least one exoplanet with this disposition will be chosen.", show_default=False), 
    strict: bool = typer.Option(False, "--strict/--no-strict", help="Only stars with all exoplanets matching the disposition will be chosen.")
):
    """Create star list from TESS project candidates table."""
    import exo_search.pipeline.tess.project_candidates_transformer as project_candidates_transformer

    project_candidates_transformer.main(filepath, dst_filepath, disposition, strict)


@app.command()
def exoplanets(
    filepath: Annotated[Path, typer.Argument(help="File path to TESS project candidates CSV.", show_default=False)], 
    dst_filepath: Annotated[Path, typer.Argument(help="File path for the exoplanets CSV.", show_default=False)]):
    """Create CSV with exoplanets info for stars."""
    import exo_search.pipeline.tess.get_exoplanet_info as get_exoplanet_info

    get_exoplanet_info.main(filepath, dst_filepath)


if __name__ == "__main__":
    app()
