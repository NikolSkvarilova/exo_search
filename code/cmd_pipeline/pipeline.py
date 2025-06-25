import typer
from pathlib import Path
from enum import Enum
from typing import List, Tuple
from typing_extensions import Annotated

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Pipeline commands.")


class Kernel(str, Enum):
    """Default kernel types."""

    se = "se"
    periodic = "periodic"
    matern12 = "matern12"
    matern32 = "matern32"
    matern52 = "matern52"
    white = "white"
    se_periodic = "se+periodic"


class DownsamplingMethod(str, Enum):
    """Available downsampling methods."""

    moving_avg = "moving-avg"
    every_nth = "every-nth"


class Comparator(str, Enum):
    """Available comparators for filtering."""

    larger = ">"
    smaller = "<"
    equal = "="
    not_equal = "!="


@app.command()
def create_stars(
    stars_csv: Annotated[Path, typer.Argument(help="Star list file path.", show_default=False)], 
    star_path: Annotated[Path, typer.Argument(help="Directory for new stars.", show_default=False)], 
    exoplanets_csv: Path = typer.Option(None, "--exoplanets-csv", help="File path to CSV with exoplanets info.", show_default=False, ), 
):
    """Create stars from star list."""
    import exo_search.pipeline.create_stars as create_stars

    create_stars.main(stars_csv, star_path, exoplanets_csv)


@app.command()
def download_lc(
    stars_path: Annotated[Path, typer.Argument(help="Directory with stars.", show_default=False)], 
    mission: Annotated[str, typer.Argument(    help="Mission to filter the light curves. Example: TESS.",    show_default=False, ), ], 
    author: Annotated[str, typer.Argument(    help="Author to filter the light curves. Example: SPOC.", show_default=False), ], 
    cadence_period: Annotated[float, typer.Argument(    help="Cadence period to filter the light curves. Example: 120.",    show_default=False, ), ], 
    n_batches: int = typer.Option(None, "--n-batches", help="Number of batches to download. Defaults to all.", show_default=False, ), 
    n_threads: int = typer.Option(0, "--n-threads", help="Number of threads."), 
):
    """Download light curves for stars."""
    import exo_search.pipeline.download_lc as download_lc

    download_lc.main(stars_path, mission, author, cadence_period, n_batches, n_threads)


@app.command()
def create_lc(
    stars_path: Annotated[Path, typer.Argument(help="Directory with stars.", show_default=False)], 
    dst_path: Annotated[Path, typer.Argument(help="Directory for light curves.", show_default=False)], 
    mission: Annotated[str, typer.Argument(    help="Mission to filter the light curves. Example: TESS.",    show_default=False, ), ], 
    author: Annotated[str, typer.Argument(    help="Author to filter the light curves. Example: SPOC.", show_default=False), ], 
    cadence_period: Annotated[float, typer.Argument(    help="Cadence period to filter the light curves. Example: 120.",    show_default=False, ), ], 
    n_batches: int = typer.Option(None, "--n-batches", help="Number of batches to use. Defaults to all.", show_default=False, ), 
    separate: bool = typer.Option(True, "--separate/--no-separate", help="Keep light curves separated into individual batches.", is_flag=True, show_default=True, ), 
    ignore_csv: List[Path] = typer.Option(None, "--ignore-csv", help="File paths to star lists. These stars will be ignored.", show_default=False, ), 
):
    """Create light curves from stars."""
    import exo_search.pipeline.create_light_curves as create_light_curves

    create_light_curves.main(stars_path, dst_path, mission, author, cadence_period, n_batches, separate, ignore_csv)


@app.command()
def downsample_lc(
    lc_path: Annotated[Path, typer.Argument(help="Directory with light curves.", show_default=False)], 
    dst_path: Annotated[Path, typer.Argument(help="Directory for new light curves.", show_default=False)], 
    method: Annotated[DownsamplingMethod, typer.Argument(help="Downsampling method.", show_default=False), ], 
    step: int = typer.Option(None, "--step", help="Step for downsampling method. Required by moving-avg, every-nth.", show_default=False, ), 
    window: float = typer.Option(None, "--window", help="Window for downsampling method. Required by moving-avg.", show_default=False, ), 
    gap_threshold: float = typer.Option(None, "--gap-threshold", help="Threshold for identifying gaps in time data. Required by moving-avg. Example: 0.0014 for TESS.", show_default=False, ), 
):
    """Downsample light curves."""
    import exo_search.pipeline.downsample_lc as downsample_lc
    from exo_search.utils.downsampling import moving_avg, every_nth

    params = {"step": step}
    selected_method = every_nth
    if method == DownsamplingMethod.moving_avg: 
        params["window"] = window
        params["gap_threshold"] = gap_threshold
        selected_method = moving_avg

    downsample_lc.main(lc_path, dst_path, selected_method, params)


@app.command()
def create_models(
    lc_path: Annotated[Path, typer.Argument(help="Directory with light curves.", show_default=False)], 
    model_path: Annotated[Path, typer.Argument(help="Directory for new models.", show_default=False)], 
    kernel: Annotated[Kernel, typer.Argument(    help="Kernel. For custom kernels, create a script.", show_default=False), ], 
    n: int = typer.Option(None, "--n", help="Create model from n-th light curve from the directory.", show_default=False, ), 
):
    """Create models from light curves using pre-defined kernels."""
    import exo_search.pipeline.create_models as create_models

    create_models.main(lc_path, model_path, kernel.value, n)


@app.command()
def create_detrending_models(
    lc_path: Annotated[Path, typer.Argument(help="Directory with light curves.", show_default=False)], 
    model_path: Annotated[Path, typer.Argument(help="Directory for new models.", show_default=False)], 
    fast: Tuple[float, float, float] = typer.Option((1 / 24, 5 / 24, 2 / 24), help="Length scales for the fast kernel. Min, max, default.", show_default="(1h, 5h, 2h)", ), 
    slow: Tuple[float, float, float] = typer.Option((0.5, 1.5, 1), help="Length scales for the slow kernel. Min, max, default.", show_default="(0.5d, 1.5d, 1d)", ), 
    kernel_name: str = typer.Option(None, "--kernel-name", help="Kernel name. Defaults to se_fast_slow_<default_fast>_<min_fast>_<max_fast>-<default_slow>_<min_slow>_<max_slow>", show_default=False, ), 
    n: int = typer.Option(None, "--n", help="Create model from n-th light curve from the directory.", show_default=False, ), 
):
    """Create models for detrending with composite kernel."""
    import exo_search.pipeline.create_detrending_models as create_detrending_models

    create_detrending_models.main(lc_path, model_path, fast, slow, kernel_name, n)


@app.command()
def train(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    n: int = typer.Option(None, "--n", help="Train n-th model from the directory.", show_default=False), 
    save_model_path: Path = typer.Option(None, "--save-model-path", help="Directory for trained models.", show_default=False, ), 
):
    """Train models."""
    import exo_search.pipeline.train as train

    train.main(model_path, n, save_model_path)


@app.command()
def extract_kernel(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    dst_path: Annotated[Path, typer.Argument(help="Directory for new models.", show_default=False)], 
    index: int = typer.Option(1, "--index", help="Index of the kernel. Starts from 0."), 
    n: int = typer.Option(None, "--n", help="Extract kernel from n-th model from the directory.", show_default=False, ), 
):
    """Extract kernel from a composite kernel and create new models with it."""
    import exo_search.pipeline.extract_kernel as extract_kernel

    extract_kernel.main(model_path, dst_path, index, n)


@app.command()
def predict(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory for new predictions.", show_default="<model_path>/predictions/", ), 
    lc_path: Path = typer.Option(None, "--lc-path", help="Directory with light curves. If not provided, light curves from with models will be used.", show_default=False, ), 
    n: int = typer.Option(None, "--n", help="Make prediction for n-th model from the directory.", show_default=False, ), 
    trained_only: bool = typer.Option(True, "--trained-only/--no-trained-only", help="Skip model if it is not trained.", ), 
    gap_s: float = typer.Option(None, "--gap-s", help="[seconds] If provided, equally spaced time values (with this gap) will be used instead of the light curve's time values.", show_default=False, ), 
    predict_y: bool = typer.Option(False, "--predict-y/--no-predict-y", help="Predict y-values. By default, only f-values are predicted.", ), 
):
    """Make predictions with models."""
    import exo_search.pipeline.predict as predict

    predict.main(model_path, prediction_path, lc_path, n, trained_only, gap_s, predict_y
    )


@app.command()
def filter_models(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    attr: Annotated[str, typer.Argument(    help="Attribute of the models. For predictions count, use 'predictions_count'.",    show_default=False, ), ], 
    comparator: Annotated[Comparator, typer.Argument(help="Comparator.", show_default=False)], 
    value: Annotated[float, typer.Argument(help="Value.", show_default=False)], 
    new_model_path: Annotated[Path, typer.Argument(    help="Directory for models satisfying the condition.", show_default=False), ], 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory with predictions.", show_default="<model_path>/predictions/", ), 
    new_prediction_path: Path = typer.Option(None, "--new-prediction-path", help="Directory for predictions for filtered models.", show_default="<new_model_path>/predictions/", ), 
    n: int = typer.Option(None, "--n", help="Filter n-th model from the directory", show_default=False), 
):
    """Filter models based on condition."""
    import exo_search.pipeline.filter_models as filter_models

    filter_models.main(model_path, attr, comparator.value, value, new_model_path, prediction_path, new_prediction_path, n)


@app.command()
def detrend(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    detrended_lc_path: Annotated[Path, typer.Argument(    help="Directory for new detrended light curves.", show_default=False), ], 
    diff_threshold: Annotated[float, typer.Argument(    help="Threshold for difference between original and detrended light curves. If the difference is higher than provided threshold, the detrended light curve is stored. Otherwise the original light curve is stored.",    show_default=False, ), ], 
    lc_path: Path = typer.Option(None, "--lc-path", help="Prediction is subtracted from the light curve's flux. By default, the model's light curve is used. If a specific light curve was provided for predicting, it needs to be provided here as well.", show_default=False, ), 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory with predictions.", show_default="<model_path>/predictions/", ), 
    results_path: Path = typer.Option(None, "--results-path", help="Directory for storing if the light curve was detrended.", show_default=False, ), 
    n: int = typer.Option(None, "--n", help="Detrend with n-th model from the directory.", show_default=False, ), 
):
    """Detrend light curves with models' predictions."""
    import exo_search.pipeline.detrend as detrend

    detrend.main(model_path, detrended_lc_path, diff_threshold, lc_path=lc_path, prediction_path=prediction_path, results_path=results_path, n=n)


@app.command()
def fold(
    lc_path: Annotated[Path, typer.Argument(help="Directory with light curves.", show_default=False)], 
    folded_lc_path: Annotated[Path, typer.Argument(help="Directory for folded light curves.", show_default=False), ], 
    plot_path: Path = typer.Option(None, "--plot-path", help="Directory for plots.", show_default=False), 
    disposition: str = typer.Option(None, "--disposition", help="Exoplanet disposition. Exoplanets with other dispositions will be skipped.", show_default=False, ), 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots.", show_default=False), 
    n: int = typer.Option(None, "--n", help="Fold n-th light curve from directory.", show_default=False), 
):
    """Fold light curves with exoplanets."""
    import exo_search.pipeline.fold as fold

    fold.main(lc_path, folded_lc_path, plot_path, disposition, file_format, n)


@app.command()
def fft(
    lc_path: Annotated[Path, typer.Argument(help="Directory with light curves.", show_default=False)], 
    plot_path: Path = typer.Option(None, "--plot-path", help="Directory for plots.", show_default=False), 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots.", show_default=False), 
    report_filepath: Path = typer.Option(None, "--report-filepath", help="File name for plot showing true and detected periods.", show_default=False, ), 
    results_filepath: Path = typer.Option(None, "--results-filepath", help="CSV file name for found periods.", show_default=False, ), 
):
    """Produce FFT plots."""
    import exo_search.pipeline.fft as fft

    fft.main(lc_path, plot_path, file_format, report_filepath, results_filepath)


@app.command()
def correlate(
    lc_path: Annotated[Path, typer.Argument(help="Directory with light curves.", show_default=False)], 
    transit_model_path: Annotated[Path, typer.Argument(help="Directory with transit models.", show_default=False)], 
    bias: float = typer.Option(4, "--bias", help="Bias for the threshold."), 
    gap: int = typer.Option(None, "--gap", help="Gap for the threshold.", show_default="1.5 times the width of the transit", ), 
    ref: int = typer.Option(None, "--ref", help="Ref for the threshold.", show_default="6 times the width of the transit", ), 
    plot_path: Path = typer.Option(None, "--plot-path", help="Directory for plots.", show_default=False), 
    results_path: Path = typer.Option(None, "--results-path", help="Directory for results.", show_default=False), 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory with predictions.", show_default="<transit_model_path>/predictions/", ), 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots."), 
    compare_existing: bool = typer.Option(False, "--compare-existing", help="Compare found transits with the known transits. Measure how many transits was detectable and correctly detected.", ), 
    n: int = typer.Option(None, "--n", help="Calculate correlation for n-th light curve from the directory.", show_default=False, ), 
    n_threads: int = typer.Option(0, "--n-threads", help="Number of threads."), 
):
    """Correlate light curves with trasit models' predictions."""
    import exo_search.pipeline.correlate as correlate

    correlate.main(lc_path, transit_model_path, bias, gap, ref, plot_path, results_path, prediction_path, file_format, compare_existing, n, n_threads)



@app.command()
def combine_correlation_results(
    results_paths: Annotated[List[Path], typer.Argument(help="Directories with correlation results.")], 
    dst_path: Annotated[Path, typer.Argument(help="Directory for combined correlation results.")], 
):
    """Combine results from correlation."""
    import exo_search.pipeline.combine_correlation_results as combine_correlation_results

    combine_correlation_results.main(results_paths, dst_path)


@app.command()
def evaluate_correlation_results(
    results_path: Annotated[Path, typer.Argument(help="Directory with combined correlation results.")], 
    dst_path: Annotated[Path, typer.Argument(help="Directory for evaluated results.")], 
    plot_path: Annotated[Path, typer.Argument(help="Directory for plots.")], 
    bad_models_filepath: Path = typer.Option(None, "--bad-models-filepath", help="Filepath to csv file with bad transit models. These models will be ignored.", show_default=False), 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots."),
):
    """Evaluate combined results from correlation."""
    import scripts.evaluation.evaluate_correlation_results as evaluate_correlation_results

    evaluate_correlation_results.main(results_path, dst_path, plot_path, bad_models_filepath, file_format)


@app.command()
def evaluate_transit_models(
    detectable_filepath: Annotated[Path, typer.Argument(help="Filename with detectable transits (CSV).")],
    detected_filepath: Annotated[Path, typer.Argument(help="Filename with detected transits (CSV).")],
    ok_detected_filepath: Annotated[Path, typer.Argument(help="Filename with correctly detected transits (CSV).")],
    results_filepath: Annotated[Path, typer.Argument(help="Filename for evaluation (CSV).")]
):
    """Evaluate transit models."""
    import scripts.evaluation.evaluate_transit_models as evaluate_transit_models

    evaluate_transit_models.main(detectable_filepath, detected_filepath, ok_detected_filepath, results_filepath)



@app.command()
def plot_lc(
    lc_path: Annotated[Path, typer.Argument(help="Dictionary with light curves.", show_default=False)], 
    plot_path: Annotated[Path, typer.Argument(help="Directory for plots.", show_default=False)], 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots."), 
    n: int = typer.Option(None, "--n", help="Plot n-th light curve from the directory.", show_default=False, ), 
    plot_exoplanets: bool = typer.Option(True, "--plot-exoplanets/--no-plot-exoplanets", help="Mark known transit positions.", ), 
):
    """Plot light curves."""
    import exo_search.pipeline.plot_lc as plot_lc

    plot_lc.main(lc_path, plot_path, file_format, n, plot_exoplanets)


@app.command()
def plot_models(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    plot_path: Annotated[Path, typer.Argument(help="Directory for plots.", show_default=False)], 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory with predictions.", show_default="<model_path>/predictions/", ), 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots."), 
    n: int = typer.Option(None, "--n", help="Plot n-th model from the directory.", show_default=False), 
    is_transit: bool = typer.Option(False, "--is-transit/--no-is-transit", help="X-axis shown in hours from transit midpoint.", ), 
):
    """Plot models' predictions."""
    import exo_search.pipeline.plot_models as plot_models

    plot_models.main(model_path, plot_path, prediction_path, file_format, n, is_transit)


@app.command()
def plot_detrending(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    plot_path: Path = typer.Option(None, "--plot-path", help="Directory for plots.", show_default=False), 
    plot_attr_path: Path = typer.Option(None, "--plot-attr-path", help="Directory for attributes plots.", show_default=False, ), 
    lc_path: Path = typer.Option(None, "--lc-path", help="Directory with light curves. If the prediction was made for a different light curve than the model's, it has to be provided.", show_default=False, ), 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory with predictions.", show_default="<model_path>/predictions/", ), 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots."), 
    label_path: Path = typer.Option(None, "--label-path", help="Directory with star labels."), 
    n: int = typer.Option(None, "--n", help="Plot detrending for n-th model from the directory."), 
):
    """Plot detrending results."""
    import exo_search.pipeline.plot_detrending as plot_detrending

    plot_detrending.main(model_path, plot_path, plot_attr_path, lc_path, prediction_path, file_format, label_path, n)


@app.command()
def plot_model_attributes(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    plot_attr_path: Annotated[Path, typer.Argument(help="Directory for plots.", show_default=False)], 
    file_format: str = typer.Option("png", "--file-format", help="File format for plots.", show_default=False), 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory with predictions.", show_default="<model_path>/predictions/", ), 
    label_path: Path = typer.Option(None, "--label-path", help="Directory with star labels."), 
):
    """Plot models' attributes against each other."""
    import exo_search.pipeline.plot_model_attributes as plot_model_attributes

    plot_model_attributes.main(model_path, plot_attr_path, file_format, prediction_path, label_path
    )


@app.command()
def stats(
    model_path: Annotated[Path, typer.Argument(help="Directory with models.", show_default=False)], 
    prediction_path: Path = typer.Option(None, "--prediction-path", help="Directory with predictions.", show_default="<model_path>/predictions/", ), 
):
    """See basic info about models."""
    import exo_search.pipeline.counts as counts

    counts.main(model_path, prediction_path)

if __name__ == "__main__":
    app()
