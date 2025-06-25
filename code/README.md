# Usage
The implementation is written in `Python 3.10`.

* `cmd_pipeline/` - command line parser.
* `exo_search/` - the main package, containing data classes, modeling classes, and implementations for commands.
* `scripts/` - various utility scripts for generating plots, star lists,  evaluation of the correlation results, and example of the commands usage.
* `star_labels/` - labels for the categorization of stars by stellar variability.
* `star_names/` - downloaded TESS project candidates table.
* `create_custom_models.py` -- to create models with complex kernels.
* `requirements.txt` - contains package requirements for `pip`. 
* `run.py` -- to run the commands.


Virtual environment for Python can be created using `python3.10 -m venv .venv`. With **activated** (for example `.\.venv\Scripts\activate` on Windows) virtual environment, packages in `requirements.txt` can be installed with `pip install -r requirements.txt` (may take a bit longer).

## Pipeline
Besides writing custom scripts, a command line interface allows to download and process the data from start to finish. This section goes through the individual commands and hints at the expected order.

Each command supports a `--help` option with explanations of the mandatory and optional arguments. Please consult this option before running the actual commands.

The commands are separated into two categories:
```
> python run.py

 Usage: run.py [OPTIONS] COMMAND [ARGS]...

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ star-list   Creating star lists.                                                                               │
│ pipeline    Pipeline commands.                                                                                 │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

For most of the work, the `pipeline` option is used. Each category offers more commands:

```
> python run.py star-list

 Usage: run.py star-list [OPTIONS] COMMAND [ARGS]...

 Creating star lists.

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ tess   Commands for working with TESS data.                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

and 

```
> python run.py pipeline

 Usage: run.py pipeline [OPTIONS] COMMAND [ARGS]...

 Pipeline commands.

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ create-stars                   Create stars from star list.                                                 │
│ download-lc                    Download light curves for stars.                                             │
│ create-lc                      Create light curves from stars.                                              │
│ downsample-lc                  Downsample light curves.                                                     │
│ create-models                  Create models from light curves using pre-defined kernels.                   │
│ create-detrending-models       Create models for detrending with composite kernel.                          │
│ train                          Train models.                                                                │
│ extract-kernel                 Extract kernel from a composite kernel and create new models with it.        │
│ predict                        Make predictions with models.                                                │
│ filter-models                  Filter models based on condition.                                            │
│ detrend                        Detrend light curves with models' predictions.                               │
│ fold                           Fold light curves with exoplanets.                                           │
│ fft                            Produce FFT plots.                                                           │
│ correlate                      Correlate light curves with trasit models' predictions.                      │
│ combine-correlation-results    Combine results from correlation.                                            │
│ evaluate-correlation-results   Evaluate combined results from correlation.                                  │
│ evaluate-transit-models        Evaluate transit models.                                                     │
│ plot-lc                        Plot light curves.                                                           │
│ plot-models                    Plot models' predictions.                                                    │
│ plot-detrending                Plot detrending results.                                                     │
│ plot-model-attributes          Plot models' attributes against each other.                                  │
│ stats                          See basic info about models.                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

Many of the commands support `--n` option, which in most cases is used when running the same command multiple times concurrently, for each file in the directory (training x amount of models at the same time, or detrending y light curves at the same time). For this purpose, the `--n` option allows to specify the file to be used from the directory. It is important to note that the option refers to the n-th file in the directory (filenames are sorted alphabetically and the n-th is processed). Therefore, running multiple commands, one creating files and the other reading them, simultaneously, will result in unfinished work and is not advised.

### Star names
A star is uniquely identified by its name. I mostly worked with TESS data, where I used the TIC ids. As stars can have many names, it is important to check the logs during download (to see whether the names are being found), as a star may be available for download only under a couple of names.

I used the TESS Project Candidates table. After downloading the original table, you can parse it with 

```
python run.py star-list tess from-project-candidates
```

which produces a file usable in further commands. A version of this table is in `code/star_names/nasa_exo_archive/`.

**In case you are using custom star list**, the `CSV` file should have a column titled `star_name`. Example:

```csv
star_name
TIC 317060587
TIC 366989877
TIC 320004517
TIC 299799658
```

Information about exoplanets is also extracted from the TESS Project Candidates into a CSV file, which is then used when initializing `Star` objects. Light curves produced from those `Star` objects also carry the information about exoplanets.

```
python run.py star-list tess exoplanets
```

**In case you are using custom exoplanet list**, the `CSV` should look like this:

```
name,star_name,disposition,period,transit_midpoint_BJD,transit_duration_h
TOI 1000.01,TIC 50365310,FP,2.1713484,2459229.630046,2.0172196
TOI 1001.01,TIC 88863718,PC,1.9316462,2459987.948873,3.166
TOI 1002.01,TIC 124709665,FP,1.8675574,2459224.687802,1.408
TOI 1003.01,TIC 106997505,FP,2.74323,2458493.3957,3.167
```

### Initializing star objects

```bash
python run.py pipeline create-stars
```

Original light curves are held by `Star` objects. From these, `LightCurve` objects are created, which can be then downsampled and used for training, predicting, etc.

The `Star` objects are necessary to download new light curves to avoid downloading already downloaded light curves. Besides that, only `LightCurve` objects are used throughout the project.

A filename with star names is passed to a command, which initializes the `Star` objects and produces `JSON` files for each one.

### Downloading light curves
Light curves are downloaded for each `Star` object, based on the provided mission, author, and cadence period. A number of batches to be downloaded can be specified (sectors for TESS) as well. Downloading in multiple threads is supported.

```bash
python run.py pipeline download-lc
```

### Creating light curves
`LightCurve` objects are saved as `JSON` files, storing the time and flux values, along with information about exoplanets (copied from the parent `Star` object). These objects are then used further down in the pipeline in various places. 

```bash
python run.py pipeline create-lc
```

### Downsampling light curves
`LightCurve` object's time and flux values can be downsampled, producing new `LightCurve` objects.

```bash
python run.py pipeline downsample-lc
```

### Creating models
Models are stored as `JSON` (holds the light curve and other information) and `pickle` (the actual model) files. A set of basic kernels is supported (SE, periodic, ...). To create models with custom kernel, use `code/create_custom_models.py` and define your kernel there. This scripts can also be used to create the detrending models, which uses a composite kernel.

```bash
python run.py pipeline create-models
```

> *NOTE:* when transferring models, the `pickle` files need to be moved as well. These are typically stored in the model directory in `gpflow_models/`.

### Training models
Already trained models are skipped.

```bash
python run.py pipeline train
```

### Detrending
For detrending, a composite kernel of two SE kernels was used. These models can be created using


```bash
python run.py pipeline create-detrending-models
```

> Each SE kernel has a different length scale (unit is days, as for the time values of the light curves). Smaller length scale means that the model is able to capture faster changes in the data, while larger value results in smoother model. Combining these two approaches, the fast changes such as transits are captured by the fast kernel, while the stellar variability can be captured by the slow kernel, which is the used further for detrending. To make the kernels more flexible, the length scale for each one is set as an allowed range with default value. For the fast kernel, the default value is 2 hours, and the range is from 1 to 5 hours. For the slow kernel, the default value is 1 day, and the range is from 0.5 to 1.5 days. 


After the model is trained, the slower kernel is extracted using

```bash
python run.py pipeline extract-kernel
```

which creates new models with the slow kernel. This new model is then used for the predictions (`predict` command).

The detrending process consists of subtracting a prediction from its light curve. Generally, the steps are as follows:

1) create downsampled light curves
2) create and train models
3) make predictions over the **original** light curve's time values
4) perform detrending with the **original** light curve and with the prediction

This way, the result is a light curve with the original resolution, but detrended.

```bash
python run.py pipeline detrend
```

### Making predictions
A prediction can be created using a trained model. The prediction can either be made over 

1) the original light curve's time values, or
2) another light curves's time values (possibly the original light curve, useful when the model was trained on a downsampled version).

At the same time, light curves have gaps due to bad quality data points. If it is necessary for the prediction to be made evenly spaced, the gap between the data points can be specified by an option. The starting and stopping time come from the used light curve. This is useful when producing models of transits, where for the correlation, it is necessary for the data points to be evenly spaced.

```bash
python run.py pipeline predict
```

`Prediction` objects are stored separately from models, in `predictions` directory in the models' directory. The model loads its predictions, is is therefore necessary to have the model as well to be able to work with the predictions.

### Filtering models
Models can be filtered by various conditions. The most typical use case is filtering trained models, and putting them into a separate directory. Other attributes of the models can be used for filtering as well.

```bash
python run.py pipeline filter-models
```

Example usage to filter trained models:

```bash
python run.py pipeline filter-models models_dir/ trained = 1 trained_models_dir/
```

Example usage to filter models with likelihood larger than -4000:

```bash
python run.py pipeline filter-models models_dir/ likelihood ">" -- -4000 filtered_models_dir/
```

> Notice that when using >, <, it is important to put it in quotes so that it is not mistakes for output. At the same time, -4000 can be interpreted as option. To make sure it is used as a negative number, put 2 extra dashes (--) in front of it.

### Folding light curves
Light curves with exoplanets can be folded over on the transits, producing new light curves.

```bash
python run.py pipeline fold
```

### Producing plots
Some commands produce plots as a byproduct, some are specifically meant for plotting. This includes commands

```bash
python run.py pipeline plot-lc
python run.py pipeline plot-models
python run.py pipeline plot-detrending
python run.py pipeline plot-model-attributes
```

The default file format is png, but it can be changed with an option.

### Computing correlation
Correlation (multiplication of the two signals) is computed between a detrended light curve and a model of a transit.

```bash
python run.py pipeline correlate
```

During the same step, a threshold is computed using a moving average with gap around the cell for which the threshold is computed. This has 3 parameters:

* number of gap values,
* number of reference values (from which the average is computed), and
* bias, which multiplies the average.

All have preset values based on the width of the transit model, bias is 4 by default.

This threshold is then compared to the correlation values, and in places where the correlation exceeds the threshold, the average time over this event (the correlation can be higher than the threshold in multiple consecutive time values), and the maximal correlation value divided by the average threshold, are then stored. 

The correlation is calculated for all transit models available in the directory, producing a single `JSON` file with information about the light curve and the time and correlation values for each transit model. If the correlation is lower than the threshold throughout the whole light curve, only a single time and correlation value is stored, and that is for the highest correlation / threshold value.


### Basic info about a batch of models
To see the number of loadable models, of which how many are trained and with at least 1 prediction, use

```bash
python run.py pipeline stats
```

## Evaluation of results
Once all the correlations are calculated, the results need to be combined

```bash
python run.py pipeline combine-correlation-results
```

> Accepts multiple directories with correlation results, and produces a `CSV` file for each transit model, with each row corresponding to a single **star** (not a light curve -- correlation results from multiple light curves for a single star are combined). This script also  computes mean period between the transits (if three transits are detected, the distance between the first and third is divided by 2, distances between adjacent detections are not scaled).

and evaluated

```bash
python run.py pipeline evaluate-correlation-results
```

> Accepts the produced `CSV` files and evaluates the results. This method produces `CSV` files containing sorted stars by the highest transit count detected, and another one with the longest mean period.

To filter out bad transit models at this stage, you can provide a `CSV` file with names of bad transit models. These models will be ignored during the evaluation.

```csv
bad_models
TIC 79748331_TOI 1064.01
```

Source code for the evaluation command is in `code/scripts/evaluation/evaluate_correlation_results.py` and is meant to be edited based on the evaluation goals.


**Transit models can be evaluated** on light curves with known exoplanets, using

```bash
python run.py pipeline evaluate-transit-models
```

For this, the correlation command needs to be run with the `--compare-existing` option, so that the found transits are evaluated against the known transits and results from this comparison are also produced, which are needed in the script.

## Typical workflow
0) Create a star list
1) Initialize stars
2) Download light curves
3) Create light curve objects
4) Downsample light curves
5) Create detrending models (with composite kernel)
6) Train detrending models
7) Extract the slow kernel
8) Make predictions
9) Detrend light curves 
10) Run correlation with transit models
11) Combine correlation results
12) Evaluate correlation results

Transit models are created in a similar fashion, but the light curves need to be folded before the model is created and trained.

## Example
`scripts/run_pipeline/run_all.sh` is an example bash script. It runs most of the steps, from start to finish, and produces various models, plots, and csv files with results. 

## Logs
Some information is outputted directly to stdout, some is outputted to logs. Logs are files named with the timestamp the script started and are saved into `logs/` directory. 

It is important to check the logs especially during the download of light curves, as errors with name resolution, or too many concurrent requests are outputted there.