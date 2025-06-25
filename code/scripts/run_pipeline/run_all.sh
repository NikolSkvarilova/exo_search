#!/bin/bash

cd ../..

# Creating star and exoplanets lists
python run.py star-list tess from-project-candidates ./star_names/nasa_exo_archive/project_candidates_TESS_test_sample.csv tmp/lists/star_list.csv --disposition=CP
python run.py star-list tess exoplanets ./star_names/nasa_exo_archive/project_candidates_TESS_test_sample.csv tmp/lists/exoplanets.csv

# Creating stars
python run.py pipeline create-stars tmp/lists/star_list.csv tmp/stars/ --exoplanets-csv=tmp/lists/exoplanets.csv

# Downloading light curves
python run.py pipeline download-lc tmp/stars TESS SPOC 120 --n-batches=1

# Creating light curve objects
python run.py pipeline create-lc tmp/stars tmp/lcs/original TESS SPOC 120

# Plot light curves
python run.py pipeline plot-lc tmp/lcs/original tmp/figures/original_lcs/

# Downsampling light curves
python run.py pipeline downsample-lc tmp/lcs/original tmp/lcs/downsampled moving-avg --step=10 --window=10 --gap-threshold=0.0014

# Creating detrending models
python run.py pipeline create-detrending-models tmp/lcs/downsampled tmp/models/detrending

# Training detrending models
python run.py pipeline train tmp/models/detrending

# Extracting slow kernel from detrending models
python run.py pipeline extract-kernel tmp/models/detrending tmp/models/detrending_slower

# Make predictions with detrending models over the original light curve's time
python run.py pipeline predict tmp/models/detrending_slower --lc-path=tmp/lcs/original

# Detrending
python run.py pipeline detrend tmp/models/detrending_slower tmp/lcs/detrended 0.2 --lc-path=tmp/lcs/original --results-path=tmp/results/detrending

# Plot detrending
python run.py pipeline plot-detrending tmp/models/detrending_slower --plot-path=tmp/figures/detrending --plot-attr-path=tmp/figures/detrending_attributes/ --lc-path=tmp/lcs/original --label-path=star_labels/labels

# Fold light curves
python run.py pipeline fold tmp/lcs/detrended tmp/lcs/folded --plot-path=tmp/figures/folded --disposition=CP

# Create transit models
python run.py pipeline create-models tmp/lcs/folded tmp/models/folded se

# Train transit models
python run.py pipeline train tmp/models/folded

# Make predictions with transit models
python run.py pipeline predict tmp/models/folded --gap-s=120

# Plot transit models
python run.py pipeline plot-models tmp/models/folded tmp/figures/transit_models --is-transit

# Compute correlations
python run.py pipeline correlate tmp/lcs/detrended tmp/models/folded --plot-path=tmp/figures/correlation/ --results-path=tmp/results/correlation/ --compare-existing

# Combine correlation results
python run.py pipeline combine-correlation-results ./tmp/results/correlation/corr_results/ tmp/results/correlation_combined

# Evaluate correlation results
python run.py pipeline evaluate-correlation-results ./tmp/results/correlation_combined/ tmp/results/correlation_evaluated/ tmp/figures/correlation_evaluation/

# Evaluate transit models
python run.py pipeline evaluate-transit-models ./tmp/results/correlation/detectable.csv ./tmp/results/correlation/detected.csv ./tmp/results/correlation/ok_detected.csv ./tmp/results/transit_models_evaluation/evaluation.csv