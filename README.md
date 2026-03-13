# High-resolution global flood projections constrained by observations reveal increasing extremes

Code for the paper: "High-resolution global flood projections constrained by 
observations reveal increasing extremes" (Nature Communications, under review)

## Overview
This repository contains the core code for the Extreme Flood-Based Correction 
(EFBC) framework, which integrates observation-based Regional Flood Frequency 
Analysis (RFFA) with CMIP6 cascade-type model projections to generate 
observation-constrained design flood estimates at 1-km resolution globally.

## Code Structure
- `efbc_correction.py` — EFBC multiplicative correction framework
- `gev_fitting.py` — GEV distribution fitting using L-moments
- `rf_downscaling.py` — Random Forest downscaling from 0.25° to 1-km
- `perturbation_validation.py` — Artificial perturbation validation experiments

## Dependencies
- Python 3.x
- numpy, scipy, pandas, scikit-learn, rasterio, geopandas

## Data
Input data sources: CMIP6 (ESGF), GRADES, MERIT Hydro, Zhao et al. (2021) 
benchmark. See manuscript Data Availability for links.

## Contact
Code available upon request: jingshan@bnu.edu.cn / zhao.g.eb91@m.isct.ac.jp
