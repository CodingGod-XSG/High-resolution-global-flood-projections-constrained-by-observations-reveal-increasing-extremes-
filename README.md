# High-resolution global flood projections constrained by observations reveal increasing extremes

Code for: "High-resolution global flood projections constrained by observations 
reveal increasing extremes" (Nature Communications, under review)

## Overview
This repository contains core code implementing the three-step framework for 
global observation-constrained design flood projections at 1-km resolution.

## Code Structure

### Step 1: Coarse-Resolution Design Flood Estimation
CaMa-Flood routing and GEV fitting are handled by external models (see Data Sources).  
The RF training uses matched coarse-to-fine pixel pairs as training data.

### Step 2: High-Resolution Downscaling (`rf_downscaling.py`)
Random Forest training and spatial prediction to downscale design floods  
from 0.25° (CaMa-Flood output) to 1-km (MERIT Hydro) resolution.  
Features: catchment area, slope, river width, flow distance, curve number,  
lake fraction, annual precipitation, precipitation seasonality, mean temperature,  
temperature range.

### Step 3: Observation-Constrained Correction (`efbc_correction.py`)
Extreme Flood-Based Correction (EFBC) anchors future projections to  
observation-based historical benchmarks:  
`Q_future_corrected = Q_obs_hist × (Q_sim_future / Q_sim_hist)`  
Target scenario: SSP5-8.5, far-future period (2051–2100).

## Dependencies
```
Python 3.x
numpy, pandas, scikit-learn, rasterio, joblib, tqdm
```

## Data Sources
| Dataset | Reference |
|---------|-----------|
| CMIP6 climate forcing | https://esgf-node.llnl.gov |
| GRADES bias-corrected runoff | Lin et al. (2019), WRR |
| MERIT Hydro river network | Yamazaki et al. (2019), WRR |
| Observation-based design floods | Zhao et al. (2021), HESS |

## Contact
For questions regarding the code, please contact:    
202331470002@mail.bnu.edu.cn/xu13667185978@gmail.com/jingshan@bnu.edu.cn / zhao.g.eb91@m.isct.ac.jp
