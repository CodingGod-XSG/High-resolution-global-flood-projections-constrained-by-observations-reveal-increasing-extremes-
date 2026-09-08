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

### Figures (`plotting/`)
Scripts that reproduce the three main figures from the projection outputs.

| Script | Figure | Content |
|--------|--------|---------|
| `plotting/Figure1.py` | Fig. 1 | Global flood-magnitude bias ratio (uncorrected cascade vs GRADES), with box plots by climate zone and by continent |
| `plotting/Figure2.py` | Fig. 2 | Multi-model uncertainty (coefficient of variation) maps, with climate-zone, continental, watershed-scale and per-river breakdowns |
| `plotting/Figure3.py` | Fig. 3 | Flood return-period change: corrected map, difference map, and signed-difference box plots / under- vs over-estimation stacked bars by climate zone and continent |

Maps use a 25× block resample for display; statistics use every native 1-km pixel.

The input rasters and per-region statistics tables live in the plotting-data archive
(`efbc_PlotData/`, laid out as `Figure_1/`, `Figure_2/`, `Figure_3/`; see Data Sources).
Each script takes its paths from a single `DATA_DIR` at the top of the file — set the
`EFBC_PLOTDATA_DIR` environment variable to the unpacked archive (or edit `DATA_DIR`),
then run e.g. `python plotting/Figure2.py`. Outputs go to `<Figure_x>/output/`.
Continent masks come from Natural Earth via `geopandas`; on `geopandas` ≥ 1.0 set
`NATURALEARTH_SHP` to a local `ne_110m_admin_0_countries.shp`.

## Dependencies
```
Python 3.x
numpy, pandas, scikit-learn, rasterio, joblib, tqdm, matplotlib, geopandas
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
