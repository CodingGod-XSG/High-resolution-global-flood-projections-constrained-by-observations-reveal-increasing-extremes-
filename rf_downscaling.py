import pandas as pd
import numpy as np
import rasterio
import joblib
import time
import warnings
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Suppress runtime warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# =================================================================
# 1. GLOBAL CONFIGURATION
# =================================================================
class RFDownscalingConfig:
    """Configuration for Random Forest Downscaling supporting EFBC framework."""
    
    # Path Placeholders (Replace with actual paths for execution)
    INPUT_EXCEL = Path("path/to/input_data/training_samples.xlsx")
    STATIC_TIFF_DIR = Path("path/to/static_factors")
    DYNAMIC_TIFF_BASE = Path("path/to/dynamic_factors")
    MODEL_SAVE_DIR = Path("path/to/saved_models")
    PREDICTION_OUT_DIR = Path("path/to/predictions")

    # Static Catchment Features (e.g., MERIT Hydro, MERIT DEM, MODIS/HWSD)
    STATIC_FEATURES = [
        'CA',  # Catchment area
        'SL',  # Catchment slope
        'RW',  # River width
        'FL',  # Flow distance
        'CN',  # Curve number
        'LK'   # Lake fraction
    ]

    # Dynamic Climate Features (e.g., derived from CMIP6)
    # AP: Annual precipitation | CV: Precip seasonality
    # AT: Annual mean temp | TR: Temp range
    RETURN_PERIODS = ['2a', '5a', '10a', '20a', '50a', '100a', '200a']

    # Scenarios focus: Historical baseline and SSP5-8.5 Far-Future (P3)
    SCENARIOS = {
        'Historical_P1': {
            'period': '1965-2014',
            'dynamic_features': ['his_AP_1965-2014', 'his_CV_1965-2014', 'his_AT_1965-2014', 'his_TR_1965-2014'],
            'target_prefix': 'his_P1',
            'folder': 'historical'
        },
        'SSP585_P3': {
            'period': '2051-2100',
            'dynamic_features': ['585_AP_2051-2100', '585_CV_2051-2100', '585_AT_2051-2100', '585_TR_2051-2100'],
            'target_prefix': '585_P3',
            'folder': 'ssp585'
        }
    }

    # Model Parameters
    RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': None,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': 42
    }
    CV_FOLDS = 10      # Standard 10-fold cross-validation
    EPSILON = 1e-10    # Small constant to handle zero discharge in log-transform
    MAX_WORKERS = 8    # CPU cores for parallel execution
    NODATA_VALUE = -9999

# =================================================================
# 2. HYDROLOGICAL METRICS UTILITIES
# =================================================================
class HydrologyMetrics:
    """Calculates Kling-Gupta Efficiency (KGE) for hydrological validation."""
    @staticmethod
    def calculate_kge(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Components of KGE (Gupta et al., 2009)
        r = np.corrcoef(y_true, y_pred)[0, 1]
        gamma = (np.std(y_pred)/np.mean(y_pred)) / (np.std(y_true)/np.mean(y_true))
        beta = np.mean(y_pred) / np.mean(y_true)
        return 1 - np.sqrt((r-1)**2 + (gamma-1)**2 + (beta-1)**2)

# =================================================================
# 3. TRAINING COMPONENT
# =================================================================
class RFDownscalingTrainer:
    """Handles parallelized 10-fold CV and final model training."""
    def __init__(self, config):
        self.cfg = config

    def train_task(self, df, scenario, target_col):
        """Processes training and 10-fold CV for a specific target period."""
        features = self.cfg.STATIC_FEATURES + self.cfg.SCENARIOS[scenario]['dynamic_features']
        X = df[features].values
        y = df[target_col].values

        # Filter positive discharge values for physical consistency
        mask = (y > 0) & (~np.isnan(y)) & (~np.isnan(X).any(axis=1))
        X_clean, y_clean = X[mask], y[mask]
        y_log = np.log10(y_clean + self.cfg.EPSILON)

        # 10-Fold Cross-Validation for KGE reporting
        kf = KFold(n_splits=self.cfg.CV_FOLDS, shuffle=True, random_state=42)
        kge_scores = []
        
        for train_idx, val_idx in kf.split(X_clean):
            # Scale separately for each fold to prevent data leakage
            sc_X, sc_y = StandardScaler(), StandardScaler()
            X_train_cv = sc_X.fit_transform(X_clean[train_idx])
            y_train_cv = sc_y.fit_transform(y_log[train_idx].reshape(-1, 1)).flatten()
            
            model_cv = RandomForestRegressor(**self.cfg.RF_PARAMS)
            model_cv.fit(X_train_cv, y_train_cv)
            
            # Predict and inverse transform to calculate KGE in original space
            y_val_pred_log = sc_y.inverse_transform(model_cv.predict(sc_X.transform(X_clean[val_idx])).reshape(-1, 1)).flatten()
            kge_scores.append(HydrologyMetrics.calculate_kge(y_clean[val_idx], 10**y_val_pred_log - self.cfg.EPSILON))

        # Final Training on complete dataset
        final_sc_X, final_sc_y = StandardScaler(), StandardScaler()
        X_final = final_sc_X.fit_transform(X_clean)
        y_final = final_sc_y.fit_transform(y_log.reshape(-1, 1)).flatten()
        
        final_model = RandomForestRegressor(**self.cfg.RF_PARAMS)
        final_model.fit(X_final, y_final)

        # Persistence
        rp_name = target_col.split('_')[-1]
        out_path = self.cfg.MODEL_SAVE_DIR / scenario / rp_name
        out_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(final_model, out_path / f"{target_col}_model.joblib")
        joblib.dump(final_sc_X, out_path / f"{target_col}_scaler_X.joblib")
        joblib.dump(final_sc_y, out_path / f"{target_col}_scaler_y.joblib")

        return f"Task {target_col} Done | Mean KGE: {np.mean(kge_scores):.3f}"

# =================================================================
# 4. PREDICTION COMPONENT
# =================================================================
class RFDownscalingPredictor:
    """Predicts spatial flood maps using memory-efficient block processing."""
    def __init__(self, config):
        self.cfg = config

    def _predict_block(self, stack, valid_mask, model, sc_X, sc_y):
        n_feat, height, width = stack.shape
        X_valid = stack.reshape(n_feat, -1).T[valid_mask.flatten()]
        if len(X_valid) == 0: return np.zeros((height, width), dtype=np.float32)

        y_log = sc_y.inverse_transform(model.predict(sc_X.transform(X_valid)).reshape(-1, 1)).flatten()
        res = np.zeros(height * width, dtype=np.float32)
        res[valid_mask.flatten()] = 10**y_log - self.cfg.EPSILON
        return res.reshape(height, width)

    def spatial_task(self, scenario, rp):
        target_col = f"{self.cfg.SCENARIOS[scenario]['target_prefix']}_{rp}"
        model_dir = self.cfg.MODEL_SAVE_DIR / scenario / rp
        
        # Load artifacts
        model = joblib.load(model_dir / f"{target_col}_model.joblib")
        sc_X = joblib.load(model_dir / f"{target_col}_scaler_X.joblib")
        sc_y = joblib.load(model_dir / f"{target_col}_scaler_y.joblib")

        # Paths for feature rasters
        dyn_info = self.cfg.SCENARIOS[scenario]
        static_paths = [self.cfg.STATIC_TIFF_DIR / f"{f}.tif" for f in self.cfg.STATIC_FEATURES]
        dyn_paths = [self.cfg.DYNAMIC_TIFF_BASE / dyn_info['folder'] / f"{f}.tif" for f in dyn_info['dynamic_features']]
        all_paths = static_paths + dyn_paths

        out_file = self.cfg.PREDICTION_OUT_DIR / scenario / f"{target_col}_predicted.tif"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Handle Management: Open source files and ensure closure
        fh_list = [rasterio.open(p) for p in all_paths]
        try:
            meta = fh_list[0].meta.copy()
            meta.update(dtype='float32', compress='lzw', nodata=0)

            with rasterio.open(out_file, 'w', **meta) as dst:
                for _, window in fh_list[0].block_windows(1):
                    # Fixed: Read using file handle list securely
                    data_stack = np.stack([fh.read(1, window=window) for fh in fh_list])
                    valid_mask = np.all(data_stack != self.cfg.NODATA_VALUE, axis=0)
                    
                    if np.any(valid_mask):
                        dst.write(self._predict_block(data_stack, valid_mask, model, sc_X, sc_y), 1, window=window)
                    else:
                        dst.write(np.zeros(valid_mask.shape, dtype=np.float32), 1, window=window)
        finally:
            for fh in fh_list: fh.close()  # Fix: Variable shadowing avoidance

        return f"Finished: {out_file.name}"

# =================================================================
# 5. WORKFLOW ENTRY
# =================================================================
def main():
    cfg = RFDownscalingConfig()
    trainer, predictor = RFDownscalingTrainer(cfg), RFDownscalingPredictor(cfg)

    # PARALLEL TRAINING
    print("--- Phase 1: Parallel Training & 10-Fold KGE Validation ---")
    df = pd.read_excel(cfg.INPUT_EXCEL)
    train_args = [(scn, f"{cfg.SCENARIOS[scn]['target_prefix']}_{rp}") 
                  for scn in cfg.SCENARIOS.keys() for rp in cfg.RETURN_PERIODS if f"{cfg.SCENARIOS[scn]['target_prefix']}_{rp}" in df.columns]

    with ProcessPoolExecutor(max_workers=4) as exec_t:
        futures_t = [exec_t.submit(trainer.train_task, df, s, t) for s, t in train_args]
        for f in as_completed(futures_t): print(f.result())

    # PARALLEL PREDICTION
    print("\n--- Phase 2: Parallel Spatial Prediction (Exporting GeoTIFFs) ---")
    pred_args = [(scn, rp) for scn in cfg.SCENARIOS.keys() for rp in cfg.RETURN_PERIODS]
    
    with ProcessPoolExecutor(max_workers=cfg.MAX_WORKERS) as exec_p:
        futures_p = [exec_p.submit(predictor.spatial_task, s, r) for s, r in pred_args]
        for f in tqdm(as_completed(futures_p), total=len(pred_args), desc="Exporting"): pass

if __name__ == "__main__":
    main()
