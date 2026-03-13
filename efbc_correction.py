import rasterio
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# Suppress runtime warnings for cleaner terminal output
warnings.filterwarnings('ignore')

class EFBCConfig:
    """
    Configuration for Extreme Flood-Based Correction (EFBC).
    All paths are placeholders for peer-review transparency.
    """
    # Root directory for the analysis
    BASE_DIR = Path("path/to/project_root")
    
    # Input data directories
    HIST_SIM_DIR = BASE_DIR / "1_Historical_Simulation"
    FUTURE_SIM_DIR = BASE_DIR / "2_Future_Simulation"
    HIST_OBS_DIR = BASE_DIR / "3_Historical_Observation"
    
    # Output directory for the EFBC-corrected products
    OUTPUT_DIR = BASE_DIR / "4_EFBC_Corrected_Results"

    # Hydrological Return Periods of interest
    RETURN_PERIODS = ['2a', '5a', '10a', '20a', '50a', '100a', '200a']
    
    # Focus exclusively on the Far-future High-Emission Scenario (SSP5-8.5 P3)
    SCENARIOS = {
        'SSP585_P3': '585_P3'   # Far-future period (e.g., 2051-2100)
    }

    # Computational settings
    MAX_WORKERS = 4
    CHUNK_SIZE = 1024  # Block size for memory-efficient raster processing
    NODATA_VAL = 0     # Target NoData value for flood discharge maps

class EFBCBiasCorrector:
    """
    Extreme Flood-Based Correction (EFBC) System.
    
    Algorithm Logic:
    The EFBC approach refines future flood projections by applying a simulated 
    change signal (delta ratio) to high-resolution historical observations.
    Formula: V_future_corr = V_obs_hist * (V_sim_future / V_sim_hist)
    """

    def __init__(self, config):
        self.cfg = config

    def _apply_efbc_logic(self, hist_sim, future_sim, hist_obs):
        """
        Implementation of the EFBC scaling with mask protection.
        Ensures results are physically plausible (>0) and handles division by zero.
        """
        # Validity Mask: Process only pixels with positive discharge values
        valid_mask = (hist_sim > 0) & (future_sim > 0) & (hist_obs > 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the scaling factor (Delta Ratio)
            # Default to 1.0 where hist_sim is non-positive to maintain stability
            ratio = np.divide(future_sim, hist_sim, 
                              out=np.ones_like(future_sim), 
                              where=hist_sim > 0)
            
            # Map the simulated signal onto historical observations
            corrected = hist_obs * ratio

        # Clean output: Enforce NoData/Zero for invalid grid cells
        result = np.where(valid_mask, corrected, self.cfg.NODATA_VAL).astype(np.float32)
        return result

    def _process_task(self, scenario, rp):
        """Processes a single raster file for a given scenario and return period."""
        try:
            # File path construction
            hist_sim_path = self.cfg.HIST_SIM_DIR / f"his_P1_{rp}_predicted.tif"
            future_sim_path = self.cfg.FUTURE_SIM_DIR / scenario / f"{self.cfg.SCENARIOS[scenario]}_{rp}_predicted.tif"
            hist_obs_path = self.cfg.HIST_OBS_DIR / f"{rp}.tif"
            
            out_dir = self.cfg.OUTPUT_DIR / scenario
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{self.cfg.SCENARIOS[scenario]}_{rp}_EFBC_corrected.tif"

            # Execute Block-wise processing via Rasterio
            with rasterio.open(hist_obs_path) as src_obs:
                meta = src_obs.meta.copy()
                meta.update({'dtype': 'float32', 'compress': 'lzw', 'nodata': self.cfg.NODATA_VAL})

                with rasterio.open(output_path, 'w', **meta) as dst:
                    with rasterio.open(hist_sim_path) as src_h_sim, \
                         rasterio.open(future_sim_path) as src_f_sim:
                        
                        # Process by block to ensure low memory footprint
                        for _, window in src_obs.block_windows(1):
                            h_sim_chunk = src_h_sim.read(1, window=window).astype(np.float32)
                            f_sim_chunk = src_f_sim.read(1, window=window).astype(np.float32)
                            h_obs_chunk = src_obs.read(1, window=window).astype(np.float32)

                            # Apply core EFBC logic
                            corrected_chunk = self._apply_efbc_logic(h_sim_chunk, f_sim_chunk, h_obs_chunk)
                            dst.write(corrected_chunk, 1, window=window)

            return f"{scenario}_{rp}", True, "Processed Successfully"
        except Exception as e:
            return f"{scenario}_{rp}", False, str(e)

    def run(self):
        """Orchestrate parallel EFBC correction."""
        start_time = time.time()
        print(f"--- Starting EFBC (Extreme Flood-Based Correction) ---")
        print(f"Target Scenario: SSP5-8.5 P3 (Far-Future)")
        print(f"Using {self.cfg.MAX_WORKERS} parallel workers.")

        tasks = [(s, rp) for s in self.cfg.SCENARIOS for rp in self.cfg.RETURN_PERIODS]
        success_count = 0

        with ProcessPoolExecutor(max_workers=self.cfg.MAX_WORKERS) as executor:
            future_to_task = {executor.submit(self._process_task, s, rp): (s, rp) for s, rp in tasks}
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="EFBC Refinement"):
                task_id, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    print(f"\n[Error] {task_id}: {msg}")

        duration = (time.time() - start_time) / 60
        print(f"\nEFBC Framework Completed. {success_count}/{len(tasks)} tasks successful.")
        print(f"Total Execution Time: {duration:.2f} minutes.")

if __name__ == "__main__":
    # Initialize and Execute the EFBC framework
    config = EFBCConfig()
    corrector = EFBCBiasCorrector(config)
    corrector.run()
