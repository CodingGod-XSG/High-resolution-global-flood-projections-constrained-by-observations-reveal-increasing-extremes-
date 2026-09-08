import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import gc
import warnings

warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION
# ================================
CONFIG = {
    'paths': {
        'simulated_cv': r'F:\Kody\Data\6_QDM_DF_Statics\6_ModelUncertaintity\1_UncertaintityMetricsCal\simulated_multi_model_CV.tif',
        'corrected_cv': r'F:\Kody\Data\6_QDM_DF_Statics\6_ModelUncertaintity\1_UncertaintityMetricsCal\corrected_multi_model_CV.tif',
        'climate_zones': r'E:\Code\Pycharm\DF_UnCC\Data\ClimateZone\koppen_geiger_GloH20tif\ClimateZone5ClassMerge.shp',
        'catchment_area': r'F:\Kody\Data\6_QDM_DF_Pred\ACCESS\1_Input\2_RF_Pred\1_StaticFa\2_Mask\CA.tif',
        'river_data': r'F:\Kody\Data\6_QDM_DF_Statics\7_2_ComWithISIMIP_Advanced\1_Ensemble\1_DataMerge\Fut_Value.xlsx',
        'output_dir': r'F:\Kody\Data\6_QDM_DF_Statics\8_PlotOptimized\3.2_Results_Optimized2-fullSample'
    },
    'data_types': {
        'Simulated': {'color': '#E07B39', 'label': 'Uncorrected'},
        'Corrected': {'color': '#4682B4', 'label': 'Corrected'}
    },
    'river_data_types': {
        'Cascade_QDM': {'color': '#4682B4', 'label': 'Corrected', 'order': 1},
        'Cascade': {'color': '#E07B39', 'label': 'Uncorrected', 'order': 2},
        'ISIMIP': {'color': '#2CA02C', 'label': 'ISIMIP3b', 'order': 3}
    },
    'climate_labels': ['Arid', 'Cold', 'Polar', 'Temperate', 'Tropical'],
    'continent_abbr': {
        'Africa': 'AF', 'Asia': 'AS', 'Europe': 'EU',
        'North America': 'NA', 'Oceania': 'OC', 'South America': 'SA'
    },
    'watershed_categories': {
        'Small': (0, 100),
        'Medium': (100, 10000),
        'Large': (10000, np.inf)
    },
    'watershed_labels': {
        'Small': 'Small\n(<100 km²)',
        'Medium': 'Medium\n(100-10k km²)',
        'Large': 'Large\n(>10k km²)'
    },
    'figure': {
        'map_width_cm': 15.92,
        'map_height_cm': 7.96,
        'box_width_cm': 7.96,
        'box_height_cm': 5.2,
        'font_size': 10,
        'font_family': 'Arial',
        'dpi': 480
    },
    'cv_percentile': (2, 98),
    'resample_factor': 25,
    'boxplot_subsample': 50000
}

plt.rcParams['font.family'] = CONFIG['figure']['font_family']
plt.rcParams['font.size'] = CONFIG['figure']['font_size']


# ================================
# COLORMAP
# ================================
def create_cv_colormap():
    """CV colormap: light grey -> deep blue."""
    colors = [
        '#d9d9d9', '#4292c6', '#2171b5', '#08519c',
        '#08306b', '#041c4a', '#021238'
    ]
    return LinearSegmentedColormap.from_list('cv_cmap', colors, N=100)


# ================================
# DATA LOADING & RESAMPLING
# ================================
def load_cv_data(file_path):
    """Load and preprocess CV data."""
    with rasterio.open(file_path) as src:
        data = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
    data = np.ma.masked_where((data == 0) | (data < 0) | ~np.isfinite(data), data)
    return data, transform, bounds


def resample_for_map(data, transform, factor=25):
    """Resample for MAP VISUALIZATION ONLY (median aggregation)."""
    work_data = data.filled(np.nan) if hasattr(data, 'mask') else data.copy()
    new_height = data.shape[0] // factor
    new_width = data.shape[1] // factor
    work_data = work_data[:new_height * factor, :new_width * factor]
    reshaped = work_data.reshape((new_height, factor, new_width, factor))
    resampled = np.nanmedian(reshaped, axis=(1, 3))
    new_transform = rasterio.Affine(
        transform.a * factor, transform.b, transform.c,
        transform.d, transform.e * factor, transform.f
    )
    return resampled, new_transform


# ================================
# MASK CREATION
# ================================
def create_raster_mask(geometries, values, out_shape, transform):
    """Generic rasterize wrapper."""
    shapes = [(g, v) for g, v in zip(geometries, values) if v > 0]
    return rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)


def build_continent_geodata(world_gdf):
    """Build continent ID mapping and filtered GeoDataFrame."""
    gdf = world_gdf[~world_gdf.continent.isin(['Seven seas (open ocean)', 'Antarctica'])].copy()
    cont_labels_map = {}
    cont_map_int = {}
    current_id = 1
    for _, row in gdf.iterrows():
        c_name = row['continent']
        if c_name in CONFIG['continent_abbr'] and c_name not in cont_map_int:
            cont_map_int[c_name] = current_id
            cont_labels_map[current_id] = CONFIG['continent_abbr'][c_name]
            current_id += 1
    gdf['cont_id'] = gdf['continent'].map(cont_map_int).fillna(0).astype(int)
    return gdf, cont_labels_map


# ================================
# FULL-RESOLUTION STATISTICS
# ================================
def extract_fullres_dual_stats(sim_full, cor_full, mask_raster,
                               labels_map, is_climate=False, add_global=False):
    """
    Extract FULL-RESOLUTION statistics for Simulated & Corrected.
    Every valid pixel contributes — NO sampling.
    """
    sim_list, cor_list, labels = [], [], []

    def _process(mask, label_name):
        count = np.sum(mask)
        if count == 0:
            sim_list.append(np.array([]))
            cor_list.append(np.array([]))
            labels.append(label_name)
            return
        s_vals = sim_full[mask]
        c_vals = cor_full[mask]
        both_valid = np.isfinite(s_vals) & np.isfinite(c_vals)
        sim_list.append(s_vals[both_valid])
        cor_list.append(c_vals[both_valid])
        labels.append(label_name)

    valid = np.isfinite(sim_full) & np.isfinite(cor_full)

    if add_global:
        global_mask = (mask_raster > 0) & valid
        _process(global_mask, "Global")

    if is_climate:
        iterator = range(1, len(CONFIG['climate_labels']) + 1)
    else:
        unique_ids = np.unique(mask_raster)
        iterator = sorted(unique_ids[unique_ids != 0])

    for region_id in iterator:
        region_mask = (mask_raster == region_id) & valid
        if is_climate and labels_map:
            lbl = labels_map[region_id - 1]
        elif labels_map:
            lbl = labels_map.get(region_id, str(region_id))
        else:
            lbl = str(region_id)
        _process(region_mask, lbl)

    return sim_list, cor_list, labels


def extract_fullres_watershed_stats(sim_full, cor_full, catchment_full):
    """
    Extract FULL-RESOLUTION statistics by watershed scale.
    Every valid pixel contributes.
    """
    sim_list, cor_list, labels = [], [], []
    valid = np.isfinite(sim_full) & np.isfinite(cor_full) & np.isfinite(catchment_full)

    # Global
    global_mask = valid & (catchment_full > 0)
    if np.sum(global_mask) > 0:
        s_vals = sim_full[global_mask]
        c_vals = cor_full[global_mask]
        sim_list.append(s_vals)
        cor_list.append(c_vals)
        labels.append("Global")

    # By category
    for cat_name, (min_size, max_size) in CONFIG['watershed_categories'].items():
        if max_size == np.inf:
            size_mask = (catchment_full >= min_size) & valid
        else:
            size_mask = (catchment_full >= min_size) & (catchment_full < max_size) & valid
        count = np.sum(size_mask)
        if count > 0:
            sim_list.append(sim_full[size_mask])
            cor_list.append(cor_full[size_mask])
        else:
            sim_list.append(np.array([]))
            cor_list.append(np.array([]))
        labels.append(cat_name)

    return sim_list, cor_list, labels


# ================================
# RIVER CV EXTRACTION (unchanged logic)
# ================================
def extract_river_cv_data(excel_path):
    """Extract CV from river data for 3 data sources."""
    print("    Loading river data from Excel...")
    df = pd.read_excel(excel_path)

    cascade_cols = ['Cascade_ACCESS', 'Cascade_EC-Earth', 'Cascade_CanESM5',
                    'Cascade_IPSL', 'Cascade_MPI', 'Cascade_MRI',
                    'Cascade_MIRO', 'Cascade_INM', 'Cascade_NorE']
    cascade_qdm_cols = ['Cascade_ACCESS_QDM', 'Cascade_CanESM5_QDM', 'Cascade_EC-Earth_QDM',
                        'Cascade_MPI_QDM', 'Cascade_IPSL_QDM', 'Cascade_INM_QDM',
                        'Cascade_MIRO_QDM', 'Cascade_MRI_QDM', 'Cascade_NorE_QDM']
    isimip_cols = [col for col in df.columns if 'ISIMIP' in col]

    cv_data = []
    for _, row in df.iterrows():
        cascade_values = row[cascade_cols].dropna().values
        if len(cascade_values) > 1:
            cv_val = np.std(cascade_values) / np.mean(cascade_values) if np.mean(cascade_values) > 0 else np.nan
            if np.isfinite(cv_val):
                cv_data.append({'Type': 'Cascade', 'CV': cv_val})

        qdm_values = row[cascade_qdm_cols].dropna().values
        if len(qdm_values) > 1:
            cv_val = np.std(qdm_values) / np.mean(qdm_values) if np.mean(qdm_values) > 0 else np.nan
            if np.isfinite(cv_val):
                cv_data.append({'Type': 'Cascade_QDM', 'CV': cv_val})

        isimip_values = row[isimip_cols].dropna().values
        if len(isimip_values) > 1:
            cv_val = np.std(isimip_values) / np.mean(isimip_values) if np.mean(isimip_values) > 0 else np.nan
            if np.isfinite(cv_val):
                cv_data.append({'Type': 'ISIMIP', 'CV': cv_val})

    cv_df = pd.DataFrame(cv_data)
    print(f"    Extracted {len(cv_df)} CV values from {len(df)} rivers")
    return cv_df


# ================================
# STATISTICS EXPORT
# ================================
def save_statistics_to_csv(sim_list, cor_list, labels, filename):
    """Save comprehensive full-resolution statistics to CSV."""
    stats_list = []
    for i, label in enumerate(labels):
        for model_key, data in [('Simulated', sim_list[i]), ('Corrected', cor_list[i])]:
            n = len(data) if hasattr(data, '__len__') else 0
            row = {
                'Region': label,
                'Model': CONFIG['data_types'][model_key]['label'],
                'Count': n,
                'Mean': np.mean(data) if n > 0 else 0,
                'Median': np.median(data) if n > 0 else 0,
                'Q25': np.percentile(data, 25) if n > 0 else 0,
                'Q75': np.percentile(data, 75) if n > 0 else 0,
                'Min': np.min(data) if n > 0 else 0,
                'Max': np.max(data) if n > 0 else 0,
                'Std': np.std(data) if n > 0 else 0,
            }
            stats_list.append(row)

    df = pd.DataFrame(stats_list)
    output_path = os.path.join(CONFIG['paths']['output_dir'], filename)
    df.to_csv(output_path, index=False)
    print(f"  Statistics saved: {output_path}")
    return df


# ================================
# PLOTTING FUNCTIONS
# ================================
def plot_global_map(ax, data, extent, cmap, label, model_text, vmin, vmax):
    """Plot global CV map with embedded colorbar."""
    im = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   alpha=0.85, aspect='auto', origin='upper', interpolation='nearest')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)

    xticks = np.arange(-150, 151, 60)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{abs(x)}°W' if x < 0 else f'{x}°E' if x > 0 else '0°' for x in xticks])

    yticks = np.arange(-60, 91, 30)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{abs(y)}°S' if y < 0 else f'{y}°N' if y > 0 else '0°' for y in yticks])

    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=CONFIG['figure']['font_size'] + 2, fontweight='bold',
            va='top', ha='left')
    ax.text(0.02, 0.08, model_text, transform=ax.transAxes,
            fontsize=CONFIG['figure']['font_size'], va='center', ha='left')

    cbar_ax = inset_axes(ax, width="36%", height="5%", loc='lower center',
                         bbox_to_anchor=(0.11, 0.07, 1, 1), bbox_transform=ax.transAxes)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Coefficient of Variation', fontsize=CONFIG['figure']['font_size'] - 2)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_label_coords(1.0, 1.5)
    cbar.ax.xaxis.label.set_horizontalalignment('right')
    cbar.ax.tick_params(labelsize=CONFIG['figure']['font_size'] - 4)
    cbar.ax.minorticks_off()
    return im


def plot_grouped_boxplot(ax, sim_data, cor_data, labels, label_text,
                         ylabel=None, show_legend=False, rotation=0):
    """Plot grouped boxplot for two models (full-res data, subsampled for rendering)."""
    n_groups = len(labels)
    indices = np.arange(n_groups)
    width = 0.3
    max_pts = CONFIG['boxplot_subsample']
    rng = np.random.default_rng(42)

    def _sub(arr):
        if hasattr(arr, '__len__') and len(arr) > max_pts:
            return arr[rng.choice(len(arr), size=max_pts, replace=False)]
        return arr

    sim_plot = [_sub(d) for d in sim_data]
    cor_plot = [_sub(d) for d in cor_data]

    bp1 = ax.boxplot(sim_plot, positions=indices - width / 1.5, widths=width,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(linewidth=1.2, color='black'),
                     medianprops=dict(color='white', linewidth=1.8),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))

    bp2 = ax.boxplot(cor_plot, positions=indices + width / 1.5, widths=width,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(linewidth=1.2, color='black'),
                     medianprops=dict(color='white', linewidth=1.8),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))

    for patch in bp1['boxes']:
        patch.set_facecolor(CONFIG['data_types']['Simulated']['color'])
        patch.set_alpha(0.9)
    for patch in bp2['boxes']:
        patch.set_facecolor(CONFIG['data_types']['Corrected']['color'])
        patch.set_alpha(0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=CONFIG['figure']['font_size'],
                       rotation=rotation, ha='right' if rotation > 0 else 'center')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=CONFIG['figure']['font_size'])

    ax.set_ylim(bottom=0)

    ax.text(0.05, 0.98, label_text, transform=ax.transAxes,
            fontsize=CONFIG['figure']['font_size'] + 2, fontweight='bold', va='top')

    if show_legend:
        legend_elements = [
            Patch(facecolor=CONFIG['data_types']['Simulated']['color'],
                  label=CONFIG['data_types']['Simulated']['label'], alpha=0.9),
            Patch(facecolor=CONFIG['data_types']['Corrected']['color'],
                  label=CONFIG['data_types']['Corrected']['label'], alpha=0.9)
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=CONFIG['figure']['font_size'] - 1, frameon=True)


def plot_river_cv_boxplot(ax, df, label):
    """Plot boxplot for 3 data sources from river data with scatter points."""
    data_types = ['Cascade_QDM', 'Cascade', 'ISIMIP']
    positions = [1, 2, 3]
    plot_data = []
    colors = []

    for dtype in data_types:
        type_data = df[df['Type'] == dtype]['CV'].values
        if len(type_data) > 0:
            plot_data.append(type_data)
            colors.append(CONFIG['river_data_types'][dtype]['color'])

    bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                    showfliers=False, widths=0.6,
                    boxprops=dict(linewidth=1.2, alpha=0.7),
                    medianprops=dict(color='white', linewidth=1.8),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Scatter points
    for i, (dtype, pos) in enumerate(zip(data_types, positions)):
        type_data = df[df['Type'] == dtype]['CV'].values
        if len(type_data) > 0:
            x_jitter = np.random.normal(pos, 0.08, size=len(type_data))
            ax.scatter(x_jitter, type_data, alpha=0.4, s=20,
                       color=colors[i], edgecolors='none', zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([CONFIG['river_data_types'][dt]['label'] for dt in data_types],
                       fontsize=CONFIG['figure']['font_size'], rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.05, 0.98, label, transform=ax.transAxes,
            fontsize=CONFIG['figure']['font_size'] + 2, fontweight='bold', va='top')


# ================================
# MAIN PIPELINE
# ================================
def create_composite_figure():
    """
    Main pipeline:
    - Maps: resampled 25km
    - Statistics: full-resolution (climate, continent, watershed)
    - Generates TWO figures:
        Version 1: Panel d = River CV boxplot (original)
        Version 2: Panel d = Continental analysis (new)
    """
    print("=" * 80)
    print("CV UNCERTAINTY ANALYSIS — FULL-RESOLUTION STATISTICS")
    print("  Maps   -> Resampled 25km (visualization)")
    print("  Stats  -> Full-resolution all pixels (accuracy)")
    print("  Output -> 2 figure versions + 3 CSV statistics files")
    print("=" * 80)

    start_time = time.time()
    os.makedirs(CONFIG['paths']['output_dir'], exist_ok=True)

    factor = CONFIG['resample_factor']

    # ──────────────────────────────────────────────
    # STEP 1: Load raw data (full resolution)
    # ──────────────────────────────────────────────
    print("\n[1/9] Loading raw data (full resolution)...")
    sim_data, sim_transform, _ = load_cv_data(CONFIG['paths']['simulated_cv'])
    cor_data, cor_transform, _ = load_cv_data(CONFIG['paths']['corrected_cv'])

    sim_full = sim_data.filled(np.nan)
    cor_full = cor_data.filled(np.nan)
    print(f"  Full-resolution shape: {sim_full.shape}  "
          f"({sim_full.shape[0] * sim_full.shape[1]:,} pixels)")

    # ──────────────────────────────────────────────
    # STEP 2: Resample for MAP visualization only
    # ──────────────────────────────────────────────
    print("\n[2/9] Resampling for map visualization...")
    sim_25km, sim_new_transform = resample_for_map(sim_data, sim_transform, factor)
    cor_25km, cor_new_transform = resample_for_map(cor_data, cor_transform, factor)
    sim_extent = array_bounds(sim_25km.shape[0], sim_25km.shape[1], sim_new_transform)
    cor_extent = array_bounds(cor_25km.shape[0], cor_25km.shape[1], cor_new_transform)

    all_valid = np.concatenate([sim_25km[~np.isnan(sim_25km)], cor_25km[~np.isnan(cor_25km)]])
    vmin = np.percentile(all_valid, CONFIG['cv_percentile'][0])
    vmax = np.percentile(all_valid, CONFIG['cv_percentile'][1])
    print(f"  Colormap range: {vmin:.3f} - {vmax:.3f}")

    del sim_data, cor_data
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 3: Build full-resolution masks
    # ──────────────────────────────────────────────
    print("\n[3/9] Building full-resolution masks...")

    # Climate zones
    print("  > Rasterizing climate zones...")
    climate_gdf = gpd.read_file(CONFIG['paths']['climate_zones'])
    zone_map = {zone: i + 1 for i, zone in enumerate(CONFIG['climate_labels'])}
    climate_gdf['zone_id'] = climate_gdf['Name'].map(zone_map)
    climate_raster_full = create_raster_mask(
        climate_gdf.geometry, climate_gdf['zone_id'],
        out_shape=sim_full.shape, transform=sim_transform
    )
    del climate_gdf
    gc.collect()

    # Continents
    print("  > Rasterizing continents...")
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except Exception:
        world = gpd.read_file("naturalearth_lowres")
    world_filtered, cont_labels_map = build_continent_geodata(world)
    continent_raster_full = create_raster_mask(
        world_filtered.geometry, world_filtered['cont_id'],
        out_shape=sim_full.shape, transform=sim_transform
    )
    del world, world_filtered
    gc.collect()

    # Catchment area
    print("  > Loading catchment area...")
    with rasterio.open(CONFIG['paths']['catchment_area']) as src:
        catchment_full = src.read(1).astype(np.float32)
    catchment_full[(catchment_full == 0) | (catchment_full == -9999)] = np.nan

    # ──────────────────────────────────────────────
    # STEP 4: Extract full-resolution statistics
    # ──────────────────────────────────────────────
    print("\n[4/9] Extracting full-resolution statistics...")

    # Climate (Global + 5 zones)
    print("  > Climate zone statistics...")
    clim_sim, clim_cor, clim_lbls = extract_fullres_dual_stats(
        sim_full, cor_full, climate_raster_full,
        CONFIG['climate_labels'], is_climate=True, add_global=True
    )
    for i, lbl in enumerate(clim_lbls):
        n = len(clim_sim[i]) if hasattr(clim_sim[i], '__len__') else 0
        print(f"    {lbl}: {n:,} pixels")

    # Continents (6 continents)
    print("  > Continental statistics...")
    cont_sim, cont_cor, cont_lbls = extract_fullres_dual_stats(
        sim_full, cor_full, continent_raster_full,
        cont_labels_map, is_climate=False, add_global=False
    )
    for i, lbl in enumerate(cont_lbls):
        n = len(cont_sim[i]) if hasattr(cont_sim[i], '__len__') else 0
        print(f"    {lbl}: {n:,} pixels")

    # Watershed scales (Global + 3 categories)
    print("  > Watershed scale statistics...")
    ws_sim, ws_cor, ws_lbls = extract_fullres_watershed_stats(
        sim_full, cor_full, catchment_full
    )
    for i, lbl in enumerate(ws_lbls):
        n = len(ws_sim[i]) if hasattr(ws_sim[i], '__len__') else 0
        print(f"    {lbl}: {n:,} pixels")

    # Free full-resolution data
    del sim_full, cor_full, climate_raster_full, continent_raster_full, catchment_full
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 5: Export statistics to CSV
    # ──────────────────────────────────────────────
    print("\n[5/9] Exporting statistics to CSV...")
    save_statistics_to_csv(clim_sim, clim_cor, clim_lbls, 'statistics_climate_fullres.csv')
    save_statistics_to_csv(cont_sim, cont_cor, cont_lbls, 'statistics_continent_fullres.csv')
    save_statistics_to_csv(ws_sim, ws_cor, ws_lbls, 'statistics_watershed_fullres.csv')

    # ──────────────────────────────────────────────
    # STEP 6: Extract river CV data
    # ──────────────────────────────────────────────
    print("\n[6/9] Extracting river CV data...")
    river_cv_df = extract_river_cv_data(CONFIG['paths']['river_data'])

    # ──────────────────────────────────────────────
    # STEP 7: Plot Version 1 — Panel d = River CV (original)
    # ──────────────────────────────────────────────
    print("\n[7/9] Plotting Version 1 (d = River CV)...")
    _plot_figure(
        sim_25km, cor_25km, sim_extent, cor_extent, vmin, vmax,
        clim_sim, clim_cor, clim_lbls,
        panel_d_type='river', river_cv_df=river_cv_df,
        output_name='cv_uncertainty_v1_river.png'
    )

    # ──────────────────────────────────────────────
    # STEP 8: Plot Version 2 — Panel d = Continental analysis
    # ──────────────────────────────────────────────
    print("\n[8/9] Plotting Version 2 (d = Continental)...")
    _plot_figure(
        sim_25km, cor_25km, sim_extent, cor_extent, vmin, vmax,
        clim_sim, clim_cor, clim_lbls,
        panel_d_type='continent', cont_sim=cont_sim, cont_cor=cont_cor, cont_lbls=cont_lbls,
        output_name='cv_uncertainty_v2_continent.png'
    )

    # ──────────────────────────────────────────────
    # STEP 9: Done
    # ──────────────────────────────────────────────
    del clim_sim, clim_cor, cont_sim, cont_cor, ws_sim, ws_cor
    gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"COMPLETED in {elapsed:.1f}s")
    print(f"  Version 1 (River CV)   : cv_uncertainty_v1_river.png")
    print(f"  Version 2 (Continental): cv_uncertainty_v2_continent.png")
    print(f"  Statistics: 3 CSV files (climate, continent, watershed)")
    print(f"{'=' * 80}")


def _plot_figure(sim_25km, cor_25km, sim_extent, cor_extent, vmin, vmax,
                 clim_sim, clim_cor, clim_lbls,
                 panel_d_type='river',
                 river_cv_df=None,
                 cont_sim=None, cont_cor=None, cont_lbls=None,
                 output_name='output.png'):
    """
    Internal function to plot the 4-panel figure.
    panel_d_type: 'river' for original river CV, 'continent' for continental analysis.
    """
    cm = 1 / 2.54
    map_height = CONFIG['figure']['map_height_cm'] * cm
    box_height = CONFIG['figure']['box_height_cm'] * cm
    fig_width = CONFIG['figure']['map_width_cm'] * cm
    fig_height = 2 * map_height + box_height + 1.5 * cm

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[map_height, map_height, box_height],
                  hspace=0.3, wspace=0.25)

    cmap = create_cv_colormap()

    # Panel (a): Simulated CV map
    ax_a = fig.add_subplot(gs[0, :])
    plot_global_map(ax_a, sim_25km,
                    [sim_extent[0], sim_extent[2], sim_extent[1], sim_extent[3]],
                    cmap, 'a', CONFIG['data_types']['Simulated']['label'], vmin, vmax)

    # Panel (b): Corrected CV map
    ax_b = fig.add_subplot(gs[1, :])
    plot_global_map(ax_b, cor_25km,
                    [cor_extent[0], cor_extent[2], cor_extent[1], cor_extent[3]],
                    cmap, 'b', CONFIG['data_types']['Corrected']['label'], vmin, vmax)

    # Panel (c): Climate zone boxplot — full-resolution
    ax_c = fig.add_subplot(gs[2, 0])
    plot_grouped_boxplot(ax_c, clim_sim, clim_cor, clim_lbls,
                         'c', ylabel='Coefficient of Variation',
                         show_legend=True, rotation=30)

    # Panel (d): depends on version
    ax_d = fig.add_subplot(gs[2, 1])

    if panel_d_type == 'river' and river_cv_df is not None:
        plot_river_cv_boxplot(ax_d, river_cv_df, 'd')
    elif panel_d_type == 'continent' and cont_sim is not None:
        plot_grouped_boxplot(ax_d, cont_sim, cont_cor, cont_lbls,
                             'd', ylabel=None, show_legend=False, rotation=0)

    output_path = os.path.join(CONFIG['paths']['output_dir'], output_name)
    plt.savefig(output_path, dpi=CONFIG['figure']['dpi'],
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    create_composite_figure()