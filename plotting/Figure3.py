"""
Figure 3 - Flood return-period change.

Panels: (a) corrected return-period map, (b) difference map
(corrected - uncorrected), (c)/(d) signed-difference box plots by
Koppen-Geiger climate zone / continent, (e)/(f) stacked bars of under- vs
over-estimation share. Maps use a 25x block resample for display only; the
statistics use every native ~1 km pixel.

Usage
-----
Download and unpack the plotting-data archive (Zenodo concept DOI
10.5281/zenodo.19357539), then either set the environment variable
EFBC_PLOTDATA_DIR to the unpacked "efbc_PlotData" folder or edit DATA_DIR below.
Continent masks come from Natural Earth via geopandas; on geopandas >= 1.0 set
NATURALEARTH_SHP to a local ne_110m_admin_0_countries.shp.

    python Figure3.py

The figure PNG and statistics_*.csv are written to <Figure_3>/output.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
import time
import gc
import warnings

warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION
# ================================
# Root of the unpacked "efbc_PlotData" archive; override with EFBC_PLOTDATA_DIR.
# Defaults to the folder holding this script.
DATA_DIR = os.environ.get('EFBC_PLOTDATA_DIR', os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(DATA_DIR, 'Figure_3')
OUTPUT_DIR = os.environ.get('EFBC_OUTPUT_DIR', os.path.join(FIG_DIR, 'output'))

CONFIG = {
    'paths': {
        'simulated_rp': os.path.join(FIG_DIR, 'uncorrected_return-period-years.tif'),
        'corrected_rp': os.path.join(FIG_DIR, 'corrected_return-period-years.tif'),
        'climate_zones': os.path.join(FIG_DIR, 'climate-zones_Koppen-Geiger-5class.shp'),
        'output_dir': OUTPUT_DIR
    },
    'climate_labels': ['Arid', 'Cold', 'Polar', 'Temperate', 'Tropical'],
    'continent_abbr': {
        'Africa': 'AF', 'Asia': 'AS', 'Europe': 'EU',
        'North America': 'NA', 'Oceania': 'OC', 'South America': 'SA'
    },
    'colors': {
        # Diff < 0 -> Underestimation (Orange); Diff > 0 -> Overestimation (Blue)
        'Underestimation': '#E07B39',
        'Overestimation': '#4682B4',
    },
    'figure': {
        'map_width_cm': 15.92,
        'map_height_cm': 7.96,
        'box_width_cm': 7.96,
        'box_height_cm': 5.5,
        'bar_height_cm': 5.5,
        'font_size': 10,
        'font_family': 'Arial',
        'dpi': 480
    },
    'rp_center': 100,
    'diff_bounds': [-250, -100, -50, -25, -10, -5, 0, 5, 10, 25, 50, 100, 250],
    'boxplot_ylim': 75,
    'resample_factor': 25  # 重采样因子，仅用于绘图
}

# Set global font
plt.rcParams['font.family'] = CONFIG['figure']['font_family']
plt.rcParams['font.size'] = CONFIG['figure']['font_size']


# ================================
# COLORMAP FUNCTIONS
# ================================
def create_rp_colormap_inverted():
    """Create inverted return period colormap."""
    colors = [
        '#a50f15', '#de2d26', '#fb6a4a', '#fcae91', '#fee5d9',
        '#ffffff',
        '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'
    ]
    return LinearSegmentedColormap.from_list('rp_cmap_inv', colors, N=100)


def create_difference_colormap():
    """Create difference colormap."""
    colors_neg = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#deebf7']
    colors_pos = ['#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#a50026']
    all_colors = colors_pos[::-1] + colors_neg[::-1]
    return mcolors.ListedColormap(all_colors)


# ================================
# DATA LOADING & RESAMPLING
# ================================
def load_rp_data(file_path):
    """Load return period raster data."""
    with rasterio.open(file_path) as src:
        data = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
    data = np.ma.masked_where((data == 0) | (data < 0) | ~np.isfinite(data), data)
    return data, transform, bounds


def resample_median(data, transform, factor):
    """
    Resample data by aggregating factor×factor blocks via median.
    Used ONLY for map visualization, NOT for statistics.
    """
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
# MASK CREATION (SHARED UTILITY)
# ================================
def build_continent_geodata(world_gdf):
    """
    Build continent ID mapping and filtered GeoDataFrame.
    Returns: (filtered_gdf, cont_labels_map)
      - filtered_gdf: GeoDataFrame with 'cont_id' column
      - cont_labels_map: {int_id: abbreviation_string}
    """
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


def create_raster_mask(geometries, values, out_shape, transform):
    """
    Generic rasterize wrapper.
    geometries: iterable of shapely geometries
    values: iterable of integer IDs matching geometries
    """
    shapes = [(g, v) for g, v in zip(geometries, values) if v > 0]
    return rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)


# ================================
# FULL-RESOLUTION STATISTICS (FROM CODE2 APPROACH)
# ================================
def extract_fullres_stats(diff_data, mask_raster, labels_map=None, is_climate=False, add_global=False):
    """
    Extract FULL-RESOLUTION statistics at original pixel level.
    No resampling — every valid pixel contributes to statistics.

    Returns:
        box_under:  list of arrays (abs values where diff < 0)
        box_over:   list of arrays (values where diff > 0)
        pct_under:  list of floats (percentage of underestimation pixels)
        pct_over:   list of floats (percentage of overestimation pixels)
        labels:     list of region label strings
    """
    box_under, box_over = [], []
    pct_under, pct_over = [], []
    labels = []

    def _process(mask, label_name):
        """Process a single region mask."""
        count = np.sum(mask)
        if count == 0:
            box_under.append(np.array([]))
            box_over.append(np.array([]))
            pct_under.append(0.0)
            pct_over.append(0.0)
            labels.append(label_name)
            return

        vals = diff_data[mask]
        total = len(vals)

        # Underestimation (diff < 0): store absolute values for boxplot magnitude
        u_vals = np.abs(vals[vals < 0])
        # Overestimation (diff > 0): store raw positive values
        o_vals = vals[vals > 0]

        box_under.append(u_vals if len(u_vals) > 0 else np.array([]))
        box_over.append(o_vals if len(o_vals) > 0 else np.array([]))

        pct_under.append((len(u_vals) / total * 100) if total > 0 else 0.0)
        pct_over.append((len(o_vals) / total * 100) if total > 0 else 0.0)
        labels.append(label_name)

    # Valid data mask (works for both np.ndarray and np.ma.MaskedArray)
    if hasattr(diff_data, 'mask'):
        valid = ~diff_data.mask & np.isfinite(diff_data.data)
        diff_flat = diff_data.data  # underlying array
    else:
        valid = np.isfinite(diff_data)
        diff_flat = diff_data

    # Global
    if add_global:
        global_mask = (mask_raster > 0) & valid
        _process(global_mask, "Global")

    # Individual regions
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

    return box_under, box_over, pct_under, pct_over, labels


# ================================
# STATISTICS EXPORT
# ================================
def save_statistics_to_csv(box_under, box_over, pct_under, pct_over, labels, filename):
    """
    Calculate comprehensive summary statistics and save to CSV.
    Enhanced with Count, Max/Min from Code2 approach.
    """
    stats_list = []

    for i, label in enumerate(labels):
        u_data = box_under[i]
        o_data = box_over[i]

        u_n = len(u_data) if hasattr(u_data, '__len__') else 0
        o_n = len(o_data) if hasattr(o_data, '__len__') else 0

        row = {
            'Region': label,
            'Total_Count': u_n + o_n,
            'Underestimation_Count': u_n,
            'Underestimation_Pct': pct_under[i],
            'Under_Mean_Abs': np.mean(u_data) if u_n > 0 else 0,
            'Under_Median_Abs': np.median(u_data) if u_n > 0 else 0,
            'Under_IQR_25': np.percentile(u_data, 25) if u_n > 0 else 0,
            'Under_IQR_75': np.percentile(u_data, 75) if u_n > 0 else 0,
            'Under_Max_Abs': np.max(u_data) if u_n > 0 else 0,
            'Overestimation_Count': o_n,
            'Overestimation_Pct': pct_over[i],
            'Over_Mean': np.mean(o_data) if o_n > 0 else 0,
            'Over_Median': np.median(o_data) if o_n > 0 else 0,
            'Over_IQR_25': np.percentile(o_data, 25) if o_n > 0 else 0,
            'Over_IQR_75': np.percentile(o_data, 75) if o_n > 0 else 0,
            'Over_Max': np.max(o_data) if o_n > 0 else 0,
        }
        stats_list.append(row)

    df = pd.DataFrame(stats_list)
    output_path = os.path.join(CONFIG['paths']['output_dir'], filename)
    df.to_csv(output_path, index=False)
    print(f"  Statistics saved: {output_path}")
    return df


# ================================
# PLOTTING FUNCTIONS (UNCHANGED LOGIC)
# ================================
def plot_global_map(ax, data, extent, cmap, norm, label, model_text, is_difference=False):
    """Plot global map panel."""
    if is_difference:
        data_plot = np.clip(data, CONFIG['diff_bounds'][0], CONFIG['diff_bounds'][-1])
    else:
        data_plot = data

    im = ax.imshow(
        data_plot, extent=extent, cmap=cmap, norm=norm,
        alpha=0.85, aspect='auto', origin='upper', interpolation='nearest'
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)

    ax.set_xticks(np.arange(-150, 151, 60))
    ax.set_xticklabels([
        f'{abs(x)}°W' if x < 0 else f'{x}°E' if x > 0 else '0°'
        for x in np.arange(-150, 151, 60)
    ])
    ax.set_yticks(np.arange(-60, 91, 30))
    ax.set_yticklabels([
        f'{abs(y)}°S' if y < 0 else f'{y}°N' if y > 0 else '0°'
        for y in np.arange(-60, 91, 30)
    ])

    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=12,
            va='top', ha='left', fontweight='bold')
    ax.text(0.02, 0.08, model_text, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='left')

    # Inset colorbar
    cbar_ax = inset_axes(
        ax, width="36%", height="5%", loc='lower center',
        bbox_to_anchor=(0.11, 0.07, 1, 1), bbox_transform=ax.transAxes
    )
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')

    if is_difference:
        ticks = [-100, -50, -25, -10, 0, 10, 25, 50, 100]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])
        cbar.set_label('Difference (years)', fontsize=8)
    else:
        ticks = [2, 50, 100, 200]
        cbar.set_ticks(ticks)
        cbar.set_label('Return Period (years)', fontsize=8)

    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=7)


def plot_grouped_boxplot(ax, data_under, data_over, labels, label_text,
                         ylabel, rotation=0, show_legend=False):
    """Plot grouped boxplot panel."""
    n_groups = len(labels)
    indices = np.arange(n_groups)
    width = 0.3

    bp1 = ax.boxplot(
        data_under, positions=indices - width / 1.5, widths=width,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=1, color='black'),
        medianprops=dict(color='white', linewidth=1.5)
    )
    bp2 = ax.boxplot(
        data_over, positions=indices + width / 1.5, widths=width,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=1, color='black'),
        medianprops=dict(color='white', linewidth=1.5)
    )

    for patch in bp1['boxes']:
        patch.set_facecolor(CONFIG['colors']['Underestimation'])
        patch.set_alpha(0.9)
    for patch in bp2['boxes']:
        patch.set_facecolor(CONFIG['colors']['Overestimation'])
        patch.set_alpha(0.9)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=CONFIG['figure']['font_size'], rotation=rotation)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=CONFIG['figure']['font_size'])

    ax.set_ylim(0, CONFIG['boxplot_ylim'])
    ax.text(0.07, 0.98, label_text, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    if show_legend:
        legend_elements = [
            Patch(facecolor=CONFIG['colors']['Underestimation'], label='Underestimation'),
            Patch(facecolor=CONFIG['colors']['Overestimation'], label='Overestimation')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=False)


def plot_stacked_bar(ax, pct_under, pct_over, labels, label_text,
                     ylabel, rotation=0, show_legend=False):
    """Plot stacked bar chart panel."""
    indices = np.arange(len(labels))
    width = 0.5

    ax.bar(indices, pct_under, width, label='Underestimation',
           color=CONFIG['colors']['Underestimation'], alpha=0.9,
           edgecolor='white', linewidth=0.5)
    ax.bar(indices, pct_over, width, bottom=pct_under, label='Overestimation',
           color=CONFIG['colors']['Overestimation'], alpha=0.9,
           edgecolor='white', linewidth=0.5)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=CONFIG['figure']['font_size'], rotation=rotation)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=CONFIG['figure']['font_size'])

    ax.set_ylim(0, 100)
    ax.text(0.065, 0.98, label_text, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    if show_legend:
        legend_elements = [
            Patch(facecolor=CONFIG['colors']['Underestimation'], label='Underestimation'),
            Patch(facecolor=CONFIG['colors']['Overestimation'], label='Overestimation')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=False)


# ================================
# MAIN PIPELINE
# ================================
def create_composite_figure():
    print("=" * 80)
    print("FIGURE GENERATION: 6-PANEL COMPOSITE")
    print("  Maps   -> Resampled 25km (for visualization)")
    print("  Stats  -> Full-resolution original pixels (for accuracy)")
    print("=" * 80)

    start_time = time.time()
    os.makedirs(CONFIG['paths']['output_dir'], exist_ok=True)

    factor = CONFIG['resample_factor']

    # ──────────────────────────────────────────────
    # STEP 1: Load raw data (full resolution)
    # ──────────────────────────────────────────────
    print("\n[1/7] Loading raw data (full resolution)...")
    sim_data, sim_transform, _ = load_rp_data(CONFIG['paths']['simulated_rp'])
    cor_data, cor_transform, _ = load_rp_data(CONFIG['paths']['corrected_rp'])

    # Full-resolution difference (for statistics)
    # Convert masked arrays to float with NaN for invalid
    sim_full = sim_data.filled(np.nan) if hasattr(sim_data, 'mask') else sim_data.copy()
    cor_full = cor_data.filled(np.nan) if hasattr(cor_data, 'mask') else cor_data.copy()
    diff_full = cor_full - sim_full  # Full-resolution signed difference
    print(f"  Full-resolution shape: {diff_full.shape}  "
          f"({diff_full.shape[0] * diff_full.shape[1]:,} pixels)")

    # ──────────────────────────────────────────────
    # STEP 2: Resample for MAP visualization only
    # ──────────────────────────────────────────────
    print("\n[2/7] Resampling for map visualization...")
    cor_25km, cor_new_transform = resample_median(cor_data, cor_transform, factor)
    sim_25km, sim_new_transform = resample_median(sim_data, sim_transform, factor)
    diff_25km = cor_25km - sim_25km
    cor_extent = array_bounds(cor_25km.shape[0], cor_25km.shape[1], cor_new_transform)
    print(f"  Resampled shape: {cor_25km.shape}  (for maps only)")

    # Free raw masked arrays (keep full float arrays for stats)
    del sim_data, cor_data
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 3: Build masks at FULL resolution (for statistics)
    # ──────────────────────────────────────────────
    print("\n[3/7] Building full-resolution masks for statistics...")

    # --- Climate Zone Mask (full resolution) ---
    print("  > Loading climate zones shapefile...")
    climate_gdf = gpd.read_file(CONFIG['paths']['climate_zones'])
    zone_map = {zone: i + 1 for i, zone in enumerate(CONFIG['climate_labels'])}
    climate_gdf['zone_id'] = climate_gdf['Name'].map(zone_map)

    print("  > Rasterizing climate zones at full resolution...")
    climate_raster_full = create_raster_mask(
        climate_gdf.geometry, climate_gdf['zone_id'],
        out_shape=diff_full.shape, transform=sim_transform
    )
    del climate_gdf
    gc.collect()

    # --- Continent Mask (full resolution) ---
    print("  > Loading continent shapefile...")
    ne_path = os.environ.get('NATURALEARTH_SHP')
    if ne_path:
        world = gpd.read_file(ne_path)
    else:
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except Exception:
            world = gpd.read_file("naturalearth_lowres")

    world_filtered, cont_labels_map = build_continent_geodata(world)

    print("  > Rasterizing continents at full resolution...")
    continent_raster_full = create_raster_mask(
        world_filtered.geometry, world_filtered['cont_id'],
        out_shape=diff_full.shape, transform=sim_transform
    )
    del world, world_filtered
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 4: Extract FULL-RESOLUTION statistics
    # ──────────────────────────────────────────────
    print("\n[4/7] Extracting full-resolution statistics...")

    # Climate zones (Global + 5 zones)
    print("  > Climate zone statistics...")
    clim_box_u, clim_box_o, clim_pct_u, clim_pct_o, clim_lbls = extract_fullres_stats(
        diff_full, climate_raster_full, CONFIG['climate_labels'],
        is_climate=True, add_global=True
    )
    for i, lbl in enumerate(clim_lbls):
        n_u = len(clim_box_u[i]) if hasattr(clim_box_u[i], '__len__') else 0
        n_o = len(clim_box_o[i]) if hasattr(clim_box_o[i], '__len__') else 0
        print(f"    {lbl}: Under={n_u:,}  Over={n_o:,}  "
              f"(%Under={clim_pct_u[i]:.1f}%  %Over={clim_pct_o[i]:.1f}%)")

    # Continents (6 continents, no Global — already in climate)
    print("  > Continental statistics...")
    cont_box_u, cont_box_o, cont_pct_u, cont_pct_o, cont_lbls = extract_fullres_stats(
        diff_full, continent_raster_full, cont_labels_map,
        is_climate=False, add_global=False
    )
    for i, lbl in enumerate(cont_lbls):
        n_u = len(cont_box_u[i]) if hasattr(cont_box_u[i], '__len__') else 0
        n_o = len(cont_box_o[i]) if hasattr(cont_box_o[i], '__len__') else 0
        print(f"    {lbl}: Under={n_u:,}  Over={n_o:,}  "
              f"(%Under={cont_pct_u[i]:.1f}%  %Over={cont_pct_o[i]:.1f}%)")

    # Free full-resolution arrays no longer needed
    del diff_full, climate_raster_full, continent_raster_full
    del sim_full, cor_full
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 5: Export statistics to CSV
    # ──────────────────────────────────────────────
    print("\n[5/7] Exporting statistics to CSV...")
    df_climate = save_statistics_to_csv(
        clim_box_u, clim_box_o, clim_pct_u, clim_pct_o, clim_lbls,
        'statistics_climate_fullres.csv'
    )
    df_continent = save_statistics_to_csv(
        cont_box_u, cont_box_o, cont_pct_u, cont_pct_o, cont_lbls,
        'statistics_continents_fullres.csv'
    )

    # ──────────────────────────────────────────────
    # STEP 6: Downsample boxplot data for rendering
    #   (Matplotlib cannot render millions of points efficiently;
    #    subsample for visual fidelity while keeping all statistics exact)
    # ──────────────────────────────────────────────
    print("\n[6/7] Preparing boxplot data for rendering...")
    max_points = 50000  # Per category per region — more than enough for visual fidelity

    def subsample_for_boxplot(data_list):
        """Subsample large arrays for boxplot rendering only.
        Statistics (median, quartiles) are preserved because we subsample
        uniformly at random from the full distribution."""
        result = []
        for arr in data_list:
            if hasattr(arr, '__len__') and len(arr) > max_points:
                rng = np.random.default_rng(42)  # Reproducible
                idx = rng.choice(len(arr), size=max_points, replace=False)
                result.append(arr[idx])
            else:
                result.append(arr)
        return result

    clim_box_u_plot = subsample_for_boxplot(clim_box_u)
    clim_box_o_plot = subsample_for_boxplot(clim_box_o)
    cont_box_u_plot = subsample_for_boxplot(cont_box_u)
    cont_box_o_plot = subsample_for_boxplot(cont_box_o)

    # Free full boxplot arrays
    del clim_box_u, clim_box_o, cont_box_u, cont_box_o
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 7: Plotting (resampled maps + full-res stats)
    # ──────────────────────────────────────────────
    print("\n[7/7] Plotting 6-panel figure...")
    cm = 1 / 2.54

    fig = plt.figure(figsize=(15.92 * cm, (7.96 * 2 + 5.5 * 2 + 2) * cm))
    gs = GridSpec(4, 2, figure=fig,
                  height_ratios=[7.96, 7.96, 5.5, 5.5],
                  hspace=0.4, wspace=0.25)

    # Colormaps & norms
    cmap_rp = create_rp_colormap_inverted()
    norm_cor = TwoSlopeNorm(vmin=2, vcenter=100, vmax=200)
    cmap_diff = create_difference_colormap()
    norm_diff = mcolors.BoundaryNorm(CONFIG['diff_bounds'], len(CONFIG['diff_bounds']) - 1)

    map_extent = [cor_extent[0], cor_extent[2], cor_extent[1], cor_extent[3]]

    # Panel (a): Corrected RP map — resampled
    ax_a = fig.add_subplot(gs[0, :])
    plot_global_map(ax_a, cor_25km, map_extent, cmap_rp, norm_cor, 'a', 'Corrected')

    # Panel (b): Difference map — resampled
    ax_b = fig.add_subplot(gs[1, :])
    plot_global_map(ax_b, diff_25km, map_extent, cmap_diff, norm_diff,
                    'b', 'Difference', is_difference=True)

    # Panel (c): Climate zone boxplot — full-resolution statistics
    ax_c = fig.add_subplot(gs[2, 0])
    plot_grouped_boxplot(ax_c, clim_box_u_plot, clim_box_o_plot, clim_lbls,
                         'c', 'Difference (years)', rotation=30, show_legend=True)

    # Panel (d): Continent boxplot — full-resolution statistics
    ax_d = fig.add_subplot(gs[2, 1])
    plot_grouped_boxplot(ax_d, cont_box_u_plot, cont_box_o_plot, cont_lbls,
                         'd', '', rotation=0, show_legend=False)

    # Panel (e): Climate zone bar chart — full-resolution percentages
    ax_e = fig.add_subplot(gs[3, 0])
    plot_stacked_bar(ax_e, clim_pct_u, clim_pct_o, clim_lbls,
                     'e', 'Percentage (%)', rotation=30, show_legend=False)

    # Panel (f): Continent bar chart — full-resolution percentages
    ax_f = fig.add_subplot(gs[3, 1])
    plot_stacked_bar(ax_f, cont_pct_u, cont_pct_o, cont_lbls,
                     'f', '', rotation=0, show_legend=False)

    # Save
    output_path = os.path.join(CONFIG['paths']['output_dir'], 'figure3_final_6panel_fullres_stats.png')
    plt.savefig(output_path, dpi=CONFIG['figure']['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"COMPLETED in {elapsed:.1f}s")
    print(f"  Output figure : {output_path}")
    print(f"  Statistics CSV : statistics_climate_fullres.csv")
    print(f"  Statistics CSV : statistics_continents_fullres.csv")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    create_composite_figure()