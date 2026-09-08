import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
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
        'cascade_bias': r'F:\Kody\Data\6_QDM_DF_Statics\1_Magnitude\6_SimHis_ObHis_Bias_GloPlot\Bias.tif',
        'vic_bias': r'F:\Kody\Data\6_QDM_DF_Statics\3_VIC_DF\1_BiasCal\100a_bias.tif',
        'climate_zones': r'E:\Code\Pycharm\DF_UnCC\Data\ClimateZone\koppen_geiger_GloH20tif\ClimateZone5ClassMerge.shp',
        'output_dir': r'F:\Kody\Data\6_QDM_DF_Statics\8_PlotOptimized\3.1_Results-Continent'
    },
    'models': {
        'Cascade': {'color': '#E07B39', 'label': 'Uncorrected'},
        'VIC': {'color': '#4682B4', 'label': 'GRADES'}
    },
    'climate_labels': ['Arid', 'Cold', 'Polar', 'Temperate', 'Tropical'],
    'continent_abbr': {
        'Africa': 'AF', 'Asia': 'AS', 'Europe': 'EU',
        'North America': 'NA', 'Oceania': 'OC', 'South America': 'SA'
    },
    'figure': {
        'map_width_cm': 15.92,
        'map_height_cm': 7.96,
        'box_height_cm': 5.2,
        'font_size': 10,
        'font_family': 'Arial',
        'dpi': 480
    },
    'resample_factor': 25,
    'boxplot_subsample': 50000  # Max points per model per region for rendering
}

plt.rcParams['font.family'] = CONFIG['figure']['font_family']
plt.rcParams['font.size'] = CONFIG['figure']['font_size']


# ================================
# COLORMAP
# ================================
def create_colormap(vmax=10.0):
    """Create hydrological bias colormap."""
    colors_blue = plt.cm.Blues_r(np.linspace(0.2, 0.9, 45))
    colors_blue_white = plt.cm.Blues_r(np.linspace(0.9, 1.0, 5))
    colors_white = np.ones((10, 4))
    colors_white_red = plt.cm.Reds(np.linspace(0.0, 0.2, 5))
    colors_red = plt.cm.Reds(np.linspace(0.2, 0.9, 45))

    all_colors = np.vstack([colors_blue, colors_blue_white, colors_white,
                            colors_white_red, colors_red])
    cmap = ListedColormap(all_colors)

    levels = np.concatenate([
        np.linspace(0.1, 0.85, 45),
        np.linspace(0.85, 0.95, 5),
        np.linspace(0.95, 1.05, 10),
        np.linspace(1.05, 1.15, 5),
        np.linspace(1.15, vmax, 45)
    ])
    norm = BoundaryNorm(levels, cmap.N)
    return cmap, norm


# ================================
# DATA LOADING & RESAMPLING
# ================================
def load_bias_data(file_path):
    """Load and preprocess bias data."""
    with rasterio.open(file_path) as src:
        data = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
    data = np.ma.masked_where((data == 0) | (data <= 0) | ~np.isfinite(data), data)
    return data, transform, bounds


def resample_for_map(data, transform, factor, method='mean'):
    """
    Resample data for MAP VISUALIZATION ONLY.
    Statistics are computed at full resolution separately.
    """
    work_data = data.filled(np.nan) if hasattr(data, 'mask') else data.copy()
    new_height = data.shape[0] // factor
    new_width = data.shape[1] // factor
    work_data = work_data[:new_height * factor, :new_width * factor]
    reshaped = work_data.reshape((new_height, factor, new_width, factor))

    if method == 'mean':
        resampled = np.nanmean(reshaped, axis=(1, 3))
    elif method == 'median':
        resampled = np.nanmedian(reshaped, axis=(1, 3))
    elif method == 'max':
        resampled = np.nanmax(reshaped, axis=(1, 3))
    else:
        raise ValueError(f"Unknown method: {method}")

    new_transform = rasterio.Affine(
        transform.a * factor, transform.b, transform.c,
        transform.d, transform.e * factor, transform.f
    )
    return resampled, new_transform


# ================================
# MASK CREATION
# ================================
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


def create_raster_mask(geometries, values, out_shape, transform):
    """Generic rasterize wrapper."""
    shapes = [(g, v) for g, v in zip(geometries, values) if v > 0]
    return rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)


# ================================
# FULL-RESOLUTION STATISTICS
# ================================
def extract_fullres_dual_model_stats(cascade_full, vic_full, mask_raster,
                                     labels_map, is_climate=False, add_global=False):
    """
    Extract FULL-RESOLUTION statistics for BOTH models.
    No sampling — every valid pixel contributes.

    Returns:
        cascade_data_list: list of arrays (one per region)
        vic_data_list:     list of arrays (one per region)
        labels:            list of region label strings
    """
    cascade_list = []
    vic_list = []
    labels = []

    def _process(mask, label_name):
        count = np.sum(mask)
        if count == 0:
            cascade_list.append(np.array([]))
            vic_list.append(np.array([]))
            labels.append(label_name)
            return

        c_vals = cascade_full[mask]
        v_vals = vic_full[mask]

        # Keep only pixels where BOTH models have valid data
        both_valid = np.isfinite(c_vals) & np.isfinite(v_vals)
        cascade_list.append(c_vals[both_valid])
        vic_list.append(v_vals[both_valid])
        labels.append(label_name)

    # Valid data mask
    valid = np.isfinite(cascade_full) & np.isfinite(vic_full)

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

    return cascade_list, vic_list, labels


# ================================
# STATISTICS EXPORT
# ================================
def save_statistics_to_csv(cascade_list, vic_list, labels, filename):
    """Calculate comprehensive summary statistics and save to CSV."""
    stats_list = []

    for i, label in enumerate(labels):
        for model_name, data in [('Cascade', cascade_list[i]), ('VIC', vic_list[i])]:
            n = len(data) if hasattr(data, '__len__') else 0
            row = {
                'Region': label,
                'Model': CONFIG['models'][model_name]['label'],
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
def plot_global_map(ax, data, extent, cmap, norm, label, model_text):
    """Plot global bias map with embedded colorbar."""
    im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm,
                   alpha=0.85, aspect='auto', origin='upper', interpolation='nearest')

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)

    xticks = np.arange(-150, 151, 60)
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        f'{abs(x)}°W' if x < 0 else f'{x}°E' if x > 0 else '0°'
        for x in xticks
    ])

    yticks = np.arange(-60, 91, 30)
    ax.set_yticks(yticks)
    ax.set_yticklabels([
        f'{abs(y)}°S' if y < 0 else f'{y}°N' if y > 0 else '0°'
        for y in yticks
    ])

    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=CONFIG['figure']['font_size'] + 2, fontweight='bold',
            va='top', ha='left')
    ax.text(0.02, 0.08, model_text, transform=ax.transAxes,
            fontsize=CONFIG['figure']['font_size'], va='bottom', ha='left')

    cbar_ax = inset_axes(ax, width="36%", height="5%", loc='lower center',
                         bbox_to_anchor=(0.11, 0.07, 1, 1), bbox_transform=ax.transAxes)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Bias Ratio', fontsize=CONFIG['figure']['font_size'] - 2)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_label_coords(1.0, 1.5)
    cbar.ax.xaxis.label.set_horizontalalignment('right')

    tick_values = [0.2, 1.0, 2.0, 5.0, norm.vmax]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])
    cbar.ax.tick_params(labelsize=CONFIG['figure']['font_size'] - 3)
    cbar.ax.minorticks_off()


def plot_grouped_boxplot(ax, cascade_data, vic_data, labels, label_text,
                         ylabel=None, show_legend=False, rotation=0, ylim=None):
    """Plot grouped boxplot for two models."""
    n_groups = len(labels)
    indices = np.arange(n_groups)
    width = 0.3

    # Subsample for rendering efficiency
    max_pts = CONFIG['boxplot_subsample']
    rng = np.random.default_rng(42)

    def _sub(arr):
        if hasattr(arr, '__len__') and len(arr) > max_pts:
            idx = rng.choice(len(arr), size=max_pts, replace=False)
            return arr[idx]
        return arr

    cascade_plot = [_sub(d) for d in cascade_data]
    vic_plot = [_sub(d) for d in vic_data]

    bp1 = ax.boxplot(cascade_plot, positions=indices - width / 1.5, widths=width,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(linewidth=1, color='black'),
                     medianprops=dict(color='white', linewidth=1.5),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))

    bp2 = ax.boxplot(vic_plot, positions=indices + width / 1.5, widths=width,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(linewidth=1, color='black'),
                     medianprops=dict(color='white', linewidth=1.5),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))

    for patch in bp1['boxes']:
        patch.set_facecolor(CONFIG['models']['Cascade']['color'])
        patch.set_alpha(0.9)
    for patch in bp2['boxes']:
        patch.set_facecolor(CONFIG['models']['VIC']['color'])
        patch.set_alpha(0.9)

    # Reference line at bias = 1.0 (perfect)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.yaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=CONFIG['figure']['font_size'],
                       rotation=rotation, ha='right' if rotation > 0 else 'center')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=CONFIG['figure']['font_size'])

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    ax.text(0.05, 0.98, label_text, transform=ax.transAxes,
            fontsize=CONFIG['figure']['font_size'] + 2, fontweight='bold', va='top')

    if show_legend:
        legend_elements = [
            Patch(facecolor=CONFIG['models']['Cascade']['color'],
                  label=CONFIG['models']['Cascade']['label'], alpha=0.9),
            Patch(facecolor=CONFIG['models']['VIC']['color'],
                  label=CONFIG['models']['VIC']['label'], alpha=0.9)
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=CONFIG['figure']['font_size'] - 2, frameon=True)


# ================================
# MAIN PIPELINE
# ================================
def create_composite_figure(method='mean'):
    print(f"\n{'=' * 80}")
    print(f"FIGURE GENERATION - {method.upper()} RESAMPLING")
    print(f"  Maps   -> Resampled 25km (for visualization)")
    print(f"  Stats  -> Full-resolution original pixels (for accuracy)")
    print(f"  Panel d -> Continental analysis (with precise SHP masking)")
    print(f"{'=' * 80}")

    start_time = time.time()
    os.makedirs(CONFIG['paths']['output_dir'], exist_ok=True)
    factor = CONFIG['resample_factor']

    # ──────────────────────────────────────────────
    # STEP 1: Load raw data (full resolution)
    # ──────────────────────────────────────────────
    print("\n[1/7] Loading raw data (full resolution)...")
    cascade_data, cascade_transform, cascade_bounds = load_bias_data(CONFIG['paths']['cascade_bias'])
    vic_data, vic_transform, vic_bounds = load_bias_data(CONFIG['paths']['vic_bias'])

    # Full-resolution float arrays for statistics
    cascade_full = cascade_data.filled(np.nan)
    vic_full = vic_data.filled(np.nan)
    print(f"  Full-resolution shape: {cascade_full.shape}  "
          f"({cascade_full.shape[0] * cascade_full.shape[1]:,} pixels)")

    # ──────────────────────────────────────────────
    # STEP 2: Resample for MAP visualization only
    # ──────────────────────────────────────────────
    print(f"\n[2/7] Resampling for map visualization ({method})...")
    cascade_25km, cascade_new_transform = resample_for_map(cascade_data, cascade_transform, factor, method)
    vic_25km, vic_new_transform = resample_for_map(vic_data, vic_transform, factor, method)
    cascade_extent = array_bounds(cascade_25km.shape[0], cascade_25km.shape[1], cascade_new_transform)
    vic_extent = array_bounds(vic_25km.shape[0], vic_25km.shape[1], vic_new_transform)
    print(f"  Resampled shape: {cascade_25km.shape} (for maps only)")

    # Free masked arrays
    del cascade_data, vic_data
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 3: Build masks at FULL resolution
    # ──────────────────────────────────────────────
    print("\n[3/7] Building full-resolution masks...")

    # --- Climate Zone Mask ---
    print("  > Rasterizing climate zones at full resolution...")
    climate_gdf = gpd.read_file(CONFIG['paths']['climate_zones'])
    zone_map = {zone: i + 1 for i, zone in enumerate(CONFIG['climate_labels'])}
    climate_gdf['zone_id'] = climate_gdf['Name'].map(zone_map)
    climate_raster_full = create_raster_mask(
        climate_gdf.geometry, climate_gdf['zone_id'],
        out_shape=cascade_full.shape, transform=cascade_transform
    )
    del climate_gdf
    gc.collect()

    # --- Continent Mask ---
    print("  > Rasterizing continents at full resolution...")
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except Exception:
        world = gpd.read_file("naturalearth_lowres")

    world_filtered, cont_labels_map = build_continent_geodata(world)
    continent_raster_full = create_raster_mask(
        world_filtered.geometry, world_filtered['cont_id'],
        out_shape=cascade_full.shape, transform=cascade_transform
    )
    del world, world_filtered
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 4: Extract FULL-RESOLUTION statistics
    # ──────────────────────────────────────────────
    print("\n[4/7] Extracting full-resolution statistics...")

    # Climate zones (Global + 5 zones)
    print("  > Climate zone statistics (full-resolution)...")
    clim_cascade, clim_vic, clim_lbls = extract_fullres_dual_model_stats(
        cascade_full, vic_full, climate_raster_full,
        CONFIG['climate_labels'], is_climate=True, add_global=True
    )
    for i, lbl in enumerate(clim_lbls):
        n_c = len(clim_cascade[i]) if hasattr(clim_cascade[i], '__len__') else 0
        n_v = len(clim_vic[i]) if hasattr(clim_vic[i], '__len__') else 0
        print(f"    {lbl}: Cascade={n_c:,}  VIC={n_v:,}")

    # Continents (6 continents, no Global)
    print("  > Continental statistics (full-resolution)...")
    cont_cascade, cont_vic, cont_lbls = extract_fullres_dual_model_stats(
        cascade_full, vic_full, continent_raster_full,
        cont_labels_map, is_climate=False, add_global=False
    )
    for i, lbl in enumerate(cont_lbls):
        n_c = len(cont_cascade[i]) if hasattr(cont_cascade[i], '__len__') else 0
        n_v = len(cont_vic[i]) if hasattr(cont_vic[i], '__len__') else 0
        print(f"    {lbl}: Cascade={n_c:,}  VIC={n_v:,}")

    # Free full-resolution rasters
    del cascade_full, vic_full, climate_raster_full, continent_raster_full
    gc.collect()

    # ──────────────────────────────────────────────
    # STEP 5: Export statistics to CSV
    # ──────────────────────────────────────────────
    print("\n[5/7] Exporting statistics to CSV...")
    save_statistics_to_csv(clim_cascade, clim_vic, clim_lbls,
                           f'statistics_climate_fullres_{method}.csv')
    save_statistics_to_csv(cont_cascade, cont_vic, cont_lbls,
                           f'statistics_continent_fullres_{method}.csv')

    # ──────────────────────────────────────────────
    # STEP 6: Compute shared ylim from full-resolution data
    # ──────────────────────────────────────────────
    print("\n[6/7] Computing shared axis limits...")
    shared_ylim = (0, 20)
    print(f"  Shared ylim: {shared_ylim}")

    # ──────────────────────────────────────────────
    # STEP 7: Plotting
    # ──────────────────────────────────────────────
    print("\n[7/7] Plotting 4-panel figure...")
    cm = 1 / 2.54
    map_height = CONFIG['figure']['map_height_cm'] * cm
    box_height = CONFIG['figure']['box_height_cm'] * cm
    fig_width = CONFIG['figure']['map_width_cm'] * cm
    fig_height = 2 * map_height + box_height + 1.5 * cm

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[map_height, map_height, box_height],
                  hspace=0.3, wspace=0.25)

    cmap, norm = create_colormap(vmax=10.0)

    # Panel (a): Cascade map — resampled
    ax_a = fig.add_subplot(gs[0, :])
    plot_global_map(ax_a, cascade_25km,
                    [cascade_extent[0], cascade_extent[2], cascade_extent[1], cascade_extent[3]],
                    cmap, norm, 'a', CONFIG['models']['Cascade']['label'])

    # Panel (b): VIC map — resampled
    ax_b = fig.add_subplot(gs[1, :])
    plot_global_map(ax_b, vic_25km,
                    [vic_extent[0], vic_extent[2], vic_extent[1], vic_extent[3]],
                    cmap, norm, 'b', CONFIG['models']['VIC']['label'])

    # Panel (c): Climate zone boxplot — full-resolution
    ax_c = fig.add_subplot(gs[2, 0])
    plot_grouped_boxplot(ax_c, clim_cascade, clim_vic, clim_lbls,
                         'c', ylabel='Bias Ratio', show_legend=True,
                         rotation=30, ylim=shared_ylim)

    # Panel (d): Continental boxplot — full-resolution
    ax_d = fig.add_subplot(gs[2, 1])
    plot_grouped_boxplot(ax_d, cont_cascade, cont_vic, cont_lbls,
                         'd', ylabel=None, show_legend=False,
                         rotation=0, ylim=shared_ylim)

    # Save
    output_path = os.path.join(CONFIG['paths']['output_dir'],
                               f'composite_bias_continent_{method}.png')
    plt.savefig(output_path, dpi=CONFIG['figure']['dpi'],
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # Free plot data
    del clim_cascade, clim_vic, cont_cascade, cont_vic
    gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"{method.upper()} COMPLETED in {elapsed:.1f}s")
    print(f"  Figure: {output_path}")
    print(f"{'=' * 80}")


def main():
    """Generate figures with all three resampling methods."""
    print("=" * 80)
    print("MULTI-METHOD COMPOSITE FIGURE GENERATION")
    print("  Maps   -> Resampled 25km")
    print("  Stats  -> Full-resolution (all original pixels)")
    print("  Panel d -> Continental analysis (precise SHP masking)")
    print("=" * 80)

    total_start = time.time()

    for method in ['mean', 'median', 'max']:
        create_composite_figure(method)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"ALL METHODS COMPLETED in {total_elapsed:.1f}s")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()