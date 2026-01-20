#!/usr/bin/env python3
"""CLI runner that executes notebook plotting code and saves figures.

This script loads necessary data files from `data/` and writes figures
to `./figures/` at the requested DPI. It re-uses the plotting code
from notebooks/main.ipynb, adjusted for command-line execution.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

STOP_REQUESTED = False


def setup_logging(level: str | int = "INFO") -> None:
	numeric = level
	if isinstance(level, str):
		numeric = getattr(logging, level.upper(), logging.INFO)
	logging.basicConfig(
		level=numeric,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)


def _signal_handler(signum, frame) -> None:  # pragma: no cover - simple glue
	global STOP_REQUESTED
	logging.getLogger(__name__).info("Received signal %s, requesting stop", signum)
	STOP_REQUESTED = True


def install_signal_handlers() -> None:
	for sig in (signal.SIGINT, signal.SIGTERM):
		signal.signal(sig, _signal_handler)


def save_fig(out_dir: Path, stem: str, dpi: int = 300, ext: str = "png") -> Path:
	out_dir.mkdir(parents=True, exist_ok=True)
	p = out_dir / f"{stem}.{ext}"
	import matplotlib.pyplot as plt

	plt.savefig(p, dpi=dpi, bbox_inches="tight")
	plt.close('all')
	return p


def run_all(output_dir: Path, dpi: int) -> None:
	"""Run the sequence of plotting blocks adapted from the notebook.

	Adjusted paths: `data/` for inputs and `figures/` for outputs.
	"""
	import re
	import glob
	import shutil
	import scipy
	import libpysal
	import itertools
	from esda.moran import Moran
	import pandas as pd
	import geopandas as gpd
	import scipy.stats as stats
	from scipy.stats import entropy, ks_2samp
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import matplotlib as mpl
	import matplotlib.ticker as ticker
	import matplotlib.ticker as mticker
	import matplotlib.patches as mpatches
	import statsmodels.api as sm
	from matplotlib.patches import Rectangle
	import matplotlib.colors as mcolors
	from matplotlib.ticker import FuncFormatter
	from matplotlib.cm import ScalarMappable
	from matplotlib.colors import ListedColormap
	import matplotlib.gridspec as gridspec
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from sklearn.preprocessing import PolynomialFeatures

	# Read data files (paths adjusted for main script location)
	baseline_df = gpd.read_parquet('data/tracts.parquet')
	usa_states = gpd.read_parquet('data/us_states.parquet')
	usa_bounds = gpd.read_parquet('data/usa_boundary.parquet')

	# ------------------------------------------------------------------
	# Spatial distribution + Moran's I
	# ------------------------------------------------------------------
	sns.set_style("white")
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset

	csfont = {'fontname': 'Muli'}
	color_list = ['#ff1b6b', '#ffcb6b', '#b6f3c9', '#45caff']
	cmap = mcolors.LinearSegmentedColormap.from_list("my_color", color_list, N=100)

	baseline_df_3857 = baseline_df.to_crs(epsg=3857)
	usa_states_3857 = usa_states.to_crs(epsg=3857)
	usa_bounds_3857 = usa_bounds.to_crs(epsg=3857)

	target_col = "Composite Index"
	w = libpysal.weights.Queen.from_dataframe(baseline_df)
	w.transform = "r"
	y = baseline_df[target_col].values
	moran = Moran(y, w)

	fig, ax = plt.subplots(figsize=(14, 10))
	usa_bounds_3857.plot(ax=ax, color='#495057', edgecolor='None')
	usa_states_3857.plot(ax=ax, facecolor='None', edgecolor='white', linewidth=0.25, linestyle='dashed')
	data = baseline_df_3857['Composite Index'].values
	vmin = baseline_df_3857['Composite Index'].quantile(0.01)
	vcenter = baseline_df_3857['Composite Index'].quantile(0.5)
	vmax = baseline_df_3857['Composite Index'].quantile(0.99)
	norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
	cs = baseline_df_3857.plot(column='Composite Index', ax=ax, edgecolor='None', cmap=cmap, norm=norm)
	ax.set_axis_off()
	cbar_ax = cs.get_figure().axes[-1]
	cbar_ax.tick_params(labelsize=18)
	cbar_ax.set_xlabel(cbar_ax.get_xlabel(), fontsize=20, labelpad=10, fontweight='bold')

	bins = np.arange(data.min(), data.max(), 0.5)
	hist, edges = np.histogram(data, bins=bins)
	hist_colors = cmap(norm(edges))
	hist_ax = plt.axes([0.2, 0.25, 0.2, 0.1], facecolor='None')
	hist_ax.bar(edges[:-1], hist, width=np.diff(edges), color=hist_colors[:-1], ec='grey', linewidth=0.5, align="edge")
	hist_ax.set_xlim(59, 93)
	hist_ax.set_xlabel('Population in Good Overall Health (%)', fontsize=12, fontweight='bold')
	for spine in ['left', 'bottom']:
		hist_ax.spines[spine].set_linewidth(2)
		hist_ax.spines[spine].set_color('black')
		hist_ax.spines[spine].set_position(('outward', 10))
	hist_ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
	hist_ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
	ax.text(0.98, 0.98, f"Moran's I = {moran.I:.3f}\n" f"p-value (perm.) = {moran.p_sim:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
	plt.yticks(fontsize=12, **csfont)
	plt.xticks(fontsize=12, **csfont)
	save_fig(output_dir, 'spatial_distribution', dpi=dpi)

	# ------------------------------------------------------------------
	# Pairwise scatter/regplots with top colorbar
	# ------------------------------------------------------------------
	plt.rcParams['font.family'] = 'Muli'
	plt.rcParams['axes.labelsize'] = 20
	plt.rcParams['xtick.labelsize'] = 30
	plt.rcParams['ytick.labelsize'] = 30
	colors = ['#e18a85', '#ffffff', '#97c1e2']
	cmap = mpl.colors.LinearSegmentedColormap.from_list("my_color", colors, N=50)
	fig = plt.figure(figsize=(10, 5))
	gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.2, wspace=0.4)
	column_pairs = [("Physical Health", "General Health"), ("Mental Health", "General Health"), ("Mental Health", "Physical Health")]
	from scipy.stats import pearsonr

	def scatter_regplot_with_corr(ax, df, x_col, y_col):
		sc = ax.scatter(df[x_col], df[y_col], c=df[y_col] - df[x_col], s=16, edgecolors='black', linewidth=0.2, cmap=cmap)
		sns.regplot(x=x_col, y=y_col, data=df, ci=99, scatter=False, color="black", line_kws={"linewidth": 0.1}, ax=ax)
		r, p = pearsonr(df[x_col], df[y_col])
		ax.text(0.95, 0.2, f"r = {r:.3f} \n p < 0.001", transform=ax.transAxes, fontsize=16, fontfamily='Muli', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle='round'))
		x_min, x_max = df[x_col].min(), df[x_col].max()
		y_min, y_max = df[y_col].min(), df[y_col].max()
		x_pad = (x_max - x_min) * 0.1
		y_pad = (y_max - y_min) * 0.1
		ax.set_xlim(x_min - x_pad, x_max + x_pad)
		ax.set_ylim(y_min - y_pad, y_max + y_pad)
		ax.set_xlabel(x_col, fontfamily='Muli', fontsize=24, labelpad=10, fontweight='bold')
		ax.set_ylabel(y_col, fontfamily='Muli', fontsize=24, labelpad=10, fontweight='bold')
		ax.tick_params(axis='both', labelfontfamily='Muli', labelsize=20)
		ax.grid(False)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("top", size="7%", pad=0.35)
		cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
		cb.ax.xaxis.set_ticks_position('top')
		cb.ax.xaxis.set_label_position('top')
		cb.ax.set_xlabel(f"{y_col} - {x_col}", labelpad=8, fontsize=16)
		cb.ax.tick_params(labelsize=14)

	for i, (x_col, y_col) in enumerate(column_pairs):
		if i == 2:
			continue
		ax = fig.add_subplot(gs[0, i])
		scatter_regplot_with_corr(ax, baseline_df, x_col, y_col)
	plt.tight_layout()
	save_fig(output_dir, 'linear_health_associations_top_cbar', dpi=dpi)

	# ------------------------------------------------------------------
	# KDE distributions and KL/KS annotations
	# ------------------------------------------------------------------
	plt.rcParams['font.family'] = 'Muli'
	plt.rcParams.update({'legend.fontsize': 22, 'legend.handlelength': 1.5})
	target_cols = ['General Health', 'Physical Health', 'Mental Health']
	colors = ['#e18a85', '#f7cd9b', '#97c1e2']
	sns.set_style("ticks")
	fig, ax = plt.subplots(figsize=(18, 12))
	max_y = 0.0
	densities = {}
	for i in range(3):
		series = pd.to_numeric(baseline_df[target_cols[i]], errors='coerce').dropna()
		if series.nunique() < 2:
			continue
		x_vals = series.values
		mean_value = np.mean(x_vals)
		std = np.std(x_vals)
		kde_ax = sns.kdeplot(x=x_vals, ax=ax, color=colors[i], bw_adjust=4, clip=(0, 100), lw=3, alpha=0.9)
		line = kde_ax.lines[-1]
		x = line.get_xdata()
		y = line.get_ydata()
		ymax = y.max()
		max_y = max(max_y, ymax)
		grid = np.linspace(40, 100, 500)
		y_interp = np.interp(grid, x, y)
		y_norm = y_interp / np.trapz(y_interp, grid)
		densities[target_cols[i]] = y_norm
		ax.fill_between(x, y, 0, color=colors[i], alpha=0.18)
		ax.fill_between(x, y, where=(x <= mean_value - std), color=colors[i], alpha=0.3, hatch='////', edgecolor='white')
		ax.fill_between(x, y, where=(x >= mean_value + std), color=colors[i], alpha=0.3, hatch='////', edgecolor='white')
		ax.vlines(mean_value, ymin=0, ymax=ymax * 1.08, color=colors[i], linestyle='--', linewidth=1.5)
		ax.annotate("", xy=(mean_value - std, ymax * 1.08), xytext=(mean_value + std, ymax * 1.08), arrowprops=dict(arrowstyle="-", color=colors[i], linewidth=1.5))
		ax.plot([mean_value - std, mean_value - std], [ymax * 1.06, ymax * 1.10], color=colors[i], linewidth=1.2)
		ax.plot([mean_value + std, mean_value + std], [ymax * 1.06, ymax * 1.10], color=colors[i], linewidth=1.2)
		ax.text(mean_value, ymax * 1.12, f'μ={mean_value:.2f}', color=colors[i], fontsize=32, ha='center')
	pairs = [("General Health", "Physical Health"), ("General Health", "Mental Health"), ("Physical Health", "Mental Health")]
	annot_text = ""
	for a, b in pairs:
		stat, p = ks_2samp(baseline_df[a].dropna(), baseline_df[b].dropna())
		annot_text += (f"{a.split()[0]} || {b.split()[0]}: KS = {stat:.3f}, p < 0.001 \n")
	ax.text(0.08, 0.8, annot_text[:-2], transform=ax.transAxes, fontsize=24, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle='round'))
	ax.set_xlim(40, 100)
	ax.set_ylim(top=max_y * 1.35)
	ax.set_ylabel('Density', fontsize=50, labelpad=50, fontweight='bold')
	ax.set_xlabel('Population in Good Health (%)', fontsize=50, labelpad=50, fontweight='bold')
	ax.tick_params(axis='x', labelsize=36, width=2)
	ax.tick_params(axis='y', labelsize=36, width=2)
	sns.despine(ax=ax, offset=10)
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_linewidth(2)
	save_fig(output_dir, 'health_distribution_kde', dpi=dpi)

	# ------------------------------------------------------------------
	# Circular feature importance plots
	# ------------------------------------------------------------------
	general_df_sorted = pd.read_parquet('data/general_feature_importance.parquet')
	physical_df_sorted = pd.read_parquet('data/physical_feature_importance.parquet')
	mental_df_sorted = pd.read_parquet('data/mental_feature_importance.parquet')
	target_cols = ['General Health', 'Physical Health', 'Mental Health']
	all_df_sorted = [general_df_sorted, physical_df_sorted, mental_df_sorted]
	
	def get_label_rotation(angle, offset):
		rotation = np.rad2deg(angle + offset)
		if angle <= np.pi:
			alignment = "right"
			rotation = rotation + 180
		else:
			alignment = "left"
		return rotation, alignment

	def add_labels(angles, values, labels, offset, ax, to_bold):
		padding = 10
		for angle, value, label in zip(angles, values, labels):
			rotation, alignment = get_label_rotation(angle, offset)
			if label in to_bold:
				ax.text(x=angle, y=value + padding, s=label, ha=alignment, va="center", fontsize=30, fontfamily='Muli', fontweight='black', rotation=rotation, rotation_mode="anchor")
			else:
				ax.text(x=angle, y=value + padding, s=label, ha=alignment, va="center", fontsize=30, fontfamily='Muli', rotation=rotation, rotation_mode="anchor")
				
    
	for target_col, df_sorted in zip(target_cols, all_df_sorted):
		ANGLES = np.linspace(0, 2 * np.pi, len(df_sorted), endpoint=False)
		VALUES = df_sorted["importance"].values * 100
		LABELS = df_sorted["variable"].values
		OFFSET = np.pi / 2.3
		GROUP = df_sorted["Category"].values
		PAD = 4
		ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
		ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
		WIDTH = (2 * np.pi) / len(ANGLES) *0.8
		GROUPS_SIZE = [len(i[1]) for i in df_sorted.groupby("Category")]
		offset = 0
		IDXS = []
		for size in GROUPS_SIZE:
			IDXS += list(range(offset + PAD, offset + size + PAD))
			offset += size + PAD
		fig, ax = plt.subplots(figsize=(20, 13), subplot_kw={"projection": "polar"})
		ax.set_theta_offset(OFFSET)
		ax.set_ylim(-150, 100)
		ax.set_frame_on(False)
		ax.xaxis.grid(False)
		ax.yaxis.grid(False)
		ax.set_xticks([])
		ax.set_yticks([])
		GROUPS_SIZE = [len(i[1]) for i in df_sorted.groupby("Category")]
		color_chart = ['#005f60', '#008083', '#249ea0', '#faab36', '#f78104', '#fd5901']
		cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", color_chart, N=100)
		norm = mpl.colors.Normalize(vmin=VALUES.min(), vmax=VALUES.max())
		COLORS = cmap(norm(VALUES))
		ax.bar(
            ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, 
            edgecolor="white", linewidth=1
        )
		to_bold = (
        df_sorted.sort_values("importance", ascending=False)
        .head(20)["variable"].values
        )
		add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax, to_bold)
		

		scalar_map = ScalarMappable(norm=norm, cmap=cmap)
		scalar_map.set_array([])  # Required for some older versions of Matplotlib
		offset = 0 
		for group, size in zip(df_sorted['Category'].unique(), GROUPS_SIZE):
			x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=100)
			ax.plot(x1, [-5] * 100, color="#333333")
			ax.text(np.pi*2.055, 21.5, 20, ha="center", size=26, fontfamily='Muli', fontweight='light')
			ax.text(np.pi*2.055, 41.5, 40, ha="center", size=26, fontfamily='Muli', fontweight='light')
			ax.text(np.pi*2.055, 61.5, 60, ha="center", size=26, fontfamily='Muli', fontweight='light')
			ax.text(np.pi*2.055, 81.5, 80, ha="center", size=26, fontfamily='Muli', fontweight='light')
			ax.text(np.pi*2.055, 101.5, 100, ha="center", size=26, fontfamily='Muli', fontweight='light')
			x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
			ax.plot(x2, [20] * 50, color="#bebebe", lw=2)
			ax.plot(x2, [40] * 50, color="#bebebe", lw=2)
			ax.plot(x2, [60] * 50, color="#bebebe", lw=2)
			ax.plot(x2, [80] * 50, color="#bebebe", lw=2)
			ax.plot(x2, [100] * 50, color="#bebebe", lw=2)
			ax.text(np.pi*2.045, -150, target_col, ha="center", size=28, fontfamily='Muli', fontweight='bold')
			offset += size + PAD

		save_fig(output_dir, f'{target_col.split()[0].lower()}_feature_importance_circular', dpi=dpi)


	# ------------------------------------------------------------------
	# Multiplier plots and per-domain multipliers
	# ------------------------------------------------------------------
	general_perturb_sorted = pd.read_parquet('data/general_feature_perturb.parquet')
	physical_df_sorted = pd.read_parquet('data/physical_feature_perturb.parquet')
	mental_df_sorted = pd.read_parquet('data/mental_feature_perturb.parquet')
	all_df_perturb = [general_perturb_sorted, physical_df_sorted, mental_df_sorted]
	services_columns = ['POI Accessibility', 'Cultural Institutions', 'Groceries', 'Parks', 'Religious Organizations', 'Restaurants', 'Schools', 'Services', 'Drugstores', 'Healthcare', 'Median Income']

	for target_col, df_perturb in zip(target_cols, all_df_perturb):
		domain = target_col.split()[0]
		fig = plt.figure(figsize=(15, 60))
		gs = gridspec.GridSpec(10, 1, hspace=0.1)
		multiplier_dict = {}
		for i, y_col in enumerate(services_columns[:-1]):
			ax = fig.add_subplot(gs[i, 0])
			X_temp = df_perturb[['Median Income', y_col]].dropna()
			X = X_temp['Median Income'].values.reshape(-1, 1)
			y = X_temp[y_col].values
			polynomial_features = PolynomialFeatures(degree=3)
			X_poly = polynomial_features.fit_transform(X)
			model = sm.OLS(y, X_poly).fit()
			x_range = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
			x_range_poly = polynomial_features.transform(x_range)
			predictions = model.get_prediction(x_range_poly)
			pred_summary_90 = predictions.summary_frame(alpha=0.1)
			pred_summary_50 = predictions.summary_frame(alpha=0.5)
			predicted = pred_summary_90['mean']
			lower_ci_90 = pred_summary_90['mean_ci_lower']
			upper_ci_90 = pred_summary_90['mean_ci_upper']
			lower_ci_50 = pred_summary_50['mean_ci_lower']
			upper_ci_50 = pred_summary_50['mean_ci_upper']
			sns.lineplot(ax=ax, x=x_range.flatten(), y=predicted, color='black', linewidth=2)
			colors_map = {'General': '#e18a85', 'Physical': '#f7cd9b', 'Mental': '#97c1e2'}
			ax.fill_between(x_range.flatten(), upper_ci_90, lower_ci_90, color=colors_map[domain], alpha=0.3, label='90% CI')
			ax.fill_between(x_range.flatten(), upper_ci_50, lower_ci_50, color=colors_map[domain], alpha=0.5, label='50% CI')
			if i == len(services_columns[:-1]) - 1:
				ax.set_xlabel('Median Income ($USD 1000)', fontsize=60, labelpad=20, fontfamily='Muli', fontweight='bold')
				sns.rugplot(ax=ax, x=X_temp['Median Income'].sample(frac=0.1), height=0.06, color='black', alpha=0.2)
				ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x/1000)))
			else:
				ax.set_xticks([])
				ax.set_xlabel('')
			ax.set_ylabel(y_col.replace(' ', '\n'), fontfamily='Muli', fontsize=50, fontweight='bold', labelpad=15)
			ax.set_xlim(X.min(), X.max())
			ax.tick_params(axis='both', labelsize=40)
			ax.grid(False)
			ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
			x_10 = np.percentile(X, 10)
			x_90 = np.percentile(X, 90)
			x_10_poly = polynomial_features.transform([[x_10]])
			x_90_poly = polynomial_features.transform([[x_90]])
			pred_10 = model.get_prediction(x_10_poly).summary_frame(alpha=0.1)['mean'].iloc[0]
			pred_90 = model.get_prediction(x_90_poly).summary_frame(alpha=0.1)['mean'].iloc[0]
			multiplier = (pred_10 - pred_90) / abs(pred_90) + 1 if pred_90 != 0 else np.nan
			multiplier_dict[y_col] = {'multiplier': multiplier}
			ax.text(0.05, 0.95, f"Multiplier: {multiplier:.2f}", transform=ax.transAxes, fontsize=50, fontfamily='Muli', verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle='round'))
		fig.align_ylabels(fig.axes)
		plt.tight_layout()
		save_fig(output_dir, f'{domain.lower()}_multiplier', dpi=dpi)

	# ------------------------------------------------------------------
	# State multiplier scatter (per-domain)
	# ------------------------------------------------------------------
	general_mult = pd.read_parquet('data/general_multiplier_per_capita.parquet')
	physical_mult = pd.read_parquet('data/physical_multiplier_per_capita.parquet')
	mental_mult = pd.read_parquet('data/mental_multiplier_per_capita.parquet')
	all_df_mult = [general_mult, physical_mult, mental_mult]
	region_markers = {'Northeast': 'o', 'Midwest': 's', 'South': 'D', 'West': '^'}
	for target_col, df_mult in zip(target_cols, all_df_mult):
		from adjustText import adjust_text
		domain = target_col.split()[0]
		pastel_color_palette = ['#e18a85','#f7cd9b', '#97c1e2', '#9cd3bf']
		plt.rcdefaults()
		fig, ax = plt.subplots(figsize=(12, 13))
		x = 'Income per capita'
		y = 'Geometric_Multiplier_Mean'
		for i, (region, marker) in enumerate(region_markers.items()):
			region_data = df_mult[df_mult['Region'] == region]
			ax.scatter(region_data[x], region_data[y], label=region, c=pastel_color_palette[i], cmap=cmap, s=150, marker=marker, edgecolor='black', linewidth=0.5, alpha=1, zorder=3)
		texts = []
		for i, row in df_mult.iterrows():
			texts.append(ax.text(row[x]+0.2, row[y], str(row['STATE_NAME']), fontsize=30, zorder=5))
		adjust_text(texts, ax=ax, expand_points=(5,5), expand_text=(5,5), arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_mult[x], df_mult[y])
		x_vals = np.linspace(0, 100, 100)
		y_vals = intercept + slope * x_vals
		ax.plot(x_vals, y_vals, color='black', linestyle=':', linewidth=1, label='OLS Regression', zorder=-1)
		equation_text = (f"y = {intercept:.2f} + {slope:.2f}x\n" f"$R^2 = {r_value**2:.3f}$, P-value {'< 0.001' if p_value.round(3) < 0.001 else p_value.round(3)}")
		ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=36, color='black', fontfamily='Muli', verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle='round'))
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.tick_params(axis='both', which='major', length=8, width=1.5, direction='out')
		ax.set_ylim(0.5, 4.5)
		ax.set_xlim(21, 38)
		plt.xticks(fontsize=40, fontname="Muli", fontweight='regular')
		plt.yticks(fontsize=40, fontname="Muli", fontweight='regular')
		plt.xlabel('Per Capita Income ($USD 1000)', fontsize=44, labelpad=20, fontweight='bold', fontname="Muli")
		plt.ylabel(f"{domain} Health Multiplier $_{{{y}}}$", fontsize=44, labelpad=20, fontweight='bold', fontname="Muli")
		plt.tight_layout()
		save_fig(output_dir, f'{domain.lower()}_scatter_income_multiplier', dpi=dpi)

	# ------------------------------------------------------------------
	# Association/motif heatmap
	# ------------------------------------------------------------------
	association_df = pd.read_parquet('data/motifs_association.parquet')
	df = association_df
	for junk in ["Unnamed: 0", "index"]:
		if junk in df.columns:
			del df[junk]
	if "Column" not in df.columns:
		logging.getLogger(__name__).warning("No 'Column' field in motifs association dataframe; skipping heatmap.")
	else:
		pat = re.compile(r"^([A-Ea-e])-(\d+)-(\d+)$")
		pattern_cols = [c for c in df.columns if pat.match(str(c)) and pat.match(c).group(1).upper() in list("ABCDE")]
		if pattern_cols:
			def sort_key(p):
				m = pat.match(p)
				L = m.group(1).upper()
				n1, n2 = int(m.group(2)), int(m.group(3))
				return (list("ABCDE").index(L), n1, n2)
			pattern_cols = sorted(pattern_cols, key=sort_key)
			mat = df.set_index("Column")[pattern_cols].copy()
			nrows, ncols = mat.shape
			fig, ax = plt.subplots(figsize=(max(12, min(28, 0.35 * ncols + 8)), 5), dpi=300)
			cmap = mpl.colors.LinearSegmentedColormap.from_list("custom3", ['#4b9ad7','#97c1e2','#c1d3e0','#ffffff','#dec2c1','#e18a85','#de514a'])
			vmin, vmax = np.nanmin(mat.values), np.nanmax(mat.values)
			im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
			ax.set_xticks(np.arange(ncols))
			ax.set_yticks(np.arange(nrows))
			ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
			ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
			ax.grid(which="minor", color="white", linewidth=6)
			ax.tick_params(which="minor", length=0)
			ax.set_xticklabels(mat.columns, fontsize=20, rotation=60, ha="center")
			ax.set_yticklabels(list(mat.index), fontsize=20)
			plt.subplots_adjust(left=0.27, right=0.98, top=0.92, bottom=0.18)
			letters = [pat.match(p).group(1).upper() for p in mat.columns]
			spans = []
			s = 0
			for i in range(1, len(letters)):
				if letters[i] != letters[i-1]:
					spans.append((letters[i-1], s, i))
					s = i
			spans.append((letters[-1], s, len(letters)))
			for L, s, e in spans:
				label = {"A":"Balanced Living","C":"Suburban Settlement","E":"Dense Urban"}.get(L, L)
				ax.text((s + e - 1) / 2, -1.2, label, ha="center", va="center", fontsize=22, fontweight="bold", color="#343a40")
				if e < ncols:
					ax.vlines(e - 0.5, -0.5, nrows - 0.5, colors="black", linewidth=1.0)
			cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.005)
			cbar.set_label("Pearson Correlation", rotation=90, labelpad=12, fontsize=18)
			cbar.set_ticks(np.linspace(vmin, vmax, 6))
			save_fig(output_dir, 'motifs_association', dpi=dpi)

	# ------------------------------------------------------------------
	# Motif entropy / line plots
	# ------------------------------------------------------------------
	general_line_df = pd.read_parquet('data/general_line_df.parquet')
	physical_line_df = pd.read_parquet('data/physical_line_df.parquet')
	mental_line_df = pd.read_parquet('data/mental_line_df.parquet')
	fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
	fig.subplots_adjust(hspace=0)
	sns.set_theme(style="ticks")
	colors = ['#e18a85', '#f7cd9b', '#97c1e2']
	ax = axes[0]
	y = 'Health Entropy'
	x = 'Health Decile'
	sns.scatterplot(data=general_line_df, x=general_line_df[x], y=general_line_df[y], color=colors[0], marker="o", facecolor='white', ec=colors[0], linewidth=1.5, s=100, zorder=3, ax=ax)
	sns.scatterplot(data=physical_line_df, x=physical_line_df[x], y=physical_line_df[y], color=colors[1], marker="o", facecolor='white', ec=colors[1], linewidth=1.5, s=100, zorder=3, ax=ax)
	sns.scatterplot(data=mental_line_df, x=mental_line_df[x], y=mental_line_df[y], color=colors[2], marker="o", facecolor='white', ec=colors[2], linewidth=1.5, s=100, zorder=3, ax=ax)
	sns.lineplot(x=general_line_df[x], y=general_line_df[y], color=colors[0], linewidth=1.5, linestyle='dashdot', label='General Health', ax=ax)
	sns.lineplot(x=physical_line_df[x], y=physical_line_df[y], color=colors[1], linewidth=1.5, linestyle='dashdot', label='Physical Health', ax=ax)
	sns.lineplot(x=mental_line_df[x], y=mental_line_df[y], color=colors[2], linewidth=1.5, linestyle='dashdot', label='Mental Health', ax=ax)
	ax.set_xlim(0, 11)
	ax.set_xticks(range(1, 11))
	ax.set_xticklabels([f'D{i}' for i in range(1, 11)], fontsize=32, fontname="Muli")
	ax.set_xlabel("Health Deciles", fontname="Muli", fontsize=36, fontweight="bold", labelpad=15)
	ax.set_ylabel("Motif Entropy", fontname="Muli", fontsize=36, fontweight="bold", labelpad=15)
	ax.legend(loc='upper center', fontsize=20, ncol=3)
	ax.tick_params(axis='both', labelsize=32, width=3)
	for spine in ax.spines.values():
		spine.set_linewidth(3)
	ax.tick_params(axis='both', labelsize=32)
	for x_value in range(1, 11):
		ax.axvline(x=x_value, color='lightgrey', linestyle='--', linewidth=1.0, alpha=0.7, zorder=-2)
	ax = axes[1]
	target_x = 'Proportion of persons below 150 percent poverty threshold'
	y = f'{target_x} Entropy'
	x = f'{target_x} decile'
	sns.scatterplot(data=general_line_df, x=general_line_df[x], y=general_line_df[y], color=colors[0], marker="o", facecolor='white', ec=colors[0], linewidth=1.5, s=100, zorder=3, ax=ax)
	sns.scatterplot(data=physical_line_df, x=physical_line_df[x], y=physical_line_df[y], color=colors[1], marker="o", facecolor='white', ec=colors[1], linewidth=1.5, s=100, zorder=3, ax=ax)
	sns.scatterplot(data=mental_line_df, x=mental_line_df[x], y=mental_line_df[y], color=colors[2], marker="o", facecolor='white', ec=colors[2], linewidth=1.5, s=100, zorder=3, ax=ax)
	sns.lineplot(x=general_line_df[x], y=general_line_df[y], color=colors[0], linewidth=1.5, linestyle='dashdot', label='General Health', ax=ax)
	sns.lineplot(x=physical_line_df[x], y=physical_line_df[y], color=colors[1], linewidth=1.5, linestyle='dashdot', label='Physical Health', ax=ax)
	sns.lineplot(x=mental_line_df[x], y=mental_line_df[y], color=colors[2], linewidth=1.5, linestyle='dashdot', label='Mental Health', ax=ax)
	ax.set_xlim(0, 11)
	ax.set_xticks(range(1, 11))
	ax.set_xticklabels([f'D{i}' for i in range(1, 11)], fontsize=32, fontname="Muli")
	ax.set_xlabel("Poverty Rate Deciles", fontname="Muli", fontsize=36, fontweight="bold", labelpad=15)
	ax.tick_params(axis='both', labelsize=32, width=3)
	for spine in ax.spines.values():
		spine.set_linewidth(3)
	ax.legend(loc='upper center', fontsize=20, ncol=3)
	for x_value in range(1, 11):
		ax.axvline(x=x_value, color='lightgrey', linestyle='--', linewidth=1.0, alpha=0.7, zorder=-2)
	ax.set_ylim(1.75,2.5)
	sns.despine(offset=5, trim=True)
	plt.tight_layout()
	save_fig(output_dir, 'motif_entropy_pair', dpi=dpi)

	# ------------------------------------------------------------------
	# Q-Q plot for entropy
	# ------------------------------------------------------------------
	mpl.rc('font', family='Muli')
	general_diversity_df = pd.read_parquet('data/general_diversity_df.parquet')
	entropy_vals = general_diversity_df['Entropy'].dropna()
	shapiro_stat, shapiro_p = stats.shapiro(entropy_vals)
	ks_stat, ks_p = stats.kstest((entropy_vals - entropy_vals.mean()) / entropy_vals.std(ddof=0), 'norm')
	plt.figure(figsize=(16, 12))
	(osm, osr), (slope, intercept, r) = stats.probplot(entropy_vals, dist="norm")
	plt.scatter(osm, osr, facecolor='white', edgecolor='black', s=20, alpha=0.5, label="Entropy")
	plt.plot(osm, intercept + slope*osm, color='grey', linestyle='--', linewidth=1, label="Fit line")
	plt.xlabel("Theoretical Quantiles (Normal Distribution)", fontname='Muli', fontsize=44, fontweight='bold', labelpad=20)
	plt.ylabel("Sample Quantiles (Entropy)", fontsize=44, fontweight='bold', labelpad=20)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.text(0.05, 0.95, f"Shapiro-Wilk Normality\nW={shapiro_stat:.3f}, p < 0.001", transform=plt.gca().transAxes, fontsize=30, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.8))
	plt.xticks(fontsize=36, fontname="Muli")
	plt.yticks(fontsize=36, fontname="Muli")
	save_fig(output_dir, 'qq_normality_entropy', dpi=dpi)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Generate all figures and save to ./figures")
	p.add_argument("--output-dir", "-o", default="figures", help="Directory to write figures")
	p.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
	p.add_argument("--log-level", default="INFO", help="Logging level")
	p.add_argument("--no-run", action='store_true', help="Don't run plots (sanity-check) ")
	return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	setup_logging(args.log_level)
	install_signal_handlers()
	out_dir = Path(args.output_dir)
	if args.no_run:
		logging.getLogger(__name__).info("No-run requested, exiting")
		return 0
	try:
		run_all(out_dir, args.dpi)
		logging.getLogger(__name__).info("All figures created in %s", out_dir)
		return 0
	except Exception:
		logging.getLogger(__name__).exception("Error while generating figures")
		return 2


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except KeyboardInterrupt:
		raise SystemExit(130)

