import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict

def plot_times_df(df_data: pd.DataFrame, times_df: pd.DataFrame, *, loglog: bool = True, annotate: bool = True, figsize=(8,6), save_path: str = None, col_name: str = None) -> Dict[str, Dict[str, float]]:
    if times_df.empty:
        raise ValueError("times_df is empty")

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 1.2
    })

    n_total = len(df_data)
    ratios = np.asarray(times_df.index.values, dtype=float)
    sort_idx = np.argsort(ratios)
    ratios = ratios[sort_idx]
    times_df = times_df.iloc[sort_idx]
    n_values = np.maximum((ratios * n_total).astype(int), 1)

    plt.figure(figsize=figsize)
    complexity_info: Dict[str, Dict[str, float]] = {}
    n_min = max(1, n_values.min())
    n_max = max(n_values.max(), n_min + 1)
    n_ref = np.logspace(np.log10(n_min), np.log10(n_max), 200)

    first_n0, first_t0 = None, None
    for col in times_df.columns:
        times = times_df[col].values.astype(float)
        mask = np.isfinite(times) & (times > 0)
        if mask.sum() > 0:
            first_n0, first_t0 = n_values[mask][0], times[mask][0]
            break

    colors = plt.get_cmap('tab10').colors
    for idx, col in enumerate(times_df.columns):
        times = times_df[col].values.astype(float)
        mask = np.isfinite(times) & (times > 0)
        if mask.sum() < 2:
            complexity_info[col] = {'slope': np.nan, 'r2': np.nan}
            continue
        x, y = n_values[mask], times[mask]
        logx, logy = np.log(x), np.log(y)
        slope, intercept = np.polyfit(logx, logy, 1)
        pred = slope * logx + intercept
        ss_res = np.sum((logy - pred) ** 2)
        ss_tot = np.sum((logy - logy.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        complexity_info[col] = {'slope': float(slope), 'r2': float(r2)}
        fitted = np.exp(intercept) * (n_ref ** slope)
        plt.plot(n_ref, fitted, color=colors[idx % 10], linewidth=2.0, label=f"{col} (s={slope:.2f}, R²={r2:.3f})")

    if first_n0 is not None and first_t0 is not None:
        y_n = (first_t0 / first_n0) * n_ref
        y_n2 = (first_t0 / (first_n0 ** 2)) * (n_ref ** 2)
        plt.fill_between(n_ref, y_n*0.85, y_n*1.15, color='green', alpha=0.15, label='O(n) zone')
        plt.fill_between(n_ref, y_n2*0.85, y_n2*1.15, color='red', alpha=0.15, label='O(n²) zone')

    if loglog:
        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel("Input size (n)")
    plt.ylabel("Time (s)")
    title_text = f"Algorithm timings for {col_name}" if col_name else "Algorithm timings (fit vs theory)"
    plt.title(title_text)
    plt.grid(False)
    plt.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return complexity_info
