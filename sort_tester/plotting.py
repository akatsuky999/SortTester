import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict

def plot_times_df(df_data: pd.DataFrame, times_df: pd.DataFrame, *, loglog: bool = True, annotate: bool = True, figsize=(8,6), save_path: str = None, col_name: str = None) -> Dict[str, Dict[str, float]]:
    if times_df.empty:
        raise ValueError("times_df is empty")

    n_total = len(df_data)
    ratios = np.asarray(times_df.index.values, dtype=float)
    sort_idx = np.argsort(ratios)
    ratios = ratios[sort_idx]
    times_df = times_df.iloc[sort_idx]
    n_values = np.maximum((ratios * n_total).astype(int), 1)

    plt.figure(figsize=figsize)
    complexity_info: Dict[str, Dict[str, float]] = {}
    n_min = max(1, n_values.min()); n_max = max(n_values.max(), n_min+1)
    n_ref = np.logspace(np.log10(n_min), np.log10(n_max), 200)

    first_n0 = None; first_t0 = None
    for col in times_df.columns:
        times = times_df[col].values.astype(float)
        mask = np.isfinite(times) & (times > 0)
        if mask.sum() > 0:
            first_n0 = n_values[mask][0]; first_t0 = times[mask][0]; break

    for col in times_df.columns:
        times = times_df[col].values.astype(float)
        mask = np.isfinite(times) & (times > 0)
        if mask.sum() < 2:
            complexity_info[col] = {'slope': np.nan, 'r2': np.nan}
            continue
        x = n_values[mask]; y = times[mask]
        logx = np.log(x); logy = np.log(y)
        slope, intercept = np.polyfit(logx, logy, 1)
        pred = slope * logx + intercept
        ss_res = np.sum((logy - pred) ** 2)
        ss_tot = np.sum((logy - logy.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        complexity_info[col] = {'slope': float(slope), 'r2': float(r2)}
        fitted = np.exp(intercept) * (n_ref ** slope)
        if loglog:
            plt.loglog(n_ref, fitted, linestyle='-', linewidth=1.8, label=f"{col} (s={slope:.2f}, R²={r2:.3f})")
        else:
            plt.plot(n_ref, fitted, linestyle='-', linewidth=1.8, label=f"{col} (s={slope:.2f}, R²={r2:.3f})")

    if first_n0 is not None:
        ref_on = (first_t0 / first_n0) * n_ref
        ref_on2 = (first_t0 / (first_n0 ** 2)) * (n_ref ** 2)
        if loglog:
            plt.loglog(n_ref, ref_on, linestyle='--', linewidth=1.2, label="O(n)")
            plt.loglog(n_ref, ref_on2, linestyle='--', linewidth=1.2, label="O(n²)")
        else:
            plt.plot(n_ref, ref_on, linestyle='--', linewidth=1.2, label="O(n)")
            plt.plot(n_ref, ref_on2, linestyle='--', linewidth=1.2, label="O(n²)")

    plt.xlabel("Input size (n)", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    title_text = f"Algorithm timings for {col_name}" if col_name else "Algorithm timings (fit vs theory)"
    plt.title(title_text, fontsize=14)
    plt.legend(fontsize=9, loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    return complexity_info
