import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_times_df(df_data: pd.DataFrame, times_df: pd.DataFrame, loglog: bool = True):
    n_total = len(df_data)
    ratios = times_df.index.values
    n_values = (ratios * n_total).astype(int)
    plt.figure(figsize=(7, 5))
    for col in times_df.columns:
        times = times_df[col].values.astype(float)
        if loglog:
            plt.loglog(n_values, times, marker='o', label=col)
        else:
            plt.plot(n_values, times, marker='o', label=col)
    plt.xlabel('Input size (n)')
    plt.ylabel('Time (s)')
    plt.title('Algorithm timings')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
