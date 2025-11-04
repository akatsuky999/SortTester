import os
import json
from typing import List, Optional

import numpy as np
import pandas as pd

from sort_tester.core import SortTester
from sort_tester.algorithms import builtin_algorithms
from sort_tester.plotting import plot_times_df

DEFAULT_COLUMNS = [
    "VehicleType",
    "DetectionTime_O",
    "GantryID_O",
    "DetectionTime_D",
    "GantryID_D",
    "TripLength",
    "TripEnd",
    "TripInformation",
]

def load_and_prepare(path: str, expected_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if expected_cols is None:
        expected_cols = DEFAULT_COLUMNS

    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        path = os.path.join(project_root, path)
        path = os.path.normpath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    if len(df.columns) == len(expected_cols):
        df.columns = expected_cols
    else:
        print(f"Warning: CSV has {len(df.columns)} columns, expected {len(expected_cols)}; keeping original names.")

    return df

def parse_ratios(ratios=None, ratmin=0.01, ratmax=1.0, nrat=6) -> List[float]:
    if ratios:
        ratios_list = [float(x) for x in ratios.split(",") if x.strip()]
    else:
        ratios_list = np.linspace(float(ratmin), float(ratmax), int(nrat)).tolist()
    for r in ratios_list:
        if not (0 < r <= 1):
            raise ValueError("ratios must be in (0,1]")
    return sorted(set(ratios_list))

def choose_algorithms(arg_algos: Optional[str]) -> dict:
    if arg_algos is None:
        chosen = {name: func for name, func in builtin_algorithms.items()}
    else:
        names = [s.strip() for s in arg_algos.split(",") if s.strip()]
        chosen = {}
        for n in names:
            if n not in builtin_algorithms:
                raise ValueError(f"Algorithm '{n}' not found. Available: {', '.join(sorted(builtin_algorithms.keys()))}")
            chosen[n] = builtin_algorithms[n]
    return chosen

def ensure_output_dir(path: Optional[str]):
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def run_benchmark(
    csv_path: str,
    col_name: str,
    algos: dict,
    ratios: List[float],
    repeat: int = 3,
    sequential: bool = False,
    save_csv: Optional[str] = None,
    save_json: Optional[str] = None,
    save_plot: Optional[str] = None,
):
    df = load_and_prepare(csv_path)
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in CSV. Available columns: {df.columns.tolist()}")
    series = df[col_name].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        series = pd.to_numeric(series, errors="coerce").dropna()
    cleaned = df.loc[series.index].copy()
    cleaned[col_name] = series.values
    tester = SortTester(cleaned, col_name, random_seed=42)
    ratios_dict = {name: ratios for name in algos.keys()}
    print(f"Running benchmark on column '{col_name}' with {len(cleaned)} rows, {cleaned[col_name].nunique()} unique values")
    print(f"Algorithms: {list(algos.keys())}")
    print(f"Ratios: {ratios}")
    print(f"Repeat: {repeat}, Sequential sampling: {sequential}")
    times_df = tester.run_algorithms(algos, ratios_dict, repeat=repeat, sequential=sequential, check_sorted=True)
    times_df.index.name = "ratio"
    print("\nTiming table (mean times in seconds):")
    print(times_df)
    complexities = tester.complexity_from_df(times_df)
    print("\nComplexity fits (slope, R²):")
    print(json.dumps(complexities, indent=2))
    if save_csv:
        ensure_output_dir(save_csv)
        times_df.to_csv(save_csv)
        print(f"Saved timing table to: {save_csv}")
    if save_json:
        ensure_output_dir(save_json)
        with open(save_json, "w", encoding="utf-8") as fh:
            json.dump({"timings": times_df.to_dict(), "complexities": complexities}, fh, indent=2)
        print(f"Saved JSON report to: {save_json}")
    comp_info = plot_times_df(
        tester.data, times_df, loglog=True, annotate=True, figsize=(8, 6), save_path=save_plot, col_name=col_name
    )
    if save_plot:
        print(f"Saved plot to: {save_plot}")
    return times_df, complexities, comp_info

if __name__ == "__main__":
    CSV_PATH = "data/TDCS_M06A_20231204_080000.csv"
    COL_NAME = "VehicleType"
    ALGOS = None
    RATIOS = None
    RATMIN = 0.01
    RATMAX = 0.1
    NRAT = 5
    REPEAT = 10
    SEQUENTIAL = False
    SAVE_CSV = None
    SAVE_JSON = None
    SAVE_PLOT = None
    ratios_list = parse_ratios(RATIOS, RATMIN, RATMAX, NRAT)
    algos_dict = choose_algorithms(ALGOS)
    run_benchmark(
        csv_path=CSV_PATH,
        col_name=COL_NAME,
        algos=algos_dict,
        ratios=ratios_list,
        repeat=REPEAT,
        sequential=SEQUENTIAL,
        save_csv=SAVE_CSV,
        save_json=SAVE_JSON,
        save_plot=SAVE_PLOT,
    )
