import os
import json
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sort_tester.core import SortTester
from sort_tester.algorithms import builtin_algorithms
from sort_tester.plotting import plot_times_df

class BenchmarkRunner:
    def __init__(
        self,
        csv_path: str,
        col_name: str,
        algos: Optional[str] = None,
        ratios: Optional[str] = None,
        ratmin: float = 0.01,
        ratmax: float = 1.0,
        nrat: int = 6,
        repeat: int = 3,
        sequential: bool = False,
        save_csv: Optional[str] = None,
        save_json: Optional[str] = None,
        save_plot: Optional[str] = None,
        random_seed: int = 42,
        expected_cols: Optional[list] = None
    ):
        self.csv_path = csv_path
        self.col_name = col_name
        self.algos = self._choose_algorithms(algos)
        self.ratios = self._parse_ratios(ratios, ratmin, ratmax, nrat)
        self.repeat = repeat
        self.sequential = sequential
        self.save_csv = save_csv
        self.save_json = save_json
        self.save_plot = save_plot
        self.random_seed = random_seed
        self.expected_cols = expected_cols
        self.data = self._load_and_prepare(csv_path)

    def _load_and_prepare(self, path: str) -> pd.DataFrame:
        if not os.path.isabs(path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, ".."))
            path = os.path.join(project_root, path)
            path = os.path.normpath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)
        if self.expected_cols:
            if len(df.columns) == len(self.expected_cols):
                df.columns = self.expected_cols
            else:
                print(f"Warning: CSV has {len(df.columns)} columns, expected {len(self.expected_cols)}; keeping original names.")
        return df

    @staticmethod
    def _parse_ratios(ratios=None, ratmin=0.01, ratmax=1.0, nrat=6) -> List[float]:
        if ratios:
            ratios_list = [float(x) for x in ratios.split(",") if x.strip()]
        else:
            ratios_list = np.linspace(float(ratmin), float(ratmax), int(nrat))
            ratios_list = [round(r, 3) for r in ratios_list]

        for r in ratios_list:
            if not (0 < r <= 1):
                raise ValueError("ratios must be in (0,1]")
        return sorted(set(ratios_list))

    @staticmethod
    def _choose_algorithms(arg_algos: Optional[str]) -> dict:
        if arg_algos is None:
            return {name: func for name, func in builtin_algorithms.items()}
        names = [s.strip() for s in arg_algos.split(",") if s.strip()]
        chosen = {}
        for n in names:
            if n not in builtin_algorithms:
                raise ValueError(f"Algorithm '{n}' not found. Available: {', '.join(sorted(builtin_algorithms.keys()))}")
            chosen[n] = builtin_algorithms[n]
        return chosen

    @staticmethod
    def _ensure_output_dir(path: Optional[str]):
        if not path:
            return
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def run(self) -> Dict:
        if self.col_name not in self.data.columns:
            raise ValueError(f"Column '{self.col_name}' not found in CSV. Available columns: {self.data.columns.tolist()}")

        series = self.data[self.col_name].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce").dropna()
        cleaned = self.data.loc[series.index].copy()
        cleaned[self.col_name] = series.values

        print(f"Running benchmark on column '{self.col_name}' with {len(cleaned)} rows, {cleaned[self.col_name].nunique()} unique values")
        print(f"Algorithms: {list(self.algos.keys())}")
        print(f"Ratios: {self.ratios}")
        print(f"Repeat: {self.repeat}, Sequential sampling: {self.sequential}")

        tester = SortTester(cleaned, self.col_name, random_seed=self.random_seed)
        ratios_dict = {name: self.ratios for name in self.algos.keys()}

        times_df = tester.run_algorithms(
            self.algos, ratios_dict, repeat=self.repeat, sequential=self.sequential, check_sorted=True
        )
        times_df.index.name = "ratio"

        print("\nTiming table (mean times in seconds):")
        print(times_df)

        complexities = tester.complexity_from_df(times_df)
        print("\nComplexity fits (slope, R²):")
        print(json.dumps(complexities, indent=2))

        if self.save_csv:
            self._ensure_output_dir(self.save_csv)
            times_df.to_csv(self.save_csv)
            print(f"Saved timing table to: {self.save_csv}")

        if self.save_json:
            self._ensure_output_dir(self.save_json)
            with open(self.save_json, "w", encoding="utf-8") as fh:
                json.dump({"timings": times_df.to_dict(), "complexities": complexities}, fh, indent=2)
            print(f"Saved JSON report to: {self.save_json}")

        comp_info = plot_times_df(
            tester.data, times_df, loglog=True, annotate=True, figsize=(8, 6), save_path=self.save_plot, col_name=self.col_name
        )

        if self.save_plot:
            print(f"Saved plot to: {self.save_plot}")

        return {
            "times_df": times_df,
            "complexities": complexities,
            "plot_info": comp_info
        }
