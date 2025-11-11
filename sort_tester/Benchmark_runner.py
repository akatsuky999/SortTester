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
        col_type: Optional[str] = None,
        category_order: Optional[List[str]] = None,
        code_suffix_order: Optional[List[str]] = None,
        algos: Optional[str] = None,
        ratios: Optional[str] = None,
        ratmin: float = 0.01,
        ratmax: float = 1.0,
        nrat: int = 6,
        repeat: int = 3,
        sequential: bool = False,
        prefix_random: bool = True,
        save_csv: Optional[str] = None,
        save_json: Optional[str] = None,
        save_plot: Optional[str] = None,
        random_seed: int = 42,
        expected_cols: Optional[list] = None,
        n_jobs: int = 1,
        copy_input: bool = False
    ):
        self.csv_path = csv_path
        self.col_name = col_name
        self.col_type = (col_type or "").strip().lower() or None
        self.category_order = category_order
        self.code_suffix_order = code_suffix_order
        # preserve inputs for title/folder info
        self._ratmin_input = ratmin
        self._ratmax_input = ratmax
        self._nrat_input = nrat
        self.ratios = self._parse_ratios(ratios, ratmin, ratmax, nrat)
        self.repeat = repeat
        self.sequential = sequential
        self.prefix_random = prefix_random
        self.save_csv = save_csv
        self.save_json = save_json
        self.save_plot = save_plot
        self.random_seed = random_seed
        self.expected_cols = expected_cols
        self.data = self._load_and_prepare(csv_path)
        self.algos_arg = algos
        self.n_jobs = int(n_jobs)
        self.copy_input = bool(copy_input)

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
    def _choose_algorithms(arg_algos: Optional[str], col_series: pd.Series) -> dict:
        if arg_algos is None:
            chosen = {name: func for name, func in builtin_algorithms.items()}
        else:
            names = [s.strip() for s in arg_algos.split(",") if s.strip()]
            chosen = {}
            for n in names:
                if n not in builtin_algorithms:
                    raise ValueError(f"Algorithm '{n}' not found. Available: {', '.join(sorted(builtin_algorithms.keys()))}")
                chosen[n] = builtin_algorithms[n]

        # 如果列是浮点数，则剔除 radix_sort
        if not pd.api.types.is_integer_dtype(col_series) and 'radix_sort' in chosen:
            print("Note: Column is not integer, skipping radix_sort")
            chosen.pop('radix_sort')

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

        series_raw = self.data[self.col_name]
        # Manual typed preprocessing
        if self.col_type in {"time", "datetime", "timestamp", "date"}:
            dt = pd.to_datetime(series_raw, errors="coerce")
            mask = dt.notna()
            if mask.sum() == 0:
                raise ValueError(f"Column '{self.col_name}' set as {self.col_type} but cannot be parsed as datetime.")
            series = dt[mask].view("int64")
            cleaned = self.data.loc[mask].copy()
            cleaned[self.col_name] = series.values
            print(f"Column '{self.col_name}' parsed as datetime -> int64 ns ({mask.sum()} rows).")
        elif self.col_type in {"category", "categorical"}:
            ser = series_raw.astype("string")
            if self.category_order:
                cat = pd.Categorical(ser, categories=self.category_order, ordered=True)
            else:
                cat = pd.Categorical(ser)
            codes = pd.Series(cat.codes, index=ser.index)
            mask = codes >= 0
            if mask.sum() == 0:
                raise ValueError(f"Column '{self.col_name}' set as category but no valid categories found.")
            cleaned = self.data.loc[mask].copy()
            cleaned[self.col_name] = codes.loc[mask].values
            series = cleaned[self.col_name]
            print(f"Column '{self.col_name}' parsed as categorical codes ({mask.sum()} rows).")
        elif self.col_type in {"code", "gantry", "alphanum"}:
            import re
            pattern = re.compile(r'^\s*(\d+)([A-Za-z])(\d+)([A-Za-z])\s*$')
            suffix_rank = {}
            if self.code_suffix_order:
                for i, k in enumerate(self.code_suffix_order):
                    suffix_rank[str(k).upper()] = i
            else:
                suffix_rank = {"N": 0, "S": 1}
            def encode_one(x):
                if not isinstance(x, str):
                    try:
                        x = str(x)
                    except Exception:
                        return None
                m = pattern.match(x)
                if not m:
                    return None
                g1, g2, g3, g4 = m.groups()
                try:
                    p1 = int(g1)
                    p2 = ord(g2.upper()) - ord('A')
                    p3 = int(g3)
                    p4 = suffix_rank.get(g4.upper(), ord(g4.upper()) - ord('A') + 100)
                    return (p1, p2, p3, p4)
                except Exception:
                    return None
            tuples = series_raw.map(encode_one)
            mask = tuples.notna()
            if mask.sum() == 0:
                raise ValueError(f"Column '{self.col_name}' set as code but values do not match expected pattern.")
            cleaned = self.data.loc[mask].copy()
            cleaned[self.col_name] = tuples.loc[mask].tolist()
            series = cleaned[self.col_name]
            print(f"Column '{self.col_name}' parsed as code tuples ({mask.sum()} rows).")
        else:
            series = series_raw.dropna()
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors="coerce").dropna()
            cleaned = self.data.loc[series.index].copy()
            cleaned[self.col_name] = series.values

        self.algos = self._choose_algorithms(self.algos_arg, series)

        print(f"Running benchmark on column '{self.col_name}' with {len(cleaned)} rows, {cleaned[self.col_name].nunique()} unique values")
        print(f"Algorithms: {list(self.algos.keys())}")
        print(f"Ratios: {self.ratios}")
        print(f"Repeat: {self.repeat}, Sequential sampling: {self.sequential}")

        tester = SortTester(cleaned, self.col_name, random_seed=self.random_seed, copy_input=self.copy_input)
        ratios_dict = {name: self.ratios for name in self.algos.keys()}

        times_df = tester.run_algorithms(
            self.algos,
            ratios_dict,
            repeat=self.repeat,
            sequential=self.sequential,
            prefix_random=self.prefix_random,
            check_sorted=True,
            show_progress=True,
            n_jobs=self.n_jobs
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

        # build title suffix with ratio info when available
        title_suffix = None
        if isinstance(self._nrat_input, (int, float)) and isinstance(self._ratmin_input, (int, float)) and isinstance(self._ratmax_input, (int, float)):
            title_suffix = f"ratmin={self._ratmin_input}, ratmax={self._ratmax_input}, nrat={self._nrat_input}"

        comp_info = plot_times_df(
            tester.data, times_df, loglog=True, annotate=True, figsize=(8, 6), save_path=self.save_plot, col_name=self.col_name, title_suffix=title_suffix
        )

        if self.save_plot:
            print(f"Saved plot to: {self.save_plot}")

        return {
            "times_df": times_df,
            "complexities": complexities,
            "plot_info": comp_info
        }
