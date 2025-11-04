from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from .utils import timeit, is_sorted, ensure_list
from .algorithms import builtin_algorithms

class SortTester:
    def __init__(self, data: pd.DataFrame, col_name: str, random_seed: int = 0, copy_input: bool = True):
        if col_name not in data.columns:
            raise ValueError(f"column '{col_name}' not found in DataFrame")
        self.data = data.reset_index(drop=True).copy()
        self.col_name = col_name
        self.random_seed = int(random_seed)
        self.copy_input = bool(copy_input)

    def get_data(self, ratio: float, sequential: bool = False, as_list: bool = True) -> Tuple[List, List]:
        if not (0 < ratio <= 1):
            raise ValueError("ratio must be in (0, 1]")
        n = len(self.data)
        if n == 0:
            return [], []
        if sequential:
            seg_len = max(1, int(n * ratio))
            sampled = self.data.iloc[:seg_len]
        else:
            sampled = self.data.sample(frac=ratio, random_state=self.random_seed)
        vals = sampled[self.col_name].tolist()
        uniq = sampled[self.col_name].unique().tolist()
        if self.copy_input:
            vals = list(vals)
        return vals, uniq

    def run_single(self, alg: Callable, arr: List, unique_list: Optional[List] = None, check_sorted: bool = True) -> Dict:
        run_args = {}
        if unique_list is not None:
            run_args['unique_list'] = unique_list
        elapsed, out = timeit(alg, list(arr), **run_args)
        correct = True
        if check_sorted:
            correct = is_sorted(out)
        return {'time': elapsed, 'correct': bool(correct), 'output': out}

    def run_algorithms(self, alg_dict: Dict[str, Callable], ratios_dict: Dict[str, List[float]], repeat: int = 3, sequential: bool = False, check_sorted: bool = True) -> pd.DataFrame:
        all_ratios = sorted(set(r for r_list in ratios_dict.values() for r in r_list))
        results = {name: [] for name in alg_dict.keys()}
        for r in all_ratios:
            arr, uniq = self.get_data(r, sequential=sequential)
            for name, alg in alg_dict.items():
                times = []
                correctness = []
                for _ in range(repeat):
                    res = self.run_single(alg, arr, uniq if name == 'counting_sort' else None, check_sorted=check_sorted)
                    times.append(res['time'])
                    correctness.append(res['correct'])
                mean_time = float(np.mean(times))
                if not all(correctness):
                    mean_time = float('nan')
                results[name].append(mean_time)
        df = pd.DataFrame(results, index=all_ratios)
        df.index.name = 'ratio'
        return df

    def complexity_from_df(self, times_df: pd.DataFrame):
        n_values = (times_df.index.values * len(self.data)).astype(int)
        out = {}
        for col in times_df.columns:
            times = times_df[col].values.astype(float)
            mask = np.isfinite(times) & (times > 0)
            if mask.sum() < 2:
                out[col] = {'slope': np.nan, 'r2': np.nan}
                continue
            logn = np.log(n_values[mask])
            logt = np.log(times[mask])
            slope, intercept = np.polyfit(logn, logt, 1)
            pred = slope * logn + intercept
            ss_res = ((logt - pred) ** 2).sum()
            ss_tot = ((logt - logt.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            out[col] = {'slope': float(slope), 'r2': float(r2)}
        return out

    def plot_summary(self, times_df: pd.DataFrame, loglog: bool = True):
        from .plotting import plot_times_df
        plot_times_df(self.data, times_df, loglog=loglog)
