from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
import inspect
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from .utils import timeit, is_sorted
from .algorithms import builtin_algorithms

def _mean_time_for_alg(alg_path: str, repeat: int, check_sorted: bool, arr: List) -> float:
    """
    Run a single algorithm multiple times on the given array and return the mean time.
    alg_path: 'module:function' path to import the algorithm to ensure picklability on Windows.
    """
    # Import inside the process to avoid pickling function objects on Windows spawn
    module_name, func_name = alg_path.rsplit(":", 1)
    mod = __import__(module_name, fromlist=[func_name])
    alg = getattr(mod, func_name)
    times: List[float] = []
    sig = inspect.signature(alg)
    total_runs = max(1, int(repeat))
    ref_sorted_first: List = []
    for i in range(total_runs):
        arr_copy = list(arr)
        if 'arr' in sig.parameters:
            elapsed, out = timeit(alg, arr=arr_copy)
        else:
            elapsed, out = timeit(alg, arr_copy)
        if i == 0 and check_sorted:
            ref_sorted_first = sorted(arr_copy)
            if out != ref_sorted_first:
                return float("nan")
        times.append(elapsed)
    return float(np.nanmean(times)) if len(times) > 0 else float("nan")

class SortTester:
    def __init__(self, data: pd.DataFrame, col_name: str, random_seed: int = 0, copy_input: bool = True):
        if col_name not in data.columns:
            raise ValueError(f"column '{col_name}' not found in DataFrame")
        self.data = data.reset_index(drop=True).copy()
        self.col_name = col_name
        self.random_seed = int(random_seed)
        self.copy_input = bool(copy_input)
        # Precompute a single random permutation for stable prefix sampling across ratios
        rng = np.random.default_rng(self.random_seed)
        self._perm_indices = rng.permutation(len(self.data)) if len(self.data) > 0 else np.array([], dtype=int)

    def get_data(self, ratio: float, sequential: bool = False, prefix_random: bool = True) -> List:
        if not (0 < ratio <= 1):
            raise ValueError("ratio must be in (0, 1]")
        n = len(self.data)
        if n == 0:
            return []
        if prefix_random:
            seg_len = max(1, int(n * ratio))
            idx = self._perm_indices[:seg_len]
            sampled = self.data.iloc[idx]
        elif sequential:
            seg_len = max(1, int(n * ratio))
            sampled = self.data.iloc[:seg_len]
        else:
            sampled = self.data.sample(frac=ratio, random_state=self.random_seed)
        vals = sampled[self.col_name].tolist()
        if self.copy_input:
            vals = list(vals)
        return vals

    def run_single(self, alg: Callable, arr: List, check_sorted: bool = True) -> Dict:
        arr_copy = list(arr)
        sig = inspect.signature(alg)
        if 'arr' in sig.parameters:
            elapsed, out = timeit(alg, arr=arr_copy)
        else:
            elapsed, out = timeit(alg, arr_copy)
        correct = True
        if check_sorted:
            correct = (out == sorted(arr_copy))
        return {'time': elapsed, 'correct': bool(correct), 'output': out}

    def run_algorithms(
        self,
        alg_dict: Dict[str, Callable],
        ratios_dict: Dict[str, List[float]],
        repeat: int = 3,
        sequential: bool = False,
        check_sorted: bool = True,
        show_progress: bool = True,
        n_jobs: int = 1,
        prefix_random: bool = True
    ) -> pd.DataFrame:
        all_ratios = sorted(set(r for r_list in ratios_dict.values() for r in r_list))
        results = {name: [] for name in alg_dict.keys()}
        total_steps = len(all_ratios) * max(1, len(alg_dict)) * max(1, int(repeat))
        done_steps = 0

        pbar = None
        use_simple_progress = False
        progress_is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        last_progress_len = 0
        if show_progress and total_steps > 0:
            try:
                from tqdm import tqdm  # type: ignore
                pbar = tqdm(total=total_steps, desc="Benchmark", unit="run")
            except Exception:
                use_simple_progress = bool(progress_is_tty)
                if use_simple_progress:
                    msg = f"Progress: 0/{total_steps} (0%)"
                    sys.stderr.write(msg)
                    sys.stderr.flush()
                    last_progress_len = len(msg)

        use_parallel = isinstance(n_jobs, int) and n_jobs > 1
        if use_parallel:
            # Avoid sending function objects across processes; send import path instead
            # Build a mapping name -> "module:function"
            alg_import_paths: Dict[str, str] = {}
            for name, alg in alg_dict.items():
                module_name = getattr(alg, "__module__", None)
                func_name = getattr(alg, "__name__", None)
                if not module_name or not func_name:
                    raise ValueError(f"Algorithm '{name}' is not importable in child process")
                alg_import_paths[name] = f"{module_name}:{func_name}"

            with ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
                for r in all_ratios:
                    arr = self.get_data(r, sequential=sequential, prefix_random=prefix_random)
                    future_to_name = {}
                    for name in alg_dict.keys():
                        fut = executor.submit(_mean_time_for_alg, alg_import_paths[name], int(repeat), bool(check_sorted), arr)
                        future_to_name[fut] = name
                    for fut in as_completed(future_to_name):
                        name = future_to_name[fut]
                        mean_time = float(fut.result())
                        results[name].append(mean_time)
                        if show_progress and total_steps > 0:
                            # One task covers 'repeat' unit-steps
                            step_inc = int(repeat)
                            done_steps += step_inc
                            if pbar is not None:
                                pbar.update(step_inc)
                            elif use_simple_progress:
                                percent = int(done_steps * 100 / total_steps)
                                msg = f"Progress: {done_steps}/{total_steps} ({percent}%)"
                                padding = " " * max(0, last_progress_len - len(msg))
                                sys.stderr.write("\r" + msg + padding)
                                sys.stderr.flush()
                                last_progress_len = len(msg)
        else:
            for r in all_ratios:
                arr = self.get_data(r, sequential=sequential, prefix_random=prefix_random)
                for name, alg in alg_dict.items():
                    times = []
                    # first run: check correctness once
                    res = self.run_single(alg, arr, check_sorted=check_sorted)
                    times.append(res['time'] if res['correct'] else np.nan)
                    if show_progress and total_steps > 0:
                        done_steps += 1
                        if pbar is not None:
                            pbar.update(1)
                        elif use_simple_progress:
                            percent = int(done_steps * 100 / total_steps)
                            msg = f"Progress: {done_steps}/{total_steps} ({percent}%)"
                            padding = " " * max(0, last_progress_len - len(msg))
                            sys.stderr.write("\r" + msg + padding)
                            sys.stderr.flush()
                            last_progress_len = len(msg)
                    # remaining runs: no correctness check to avoid O(n) overhead
                    for _ in range(max(0, int(repeat) - 1)):
                        arr_copy = list(arr)
                        sig = inspect.signature(alg)
                        if 'arr' in sig.parameters:
                            elapsed, _ = timeit(alg, arr=arr_copy)
                        else:
                            elapsed, _ = timeit(alg, arr_copy)
                        times.append(elapsed)
                        if show_progress and total_steps > 0:
                            done_steps += 1
                            if pbar is not None:
                                pbar.update(1)
                            elif use_simple_progress:
                                percent = int(done_steps * 100 / total_steps)
                                msg = f"Progress: {done_steps}/{total_steps} ({percent}%)"
                                padding = " " * max(0, last_progress_len - len(msg))
                                sys.stderr.write("\r" + msg + padding)
                                sys.stderr.flush()
                                last_progress_len = len(msg)
                    mean_time = float(np.nanmean(times)) if len(times) > 0 else float('nan')
                    results[name].append(mean_time)
        if pbar is not None:
            pbar.close()
        elif use_simple_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()
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
