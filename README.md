## SortTester

Benchmarking toolkit for comparing sorting algorithms on real-world tabular data. It supports parallel execution, robust progress reporting, reproducible sampling, complexity estimation, and flexible manual preprocessing for time, category, and alphanumeric code columns.


### Key Features
- Multiple built-in algorithms: Insertion, Merge, Quick Sort (two-way and three-way, iterative), Comb, Heap, and Radix (integers only).
- Parallel execution with per-(algorithm × ratio) batching; progress via `tqdm` (fallback to console percentage).
- Reproducible sampling via a single random permutation prefix per ratio, reducing variance and better matching theory.
- Complexity fitting on log–log scale with slope and R², overlaid with O(n) and O(n²) reference zones.
- Manual column preprocessing via `col_type` for time, categorical, and structured alphanumeric codes.
- Experiment outputs organized by column name and sampling parameters; titles include key settings.


## Project Structure
```
SortTester/
├── data/
│   └── TDCS_M06A_20231204_080000.csv          # Example dataset
├── main/
│   ├── benchmark_example.py                   # Minimal demo (optional)
│   └── runner.py                              # Recommended entry point
├── result/                                    # Created after runs (sibling to main/)
│   └── <col>_r<ratmin>-<ratmax>_n<nrat>/      # e.g., TripLength_r0.01-0.1_n5
│       ├── timing.csv                         # Mean timing table
│       ├── report.json                        # Timings + complexity fits
│       └── plot.png                           # Visualization
├── sort_tester/
│   ├── __init__.py
│   ├── algorithms.py                          # Algorithm implementations and registry
│   ├── Benchmark_runner.py                    # High-level runner and I/O
│   ├── core.py                                # Benchmark core (sampling, parallelism, fits)
│   ├── plotting.py                            # Plotting and complexity estimation
│   └── utils.py                               # Utilities (timeit, is_sorted, etc.)
└── README.md
```


## Installation
- Python 3.8+
- Dependencies: `numpy`, `pandas`, `matplotlib`
- Optional: `tqdm` (for better progress bars)

Example:
```bash
pip install numpy pandas matplotlib tqdm
```


## Quick Start
Configure and run `main/runner.py`. Example:

```python
col_name = "GantryID_O"
col_type = "code"         # None / "time" / "category" / "code"
ratmin = 0.2
ratmax = 1
nrat = 5

result_dir = os.path.join(project_root, "result", f"{col_name}_r{ratmin}-{ratmax}_n{nrat}")

runner = BenchmarkRunner(
    csv_path="data/TDCS_M06A_20231204_080000.csv",
    col_name=col_name,
    col_type=col_type,                  # manual preprocessing (see below)
    algos=None,                         # use all built-ins or "quick_sort_2,quick_sort_3"
    ratios=None,                        # or "0.02,0.05,0.1" to customize
    ratmin=ratmin,
    ratmax=ratmax,
    nrat=nrat,
    repeat=10,                          # repetitions per (algo, ratio)
    sequential=False,                   # sequential prefix vs randomized prefix
    prefix_random=True,                 # use randomized permutation prefix (recommended)
    save_csv=os.path.join(result_dir, "timing.csv"),
    save_json=os.path.join(result_dir, "report.json"),
    save_plot=os.path.join(result_dir, "plot.png"),
    expected_cols=DEFAULT_COLUMNS,      # optional: rename CSV columns if needed
    n_jobs=max(1, (os.cpu_count() or 2) - 1),  # parallel workers
    copy_input=False
)

runner.run()
```

Results are written to `result/<col>_r<ratmin>-<ratmax>_n<nrat>/`. The figure title includes the column and sampling parameters.


## Data Preprocessing (Manual)
Control via `col_type` in `BenchmarkRunner`:

- `None` (default): use numeric as-is; for strings, attempt `pandas.to_numeric(..., errors="coerce")` and drop non-numeric rows.

- `time` | `datetime` | `timestamp` | `date`: parse strings using `pandas.to_datetime(..., errors="coerce")`; convert to 64-bit nanosecond integers for numeric sorting. Parsing failures raise errors.

- `category` | `categorical`: encode strings as categorical codes. You can specify a custom order via `category_order=[...]`. Rows outside the category set are dropped.

- `code` | `gantry` | `alphanum`: parse structured alphanumeric codes like `01F3640S` using the regex
  `^(\d+)([A-Za-z])(\d+)([A-Za-z])$`, and map to sortable tuples `(p1, p2, p3, p4)` where:
  - `p1`: integer of the first number block
  - `p2`: alphabetical rank of the first letter (A=0, B=1, ...)
  - `p3`: integer of the second number block
  - `p4`: rank of the trailing letter (default `{"N":0, "S":1}`; override with `code_suffix_order=[...]`)
  Rows failing the pattern are dropped with an informative error.


## Sampling & Benchmark Methodology

### Stable Sampling by Ratio
For each ratio \( r \in (0, 1] \), we select the first \( \lfloor r \cdot n \rfloor \) items from a single random permutation of the dataset (controlled by a fixed seed). This makes samples across ratios nested and reduces variance compared to independent re-sampling.

### Repetitions and Correctness
Each (algorithm, ratio) is run `repeat` times and the mean time is recorded. The first run checks correctness by comparing the output with Python’s `sorted(arr)`; if it differs, the mean time is set to NaN for that (algorithm, ratio), and the benchmark continues without interruption.

### Parallel Execution
When `n_jobs > 1`, (algorithm × ratio) tasks are executed in a process pool. Progress updates are aggregated: `tqdm` is used when available; otherwise, a concise console percentage is shown.


## Complexity Estimation
We fit a straight line to $(\log n, \log t)$ pairs to estimate the exponent $a$ in $t \propto n^a$:

$$
x = \log n,\quad y = \log t,\quad \hat{y} = a x + b
$$

The slope $a$ and coefficient of determination $R^2$ are computed via ordinary least squares:

$$
(a, b) = \arg\min_{a,b} \sum_i \bigl(y_i - (a x_i + b)\bigr)^2,\quad
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Interpretation: $a \approx 1$ suggests near $O(n)$, $a \approx 2$ suggests near $O(n^2)$, and so on. The plot overlays reference bands for $O(n)$ and $O(n^2)$ to aid visual comparison.


## Algorithms (`sort_tester/algorithms.py`)
- `insertion_sort` (O(n²))
- `merge_sort` (O(n log n))
- `quick_sort_2` (iterative two-way partition; random pivot; robust for duplicates)
- `quick_sort_3` (iterative three-way partition; random pivot; duplicate-friendly)
- `comb_sort`
- `heap_sort` (O(n log n))
- `radix_sort` (integers only; auto-skipped otherwise)

You can register custom algorithms via `add_algorithm(name, fn)` and select them with `BenchmarkRunner(algos="name1,name2")`.


## Outputs & Visualization
- `timing.csv`: rows = ratios, columns = algorithms, values = mean time (seconds).
- `report.json`: contains `"timings"` (table as dict) and `"complexities"` (per-algorithm `slope` and `r2`).
- `plot.png`: log–log fitted lines per algorithm with reference zones. Title:
  `Algorithm timings for <col_name> | ratmin=<...>, ratmax=<...>, nrat=<...>`.


## Performance Tips
- Choose `n_jobs` near CPU cores (e.g., `cores - 1`).
- Prefer wider ratio ranges (e.g., 0.02–0.2) over very large `repeat` to stabilize slope estimation.
- Limit algorithms under test (e.g., `algos="quick_sort_2,quick_sort_3"`) to speed up runs.
- Disable plotting (`save_plot=None`) if you only need tables or JSON.


## Troubleshooting
- `ModuleNotFoundError: sort_tester`: `main/runner.py` prepends the project root to `sys.path`. If you relocate the entry point, ensure the same.
- Recursive quicksort overflow: provided quicksorts are iterative; if you add a recursive one, guard against equal elements and recursion depth.
- `radix_sort` skipped: only included for integer columns.
- Time parsing failed: set `col_type="time"` and ensure the format is parsable by `pandas.to_datetime`.
- Code parsing failed: verify your alphanumeric pattern; adjust `code_suffix_order` or extend the regex if needed.
