import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sort_tester.Benchmark_runner import BenchmarkRunner

if __name__ == "__main__":

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

    col_name = "GantryID_O"
    # None / "time" / "category" / "code"
    col_type = "code"
    ratmin = 0.01
    ratmax = 0.1
    nrat = 5

    result_dir = os.path.join(project_root, "result", f"{col_name}_r{ratmin}-{ratmax}_n{nrat}")

    runner = BenchmarkRunner(
        csv_path="data/TDCS_M06A_20231204_080000.csv",
        col_name=col_name,
        col_type=col_type,
        algos=None,
        ratios=None,
        ratmin=ratmin,
        ratmax=ratmax,
        nrat=nrat,
        repeat=10,
        sequential=False,
        prefix_random=True,
        save_csv=os.path.join(result_dir, "timing.csv"),
        save_json=os.path.join(result_dir, "report.json"),
        save_plot=os.path.join(result_dir, "plot.png"),
        expected_cols=DEFAULT_COLUMNS,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        copy_input=False
    )

    results = runner.run()
    print("Benchmark completed!")
