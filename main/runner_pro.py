import os
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sort_tester.Benchmark_runner import BenchmarkRunner

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

    # tasks: (column name, col_type)
    tasks = [
        ("VehicleType", None),
        ("DetectionTime_O", "time"),
        ("GantryID_O", "code"),
        ("TripLength", None),
    ]

    csv_path = "data/TDCS_M06A_20231204_080000.csv"
    ratmin = 0.2
    ratmax = 1
    nrat = 10
    repeat = 10
    sequential = False
    prefix_random = True
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    copy_input = False

    for idx, (col_name, col_type) in enumerate(tasks, start=1):
        print(f"\n=== Task {idx}/{len(tasks)}: col_name={col_name}, col_type={col_type} ===")
        result_dir = os.path.join(project_root, "result", f"{col_name}_r{ratmin}-{ratmax}_n{nrat}")
        try:
            runner = BenchmarkRunner(
                csv_path=csv_path,
                col_name=col_name,
                col_type=col_type,
                algos=None,
                ratios=None,
                ratmin=ratmin,
                ratmax=ratmax,
                nrat=nrat,
                repeat=repeat,
                sequential=sequential,
                prefix_random=prefix_random,
                save_csv=os.path.join(result_dir, "timing.csv"),
                save_json=os.path.join(result_dir, "report.json"),
                save_plot=os.path.join(result_dir, "plot.png"),
                expected_cols=DEFAULT_COLUMNS,
                n_jobs=n_jobs,
                copy_input=copy_input,
            )
            _ = runner.run()
            print(f"Task {idx}: completed. Outputs in: {result_dir}")
        except Exception as e:
            print(f"Task {idx}: FAILED with error: {e}")

    print("\nAll tasks completed (some may have failed; see logs above).")


if __name__ == "__main__":
    main()

