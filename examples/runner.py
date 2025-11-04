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

    runner = BenchmarkRunner(
        csv_path="data/TDCS_M06A_20231204_080000.csv",
        col_name="TripLength",
        algos=None,
        ratios=None,
        ratmin=0.01,
        ratmax=0.1,
        nrat=5,
        repeat=10,
        sequential=False,
        save_csv="results/timing.csv",
        save_json="results/report.json",
        save_plot="results/plot.png",
        expected_cols=DEFAULT_COLUMNS
    )

    results = runner.run()
    print("Benchmark completed!")
