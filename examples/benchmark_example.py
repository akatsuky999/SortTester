import pandas as pd
import numpy as np
from sort_tester.core import SortTester
from sort_tester.algorithms import builtin_algorithms

N = 10000
rng = np.random.default_rng(0)
df = pd.DataFrame({'score': rng.integers(0, 10000, size=N)})
st = SortTester(df, 'score', random_seed=123)

algos = builtin_algorithms
ratios = {k: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0] for k in algos.keys()}

res = st.run_algorithms(algos, ratios, repeat=3)
print(res)
st.plot_summary(res)
