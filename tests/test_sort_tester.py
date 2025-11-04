import pandas as pd
from sort_tester.core import SortTester
from sort_tester.algorithms import builtin_algorithms

def test_small_array():
    df = pd.DataFrame({'a': [3, 1, 2]})
    st = SortTester(df, 'a')
    algos = {'py': builtin_algorithms['python_sorted']}
    res = st.run_algorithms(algos, {'py': [1.0]}, repeat=2)
    assert res.loc[1.0, 'py'] > 0

def test_counting_sort_correctness():
    df = pd.DataFrame({'a': [5, 1, 3, 1, 2, 2]})
    st = SortTester(df, 'a')
    arr, uniq = st.get_data(1.0, sequential=True)
    out = builtin_algorithms['counting_sort'](arr, unique_list=sorted(uniq))
    assert out == sorted(arr)
