from typing import Iterable, Callable, Tuple, List
import time
import numpy as np

def timeit(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0, res)

def is_sorted(seq: Iterable) -> bool:
    it = iter(seq)
    try:
        prev = next(it)
    except StopIteration:
        return True
    for x in it:
        if prev > x:
            return False
        prev = x
    return True

def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(x)
    except Exception:
        return [x]
