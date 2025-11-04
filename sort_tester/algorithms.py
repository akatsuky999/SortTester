from typing import Callable, Dict, List
from .utils import ensure_list

Algorithm = Callable[[List], List]
builtin_algorithms: Dict[str, Algorithm] = {}

def register(name: str):
    def _decorator(fn: Algorithm):
        builtin_algorithms[name] = fn
        return fn
    return _decorator

@register('python_sorted')
def python_sorted(arr: List, **kw):
    return sorted(arr)

@register('counting_sort')
def counting_sort(arr: List, unique_list=None, **kw):
    if unique_list is None:
        unique_list = sorted(set(arr))
    counts = {v: 0 for v in unique_list}
    for v in arr:
        counts[v] = counts.get(v, 0) + 1
    out = []
    for v in unique_list:
        out.extend([v] * counts.get(v, 0))
    return out

@register('insertion_sort')
def insertion_sort(arr: List, **kw):
    a = list(arr)
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a

@register('merge_sort')
def merge_sort(arr: List, **kw):
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out

@register('quick_sort')
def quick_sort(arr: List, **kw):
    if len(arr) <= 1:
        return list(arr)
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)

def add_algorithm(name: str, fn: Algorithm):
    if name in builtin_algorithms:
        raise ValueError(f"Algorithm '{name}' already exists")
    builtin_algorithms[name] = fn
