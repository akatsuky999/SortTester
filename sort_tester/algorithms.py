from typing import Callable, Dict, List
from .utils import ensure_list

Algorithm = Callable[[List], List]
builtin_algorithms: Dict[str, Algorithm] = {}

def register(name: str):
    def _decorator(fn: Algorithm):
        builtin_algorithms[name] = fn
        return fn
    return _decorator

# @register('python_sorted')
# def python_sorted(arr: List, **kw):
#     return sorted(arr)

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

@register('comb_sort')
def comb_sort(arr: List, **kw):
    a = list(arr)
    n = len(a)
    gap = n
    shrink = 1.3
    sorted_flag = False
    while not sorted_flag:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_flag = True
        i = 0
        while i + gap < n:
            if a[i] > a[i + gap]:
                a[i], a[i + gap] = a[i + gap], a[i]
                sorted_flag = False
            i += 1
    return a

@register('radix_sort')
def radix_sort(arr: List, **kw):
    a = list(arr)
    if not a:
        return a
    max_val = max(a)
    exp = 1
    while max_val // exp > 0:
        output = [0]*len(a)
        count = [0]*10
        for i in a:
            index = (i // exp) % 10
            count[index] += 1
        for i in range(1, 10):
            count[i] += count[i-1]
        for i in reversed(a):
            index = (i // exp) % 10
            output[count[index]-1] = i
            count[index] -= 1
        a = list(output)
        exp *= 10
    return a

@register('heap_sort')
def heap_sort(arr: List, **kw):
    a = list(arr)
    n = len(a)
    def heapify(n, i):
        largest = i
        l = 2*i + 1
        r = 2*i + 2
        if l < n and a[l] > a[largest]:
            largest = l
        if r < n and a[r] > a[largest]:
            largest = r
        if largest != i:
            a[i], a[largest] = a[largest], a[i]
            heapify(n, largest)
    for i in range(n//2 - 1, -1, -1):
        heapify(n, i)
    for i in range(n-1, 0, -1):
        a[0], a[i] = a[i], a[0]
        heapify(i, 0)
    return a


def add_algorithm(name: str, fn: Algorithm):
    if name in builtin_algorithms:
        raise ValueError(f"Algorithm '{name}' already exists")
    builtin_algorithms[name] = fn
