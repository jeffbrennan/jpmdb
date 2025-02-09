import time
from functools import wraps


def timeit(func):
    # https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        print(f"Function {func.__name__} Took {total_time:.2f} ms")
        return result

    return timeit_wrapper
