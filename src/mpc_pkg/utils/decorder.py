import types
'''
时间打印装饰器,统计函数执行时间并每n次打印一次平均时间
'''
def time_print(n=1):
    def decorator(f:types.FunctionType):
        '''装饰器，打印函数执行时间'''
        count=0
        sum_time=0
        import time
        def wrapper(*args, **kwargs):
            nonlocal count, sum_time
            
            start_time = time.time()
            result = f(*args, **kwargs)
            end_time = time.time()
            
            sum_time += (end_time - start_time)
            count += 1
            
            if count >= n:
                print(f"\033[95m{f.__qualname__}\033[0m {n}次平均{(sum_time / n) * 1000:.4f} ms")
                count = 0
                sum_time = 0
            return result
        return wrapper
    return decorator
