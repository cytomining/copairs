from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        args_to_print = list(args)
        if hasattr(args[0], "__iter__"):
            args_to_print = (*args[0].shape, *args[1:])

        print(
            "func:%r args:[%r, %r] took: %2.4f sec"
            % (f.__name__, args_to_print, kw, te - ts)
        )
        return result

    return wrap
