"""
Write doc of commands!

### Terminal color:
`termcol.coral("TEXT")`
needs str() conversion though

### Curried functions
cfilter, cgroupby, cmap, caccumulate, cmax, cmin, czip

### Countdown
c = Countdown(10)
for ...:
    if c: break

### Pipe
P(f1, f2, ...)(data)

### Other functions
contains("val")          | Curried contains value test
zipl                     | Shorter than zip_longest
"""


class catch_exc:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, exc_tb):
        if exc is not None:
            print("Skipped some default imports due to : {} {}".format(exc_type.__name__, exc))
        return True


import random
from itertools import *

random.seed(123)

with catch_exc():  # due to Python 2
    pass

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    # from IPython.display import display   # already existing in new IPython
    pass

with catch_exc():
    import numpy as np

    np.random.seed(123)

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    pass

with catch_exc():
    from sklearn.ensemble import *
    from sklearn.metrics import *
    from sklearn.model_selection import *
    from sklearn.linear_model import *
    from sklearn.preprocessing import *
    from sklearn.feature_extraction import *
    from sklearn.feature_selection import *
    from sklearn.pipeline import *
    from sklearn.svm import *

with catch_exc():
    from cytoolz.curried import *

    del filter  # for performance reasons
    del map
    del sorted
    del groupby

    from cytoolz import curry

    cmax = curry(max)
    cmin = curry(min)

    czip = lambda xs: zip(*xs)

    listpluck = lambda *args: list(tz.pluck(*args))

with catch_exc():
    import colorful as termcol

    termcol.use_true_colors()


def contains(val):
    return lambda x: val in x


with catch_exc():
    from pyspark.sql.types import *

    print("Using Spark version {}".format(sc.version))  # sc obsolete?


def P(*funcs):
    def wrapped_pipe(data):
        for func in funcs:
            data = func(data)
        return data

    if callable(funcs[0]):
        return wrapped_pipe

    data = funcs[0]
    for func in funcs[1:]:
        data = func(data)
    return data


def countdowns(*counts):
    return chain(*[repeat(i, c) for i, c in enumerate(counts)] +
                  [repeat(None)])


class Countdown:
    """
    Usage:
    >>> c = Countdown(10)
    >>> for ...:
    >>>     if c: break
    """
    def __init__(self, num):
        self.num = num

    def __bool__(self):
        self.num -= 1
        return bool(self.num < 0)


# def clipboard(text=None):
#    from subprocess import Popen, PIPE

#    if text is None:
#        text=_

#    Popen(("xsel", "-i"), stdin=PIPE).communicate(text)
