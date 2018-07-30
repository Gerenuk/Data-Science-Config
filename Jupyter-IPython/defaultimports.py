class catch_exc:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, exc_tb):
        if exc is not None:
            print("Skipped some default imports due to : {} {}".format(exc_type.__name__, exc))
        return True


from functools import reduce, partial
from operator import *
from pprint import pprint
from copy import deepcopy

from bisect import *
from abc import *
from math import *
from itertools import *
from collections import *

from itertools import zip_longest as zipl

import contextlib
import csv
import glob
import logging
import os
import pickle
import random
import re
import sys
import time
import sqlite3

from pathlib import Path

import datetime as dt

random.seed(123)

with catch_exc():  # due to Python 2
    from reprlib import repr as arepr

with catch_exc():
    import sklearn

with catch_exc():
    from tqdm import tqdm_notebook as tqdm

with catch_exc():
    from IPython.display import display
    from IPython.display import HTML

with catch_exc():
    import numpy as np

    np.random.seed(123)

with catch_exc():
    import pandas as pd

with catch_exc():
    import statsmodels.api as sm

with catch_exc():
    import scipy

with catch_exc():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

with catch_exc():
    import bokeh.plotting as bk

with catch_exc():
    import seaborn as sns

with catch_exc():
    from statistics import *

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

    from cytoolz.curried import filter as cfilter
    from cytoolz.curried import groupby as cgroupby
    from cytoolz.curried import accumulate as caccumulate

    from cytoolz import curry

    cmax = curry(max)
    cmin = curry(min)

    czip = lambda xs: zip(*xs)

    listpluck = lambda *args: list(tz.pluck(*args))


def contains(val):
    return lambda x: val in x


from itertools import groupby

with catch_exc():
    import pyspark.sql.functions as F
    from pyspark.sql.types import *
    from pyspark.sql.window import Window

    print("Using Spark version {}".format(sc.version))

import itertools


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
