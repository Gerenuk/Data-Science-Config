"""
Write doc of commands!
"""


class catch_exc:
    def __init__(self, print_error=True):
        self.print_error = print_error

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, exc_tb):
        if self.print_error and exc is not None:
            print(
                "Skipped some default imports due to : {} {}".format(
                    exc_type.__name__, exc
                )
            )
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
import datetime as dt

random.seed(123)

with catch_exc():
    from pathlib import Path


with catch_exc():
    import sklearn

    from sklearn.cluster import *
    from sklearn.decomposition import *
    from sklearn.discriminant_analysis import *
    from sklearn.ensemble import *
    from sklearn.linear_model import *
    from sklearn.metrics import *
    from sklearn.model_selection import *
    from sklearn.pipeline import *
    from sklearn.preprocessing import *

with catch_exc():
    from tqdm import tqdm


with catch_exc():
    import numpy as np

    np.random.seed(123)


with catch_exc():
    import pandas as pd

    from pandas import DataFrame
    from pandas import Series


with catch_exc():
    import statsmodels.api as sm

with catch_exc():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    def hist(*args, **kwargs):
        for param, val in [
            ("bins", "doane"),
            ("density", True),
            ("histtype", "stepfilled"),
        ]:
            if param not in kwargs:
                kwargs[param] = val

        return plt.hist(*args, **kwargs)

    def figsize(size_x, size_y):
        mpl.rcParams["figure.figsize"] = (size_x, size_y)


with catch_exc():
    import bokeh.plotting as bk

with catch_exc():
    import holoviews as hv

with catch_exc():
    import seaborn as sns

with catch_exc():
    from statistics import *

with catch_exc():
    from cytoolz.curried import *

    del filter  # for performance reasons
    del map
    del sorted
    from itertools import groupby, accumulate

    # curried versions
    from cytoolz.curried import filter as cfilter
    from cytoolz.curried import map as cmap
    from cytoolz.curried import sorted as csorted
    from cytoolz.curried import groupby as cgroupby
    from cytoolz.curried import accumulate as caccumulate

    from cytoolz import curry

    cmax = curry(max)
    cmin = curry(min)

    czip = lambda xs: zip(*xs)


# def contains(val):
#    return lambda x: val in x


with catch_exc(print_error=False):
    import pyspark.sql.functions as F
    from pyspark.sql.types import *
    from pyspark.sql.window import Window
