import logging

import warnings

warnings.simplefilter("once")  # does it work?
logging.captureWarnings(True)


def create_logger():
    import colorlog

    log_filename = "out.log"
    log_format = "%(log_color)s[%(levelname)s]%(white)s %(asctime)s %(name)s:%(funcName)s(L%(lineno)s)%(reset)s %(message)s"
    date_format = "%H:%M"

    logger = logging.getLogger("mylog")
    logger.setLevel("DEBUG")

    handler = logging.FileHandler(log_filename)

    formatter = colorlog.ColoredFormatter(log_format, datefmt=date_format)

    handler.setFormatter(formatter)
    logger.addHandler(handler)


#logger = create_logger()

#del create_logger


try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["figure.figsize"] = (15, 15)  # doesnt work?

    # 'axes.titlesize', 'legend.fontsize', 'axes.labelsize', 'axes.titlesize'
    # 'xtick.labelsize', 'ytick.labelsize' 'figure.titlesize'

    mpl.rcParams[
        "hist.bins"
    ] = (
        "auto"
    )  # "auto" failed for some data; would like "doane"; "sturges" was ok; rest not
except ImportError as e:
    pass
except Exception as e:
    logging.warning("Failed setting Matplotlib settings ({})".format(e))


try:
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_context("notebook")  # poster
    sns.set_palette("Set1", 9)
except ImportError:
    pass
except Exception as e:
    logging.warning("Failed setting Seaborn settings ({})".format(e))


try:
    import numpy as np

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(precision=4, edgeitems=4, threshold=100, floatmode="maxprec")

except Exception as e:
    logging.warning("Failed setting Numpy settings ({})".format(e))


try:
    import pandas as pd

    pd.set_option("display.max_rows", 40)   # so that still can scroll
    pd.set_option("display.max_columns", 300)

    pd.set_option("display.large_repr", "truncate")
    pd.set_option("display.max_info_columns", 1000)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.date_dayfirst", True)
    pd.set_option("display.max_seq_items", 10)
    # pd.set_option("display.precision", 4)

    #pd.set_option("float_format", "{:,.5g}".format)  # too dangerous for not seeing decimal places

    # pd.set_option('precision',2)  # decimal determined by number precision in digits
except ImportError:
    pass
except Exception as e:
    logging.warning("Failed setting Pandas settings ({})".format(e))

