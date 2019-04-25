"""
TODO:
* Bokeh settings
"""

import warnings
#warnings.simplefilter("once")
warnings.filterwarnings(action='once')

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    #plt.style.use("seaborn-whitegrid")      # works when no seaborn?
    mpl.rcParams["figure.figsize"] = (15, 15)  # doesnt work?

    # 'axes.titlesize', 'legend.fontsize', 'axes.labelsize', 'axes.titlesize'
    # 'xtick.labelsize', 'ytick.labelsize' 'figure.titlesize'

    mpl.rcParams["hist.bins"]="auto"        # "auto" failed for some data; would like "doane"; alternative "rice" but only takes data size and not variability; "stone" was not found
except ImportError:
    pass
except Exception as e:
    print("Failed setting Matplotlib settings ({})".format(e))


try:
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("notebook")  # poster
    sns.set_palette("Set1", 9)
except ImportError:
    pass
except Exception as e:
    print("Failed setting Seaborn settings ({})".format(e))


try:
    import pd

    pd.set_option("display.max_columns", 500)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.large_repr", "truncate")
    pd.set_option("display.max_info_columns", 1000)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.date_dayfirst", True)
    pd.set_option("display.max_seq_items", 10)
    #pd.set_option("display.precision", 4)

    pd.set_option('float_format', '{:,.5g}'.format)

    # pd.set_option('precision',2)  # decimal determined by number precision in digits
except ImportError:
    pass
except Exception as e:
    print("Failed setting Pandas settings ({})".format(e))


# try:
# import bokeh

# html_printer.for_type(bokeh.charts.chart.Chart, lambda chart:chart.html)
# except Exception as e:
#    print("Failed to set text/html Bokeh pretty display ({})".format(e))
