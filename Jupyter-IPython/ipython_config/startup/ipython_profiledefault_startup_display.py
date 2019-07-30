"""
* Special partial printing (e.g. only some keys of a dict) are easier done with a (recursive) preprocessing
* See http://ipython.readthedocs.io/en/stable/api/generated/IPython.core.formatters.html#IPython.core.formatters.BaseFormatter.for_type

>>> iprint(obj, 100)     # command to display more elements

TODO:
* HTML cannot react to changes of max_seq_length?
* display None and mappingproxy currently not working
"""

import datetime
import string
from collections import Counter, OrderedDict
from functools import partial
from itertools import islice
from operator import itemgetter

from IPython import get_ipython
from IPython.display import display

string_display_quote_char = '"'
string_display_quote_char2 = "'"
unquoted_chars = (
    set(string.ascii_letters)
    | set(string.digits)
    | set(" äöüÄÖÜß!?$%&/=`'+*~;_.-<>|@€")
)
quoted_boundary_chars = set(" ")

thousands_separator = "\u066c"  # Use "_" to make it work with Python?
ellipsis = "\u2026"
times_char = "\u00d7"
max_sort_length = 10000
max_seq_length = 20
MAX_NUM_PD_COLNAMES = 30

fore_col_grey = "\033[37m"
back_col_black = "\033[40m"
back_col_red = "\033[41m"
back_col_green = "\033[42m"
back_col_yellow = "\033[43m"
back_col_blue = "\033[44m"
back_col_pink = "\033[45m"
back_col_teal = "\033[46m"
back_col_grey = "\033[47m"
col_reset = "\033[0m"

html_grey = "<font color='grey'>"
html_close_color = "</font>"


def iprint(obj, max_seq_length=1000, **kwargs):
    text_printer = get_ipython().display_formatter.formatters["text/plain"]
    if max_seq_length is not None:
        kwargs["max_seq_length"] = max_seq_length
    orig_params = {key: getattr(text_printer, key) for key in kwargs.keys()}
    for key, val in kwargs.items():
        setattr(text_printer, key, val)

    display(obj)

    for key, orig_val in orig_params.items():
        setattr(text_printer, key, orig_val)


def _is_numpy_number(x):
    try:
        import numpy as np

        return np.issubdtype(type(x), np.number)
    except (ImportError, TypeError):
        return False


def _has_short_repr(
    obj
):  # these values can be printed on one line without breaks since they are small in repr
    return (
        isinstance(obj, (int, float, str, datetime.datetime, datetime.date))
        or obj is None
        or (isinstance(obj, (list, dict, set, frozenset)) and len(obj) == 0)
        or _is_numpy_number(  # just checking __len__ is bad since scipy.sparse has len but fails
            obj
        )
    )


def format_len(obj):
    return fore_col_grey + f"{format_int(len(obj))}\u2300" + col_reset


def ipy_prettyprint_int(obj, printer, is_cycle):
    printer.text(format_int(obj))


def format_int(obj):
    return f"{obj:,}".replace(",", thousands_separator)


def _quote_text(text):
    # some inconsistency left with backslashes and repr
    text_repr = repr(text)[1:-1]

    if {text[0], text[-1]} & quoted_boundary_chars or set(text) - unquoted_chars:
        if string_display_quote_char not in text:
            return string_display_quote_char + text_repr + string_display_quote_char
        elif string_display_quote_char2 not in text:
            return string_display_quote_char2 + text_repr + string_display_quote_char2
        else:
            return (
                string_display_quote_char
                + text_repr.replace(
                    string_display_quote_char, string_display_quote_char * 2
                )
                + string_display_quote_char
            )
    else:
        return text_repr


def ipy_prettyprint_iter(
    obj,
    printer,
    is_cycle,
    opentext,
    closetext,
    maxunnested=3,
    max_tail_length=3,
    empty_iter=None,
    sort=False,
):
    if is_cycle:
        printer.text(f"{opentext}\u2941{closetext}")
    elif len(obj) == 0:
        printer.text(opentext + closetext if empty_iter is None else empty_iter)
    elif (len(obj) <= maxunnested and all(_has_short_repr(o) for o in obj)) or len(
        obj
    ) == 1:
        printer.text(opentext + " ")

        if sort and len(obj) < max_sort_length:
            try:
                obj = sorted(obj)
            except TypeError:
                pass

        for i, subobj in enumerate(obj):
            if i > 0:
                printer.text(", ")
                printer.breakable()  # NOTE: this may create a line break, which is no indented by a group
            printer.pretty(subobj)
        printer.text(" " + closetext)
    else:
        printer.begin_group(2, f"{opentext}{format_len(obj)}")
        printer.break_()

        if sort and len(obj) < max_sort_length:
            try:
                obj = sorted(obj)
            except TypeError:
                pass

        for i, item in enumerate(islice(obj, printer.max_seq_length)):
            if i > 0:
                printer.break_()
            printer.pretty(item)
        if len(obj) > printer.max_seq_length:
            if not isinstance(obj, list):
                printer.break_()
                printer.text(ellipsis)
            elif len(obj) <= printer.max_seq_length + max_tail_length + 1:
                for item in obj[printer.max_seq_length :]:
                    printer.break_()
                    printer.pretty(item)
            else:
                printer.break_()
                printer.text(ellipsis)
                for item in obj[-max_tail_length:]:
                    printer.break_()
                    printer.pretty(item)

        printer.end_group(2)
        printer.break_()
        printer.text(closetext)


def ipy_prettyprint_dict(
    obj,
    printer,
    is_cycle,
    opentext="{",
    closetext="}",
    maxunnested=3,
    max_tail_length=3,
    sort=True,
):
    if is_cycle:
        printer.text(f"{opentext}\u2941{closetext}")
    elif len(obj) == 0:
        printer.text(opentext + closetext)
    elif (
        len(obj) <= maxunnested
        and all(_has_short_repr(k) and _has_short_repr(v) for k, v in obj.items())
    ) or len(obj) == 1:
        printer.text(opentext + " ")

        if sort and len(obj) <= max_sort_length:
            try:
                k_v_list = sorted(obj.items(), key=itemgetter(0))
            except TypeError:
                k_v_list = obj.items()
        else:
            k_v_list = obj.items()

        for i, (k, v) in enumerate(k_v_list):
            if i > 0:
                printer.text(", ")
                printer.breakable()
            printer.pretty(k)
            printer.text(" : ")
            printer.pretty(v)
        printer.text(" " + closetext)
    else:
        printer.begin_group(2, f"{opentext} {format_len(obj)}")
        printer.break_()

        if sort and len(obj) <= max_sort_length:
            try:
                k_v_list = sorted(obj.items(), key=itemgetter(0))
            except TypeError:
                k_v_list = obj.items()
        else:
            k_v_list = obj.items()

        for i, (key, val) in enumerate(islice(k_v_list, printer.max_seq_length)):
            if i > 0:
                printer.break_()
            printer.pretty(key)
            printer.text(" : ")
            with printer.group(4):
                printer.pretty(val)

        if len(k_v_list) > printer.max_seq_length:
            if not isinstance(k_v_list, list):
                printer.break_()
                printer.text(ellipsis)
            elif len(obj) <= printer.max_seq_length + max_tail_length + 1:
                for key, val in k_v_list[printer.max_seq_length :]:
                    printer.break_()
                    printer.pretty(key)
                    printer.text(" : ")
                    with printer.group(4):
                        printer.pretty(val)
            else:
                printer.break_()
                printer.text(ellipsis)
                for key, val in k_v_list[-max_tail_length:]:
                    printer.break_()
                    printer.pretty(key)
                    printer.text(" : ")
                    with printer.group(4):
                        printer.pretty(val)

        printer.end_group(2)
        printer.break_()
        printer.text(closetext)


def ipy_prettyprint_tuple(obj, printer, is_cycle):
    # TODO: may catch popular subtypes of tuple and display insuffcient information
    if hasattr(obj, "_fields"):  # namedtuple
        ipy_prettyprint_dict(
            obj._asdict(),
            printer,
            is_cycle,
            opentext=back_col_green + "(" + col_reset,
            closetext=back_col_green + ")" + col_reset,
            sort=False,
        )
        return

    # if type(obj)==tuple:   # Only do that for real tuples
    ipy_prettyprint_iter(
        obj,
        printer,
        is_cycle,
        opentext=back_col_grey + "(" + col_reset,
        closetext=back_col_grey + ")" + col_reset,
    )
    # else:
    #    display_pretty(obj)


def ipy_prettyprint_Counter(obj, printer, is_cycle, max_tail_length=3):
    if is_cycle:
        printer.text("Cntr \u2941")
    else:
        total = sum(obj.values())
        cumsum = 0
        with printer.group(2, f"Cntr {format_len(obj)}", ""):
            printer.break_()
            item_cnt_list = obj.most_common()
            for rank, (val, cnt) in enumerate(
                item_cnt_list[: printer.max_seq_length], 1
            ):
                cumsum += cnt
                if rank > 1:
                    printer.break_()
                printer.text(
                    fore_col_grey
                    + f"{rank:2} {f'{cumsum / total:3.0%}' if total > 0 else 'N.A'})  {format_int(cnt)}{times_char} "
                    + col_reset
                )
                with printer.group(2):
                    printer.pretty(val)
            if len(item_cnt_list) > printer.max_seq_length:
                if len(item_cnt_list) > printer.max_seq_length:
                    if not isinstance(item_cnt_list, list):
                        printer.break_()
                        printer.text(ellipsis)
                    elif (
                        len(item_cnt_list)
                        <= printer.max_seq_length + max_tail_length + 1
                    ):
                        for rank, (val, cnt) in enumerate(
                            item_cnt_list[printer.max_seq_length :],
                            printer.max_seq_length + 1,
                        ):
                            printer.break_()
                            printer.text(
                                fore_col_grey
                                + f"{rank:2})      {format_int(cnt)}{times_char} "
                                + col_reset
                            )
                            with printer.group(2):
                                printer.pretty(val)
                    else:
                        printer.break_()
                        printer.text(ellipsis)
                        for rank, (val, cnt) in enumerate(
                            item_cnt_list[-max_tail_length:],
                            len(item_cnt_list) - max_tail_length + 1,
                        ):
                            printer.break_()
                            printer.text(
                                fore_col_grey
                                + f"{rank:2})      {format_int(cnt)}{times_char} "
                                + col_reset
                            )
                            with printer.group(2):
                                printer.pretty(val)


def ipy_prettyprint_str(obj, printer, is_cycle, maxstrlen=200):
    if is_cycle:
        printer.text(string_display_quote_char + "\u2941" + string_display_quote_char)
    elif obj == "":
        printer.text(string_display_quote_char + string_display_quote_char)
    elif len(obj) > maxstrlen and len(printer.stack) > 1:
        printer.text(
            _quote_text(obj[: maxstrlen // 2] + ellipsis + obj[-maxstrlen // 2 :])
        )
    else:
        try:
            float(obj)
            printer.text(
                string_display_quote_char + str(obj) + string_display_quote_char
            )  # to disambiguate floats which are actually str
            # note that this does not catch something like IPs
        except ValueError:
            printer.text(_quote_text(obj))  # this is the default printer


def ipy_prettyprint_datetime(obj, printer, is_cycle, prefix):
    if obj.hour == 0 and obj.minute == 0 and obj.second == 0:
        printer.text(f"{prefix}.{obj:%Y-%m-%d}")
    elif obj.second == 0:
        printer.text(f"{prefix}.{obj:%Y-%m-%d-%H:%M}")
    else:
        printer.text(f"{prefix}.{obj:%Y-%m-%d-%H:%M:%S}")


def ipy_prettyprint_date(obj, printer, is_cycle):
    printer.text(f"d.{obj:%Y-%m-%d}")


def ipy_prettyprint_numpy_array(obj, printer, is_cycle):
    try:
        shape = obj.shape
    except AttributeError:
        shape = None

    try:
        dtype = obj.dtype.name
    except AttributeError:
        dtype = None

    try:
        nbytes = obj.nbytes
    except AttributeError:
        nbytes = None

    info = " ".join(
        [
            str(shape) if shape is not None else "",
            dtype if dtype is not None else "",
            format_int(nbytes) + "bytes" if nbytes is not None else "",
        ]
    )

    printer.text(fore_col_grey + info + col_reset)
    printer.break_()
    printer.text(repr(obj))


try:  # Pandas DataFrame Integer Formatter section
    import pandas as pd

    try:
        formatter_module = pd.io.formats.format
    except AttributeError:
        formatter_module = pd.formats.format  # before Pandas 0.20.0

    class _IntArrayFormatter(formatter_module.GenericArrayFormatter):
        def _format_strings(self):
            formatter = self.formatter or format_int
            fmt_values = [formatter(x) for x in self.values]
            return fmt_values

    formatter_module.IntArrayFormatter = _IntArrayFormatter
except ImportError:
    pass
except Exception as e:
    print(f"Failed setting Pandas settings ({e})")

try:  # Section for Text Printer
    text_printer = get_ipython().display_formatter.formatters["text/plain"]

    text_printer.for_type(
        list,
        partial(
            ipy_prettyprint_iter,
            opentext=back_col_blue + "[" + col_reset,
            closetext=back_col_blue + "]" + col_reset,
        ),
    )
    text_printer.for_type(tuple, ipy_prettyprint_tuple)
    text_printer.for_type(
        set,
        partial(
            ipy_prettyprint_iter,
            opentext=back_col_teal + "{" + col_reset,
            closetext=back_col_teal + "}" + col_reset,
            empty_iter="s{}",
            sort=True,
        ),
    )
    text_printer.for_type(
        frozenset,
        partial(
            ipy_prettyprint_iter,
            opentext=back_col_teal + "f{" + col_reset,
            closetext=back_col_teal + "}" + col_reset,
            sort=True,
        ),
    )
    text_printer.for_type(
        dict,
        partial(
            ipy_prettyprint_dict,
            opentext=back_col_yellow + "{" + col_reset,
            closetext=back_col_yellow + "}" + col_reset,
        ),
    )
    text_printer.for_type(str, ipy_prettyprint_str)
    text_printer.for_type(Counter, ipy_prettyprint_Counter)
    text_printer.for_type(int, ipy_prettyprint_int)
    text_printer.for_type(
        OrderedDict,
        lambda obj, printer, is_cycle: ipy_prettyprint_dict(
            obj,
            printer,
            is_cycle,
            opentext=back_col_yellow + "f{" + col_reset,
            closetext=back_col_yellow + "}" + col_reset,
            sort=False,
        ),
    )
    text_printer.for_type(datetime.date, ipy_prettyprint_date)
    text_printer.for_type(
        datetime.datetime, partial(ipy_prettyprint_datetime, prefix="dt")
    )
    # text_printer.for_type("builtins.mappingproxy",   # doesnt work
    #                      lambda obj, printer, is_cycle: ipy_prettyprint_dict(obj, printer, is_cycle, opentext="mapprox{",
    #                                                                          closetext="}", sort=False))
    text_printer.max_seq_length = max_seq_length

    try:
        import pandas as pd

        text_printer.for_type(
            pd.Timestamp, partial(ipy_prettyprint_datetime, prefix="pdTs")
        )
    except ImportError:
        pass

    try:
        import numpy as np

        text_printer.for_type(np.ndarray, ipy_prettyprint_numpy_array)

    except ImportError:
        pass

except Exception as e:
    print(f"Failed to set text/plain pretty printers ({e})")

try:  # Section for HTML printer
    html_printer = get_ipython().display_formatter.formatters["text/html"]

    try:
        from pyspark.sql.dataframe import DataFrame as SparkDataFrame
        import pandas as pd

        def ipy_html_dataframe(df, maxrows=max_seq_length):
            return pd.DataFrame(df.take(maxrows), columns=df.columns)._repr_html_()

        html_printer.for_type(SparkDataFrame, ipy_html_dataframe)
    except ImportError as e:
        pass
    except Exception as e:
        print(f"Failed to set text/html Spark DataFrame pretty display ({e})")

    try:
        import pandas as pd

        def ipy_html_pandasdataframe(df):
            import html
            
            def type_icon(dtype):
                if np.issubdtype(dtype, np.number):
                    return " &#x3253;"

                if np.issubdtype(dtype, np.datetime64):
                    return " &#128337;"

                return ""

            num_rows, num_cols = df.shape
            colnames = [
                (html.escape(str(colname), quote=False) + type_icon(coltype))
                for colname, coltype in df.dtypes.iteritems()
            ]

            if len(colnames) >= MAX_NUM_PD_COLNAMES:
                colnames = (
                    colnames[: MAX_NUM_PD_COLNAMES // 2]
                    + [ellipsis]
                    + colnames[-MAX_NUM_PD_COLNAMES // 2 :]
                )

            col_type_counts = df.dtypes.value_counts()

            return (
                html_grey
                + "<br>".join(
                    [
                        f"{format_int(num_rows)} rows; index: {df.index.dtype.name}",
                        f"{num_cols} cols: {', '.join(colnames)}",
                        "types: "
                        + ", ".join(
                            f"{count}{times_char} {dtype.name}"
                            for dtype, count in col_type_counts.items()
                        ),
                        df._repr_html_(),
                    ]
                )
                + html_close_color
            )

        html_printer.for_type(pd.DataFrame, ipy_html_pandasdataframe)

        def ipy_html_pandasseries(s):
            length = len(s)
            return "<br>".join(
                [
                    f"{html_grey}{length} elements of {s.dtype.name}",
                    f"{s.index.dtype.name} index{html_close_color}",
                    pd.DataFrame(s)._repr_html_(),
                ]
            )

        html_printer.for_type(pd.Series, ipy_html_pandasseries)
    except ImportError as e:
        pass
    except Exception as e:
        print(f"Failed to set text/html Pandas DataFrame pretty display ({e})")

except Exception as e:
    print(f"Failed setting IPython HTML settings ({e})")
