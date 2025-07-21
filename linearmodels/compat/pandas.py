from packaging.version import parse
import pandas as pd

PANDAS_VERSION = parse(pd.__version__)
PD_GTE_21 = not (PANDAS_VERSION <= parse("2.0.99"))
PD_LT_22 = PANDAS_VERSION <= parse("2.1.99")
ANNUAL_FREQ = "A-DEC" if PD_LT_22 else "YE-DEC"

__all__ = ["ANNUAL_FREQ", "PD_GTE_21", "PD_LT_22"]
