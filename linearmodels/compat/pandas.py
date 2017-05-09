from pandas.core.common import is_string_like
from pandas.util.testing import assert_panel_equal

try:
    from pandas.api.types import (is_numeric_dtype, is_categorical,
                                  is_string_dtype, is_categorical_dtype,
                                  is_datetime64_any_dtype)
except ImportError:  # pragma: no cover
    from pandas.core.common import (is_string_dtype, is_numeric_dtype,
                                    is_categorical, is_categorical_dtype,
                                    is_datetime64_any_dtype)

try:
    from pandas.testing import assert_frame_equal, assert_series_equal
except ImportError:
    from pandas.util.testing import assert_frame_equal, assert_series_equal


__all__ = ['is_string_dtype', 'is_numeric_dtype', 'is_categorical',
           'is_string_like', 'is_categorical_dtype', 'is_datetime64_any_dtype',
           'assert_frame_equal', 'assert_series_equal', 'assert_panel_equal']
