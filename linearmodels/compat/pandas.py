from pandas.util.testing import assert_panel_equal

try:
    from pandas.api.types import (is_numeric_dtype, is_categorical,
                                  is_string_dtype, is_categorical_dtype,
                                  is_datetime64_any_dtype)

    # From pandas 0.20.1
    def is_string_like(obj):
        """
        Check if the object is a string.

        Parameters
        ----------
        obj : The object to check.

        Returns
        -------
        is_str_like : bool
            Whether `obj` is a string or not.
        """
        return isinstance(obj, str)

except ImportError:  # pragma: no cover
    from pandas.core.common import (is_string_dtype, is_numeric_dtype,
                                    is_categorical, is_categorical_dtype,
                                    is_datetime64_any_dtype, is_string_like)

try:
    from pandas.testing import assert_frame_equal, assert_series_equal
except ImportError:
    from pandas.util.testing import assert_frame_equal, assert_series_equal

__all__ = ['is_string_dtype', 'is_numeric_dtype', 'is_categorical',
           'is_string_like', 'is_categorical_dtype', 'is_datetime64_any_dtype',
           'assert_frame_equal', 'assert_series_equal', 'assert_panel_equal']
