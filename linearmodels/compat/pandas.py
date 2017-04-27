from pandas.core.common import is_string_like

try:
    from pandas.api.types import (is_numeric_dtype, is_categorical,
                                  is_string_dtype, is_categorical_dtype,
                                  is_datetime64_any_dtype)
except ImportError:  # pragma: no cover
    from pandas.core.common import (is_string_dtype, is_numeric_dtype,
                                    is_categorical, is_categorical_dtype,
                                    is_datetime64_any_dtype)

__all__ = ['is_string_dtype', 'is_numeric_dtype', 'is_categorical',
           'is_string_like', 'is_categorical_dtype', 'is_datetime64_any_dtype']
