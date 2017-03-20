try:
    from pandas.api.types import (is_string_like, is_numeric_dtype,
                                  is_categorical, is_string_dtype,
                                  is_categorical_dtype)
except ImportError:
    from pandas.core.common import (is_string_dtype, is_string_like,
                                    is_numeric_dtype, is_categorical,
                                    is_categorical_dtype)
