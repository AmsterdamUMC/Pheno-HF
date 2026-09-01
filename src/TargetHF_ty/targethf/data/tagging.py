import pandas as pd
import numpy as np
import functools as ft

year_delta = pd.Timedelta(365.25, "days")

# Calculate the years between two timestamps and return as float
def year_diff(a, b):
    return (a-b)/year_delta

# Reduce a list of series to a single series according to a logical function
# functools.reduce is ~2x faster than (pd.concat --> .any/.all), ~1.5x faster than (np.any/.all --> pd.Series)
def reduce_series(reduce, series: pd.Series):
    return ft.reduce({"OR":np.logical_or, 
                     "AND":np.logical_and}[reduce], series, False)

# Nested AND/OR pattern matching
def boolex(series: pd.Series, pattern):
    # Recursively iterate over the defined pattern
    def recursive_regex(series, pattern):
        # If a string is supplied, apply this as a regex pattern
        if isinstance(pattern, str):
            matches = series.str.contains(pattern, na=False, case=False)
        # If a list is supplied, step in and apply supplied operator to individual results
        elif isinstance(pattern, list):
            operator, pattern_list = pattern[0], pattern[1:]
            if operator in ("AND", "OR"):
                matches = reduce_series(operator, [recursive_regex(series, p) for p in pattern_list])
            elif operator == "NOT" and len(pattern_list)==1:
                matches = ~recursive_regex(series, pattern_list[-1])
            else:
                raise ValueError("Supplied list has an incorrect operator", pattern)
        # Catch incorrect structures
        else:
            raise ValueError("Supplied pattern is not a list or string", pattern)
        return matches

    # Transform a pattern to a humanly readable format
    def pattern_to_string(pattern):
        if isinstance(pattern, list):
            operator, pattern_list = pattern[0], pattern[1:]
            return "(" + " {} ".format(operator).join([pattern_to_string(p) for p in pattern_list]) + ")"
        elif isinstance(pattern, str):
            return "\"{}\"".format(pattern)
    return recursive_regex(series, pattern), None

# Apply boolean regex to multiple columns (results combined across columns on "any"/OR basis)
def multi_boolex(table: pd.DataFrame, pattern, log_hooks=None, log_path=None):
    log = []
    pattern_matches = []
    for col in list(table):
        # Apply the (nested) boolean regex pattern to this column 
        matches, column_log = boolex(table[col], pattern)
        pattern_matches.append(matches)
    # Combine columnar matches to one series and return
    return reduce_series("OR", pattern_matches)

# Simple but efficient matching for categorical ICPC columns. If pattern is a list then results are combined on "any"/OR basis
# When DataFrame is supplied: apply categorical matching to multiple columns (results combined across columns on "any"/OR basis)
def icpc_match(data, pattern):
    if isinstance(data, pd.DataFrame):
        return reduce_series("OR", [icpc_match(data[col], pattern) for col in list(data)])
    elif isinstance(data, pd.Series):
        # Look for main and subcategories and apply OR operator to the results of all
        def cat_match(data, icpc):
            icpc_codes = data.cat.categories
            icpc_variants = icpc_codes[icpc_codes.str.startswith(icpc)]
            return reduce_series("OR", [data==variant for variant in icpc_variants])
        
        # If a string is supplied, apply cat_match
        if isinstance(pattern, str):
            return cat_match(data, pattern)
        # If a list is supplied, apply cat_match to each string and apply OR operator to individual results
        elif isinstance(pattern, list):
            return reduce_series("OR", [cat_match(data, p) for p in pattern])
        # Catch incorrect structures
        else:
            raise ValueError("Supplied pattern is not a list or string", pattern)
