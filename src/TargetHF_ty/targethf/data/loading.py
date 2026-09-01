import pandas as pd
import numpy as np
from pathlib import Path

# Load table based on definitions
def read_defined_csv(csv_path, json_path, sep="|", encoding="latin1", drop_duplicates=False, **read_args):

    # Read table definitions from JSON file
    defs = pd.read_json(json_path, orient='index')
    
    # Extract non-date dtypes to dictionary
    dtypes = defs.loc[defs["dtype"]!="datetime", "dtype"].to_dict() 
    
    # Read table efficiently
    table = pd.read_csv(csv_path, sep=sep,
                        encoding=encoding,      # latin1 for SQL Server 2016 (not ASCII or UTF8)
                        engine='c',             # Faster parser than "python"
                        usecols=defs.index,     # Save memory by dropping unnecessary cols
                        dtype=dtypes,           # More memory efficient by directly encoding as categorical etc.
                        skipinitialspace=True,  # Strip leading spaces from strings
                        **read_args)            # Allows for specifying e.g. nrows for subset selection for testing
    # Rename columns
    table = table.rename(columns=defs["name"].to_dict())

    # Drop full-row duplicates (all fields matching)
    if drop_duplicates:
        table = table.drop_duplicates(ignore_index=True)
    
    # Set index and check for index duplicates
    if "index" in defs.columns:
        index_cols = list(defs.loc[defs["index"]==True, "name"])
        try:
            table = table.set_index(index_cols, verify_integrity=True)
        except ValueError:
            print(csv_path, ": could not set index on", index_cols, ", please check for duplicates")

    # Parse dates. Done seperately to enable coercing errors
    for date_col in list(defs.loc[defs["dtype"]=="datetime", "name"]):
        table[date_col] = pd.to_datetime(table[date_col],
                            errors="coerce",              # Silently NULL invalid entries (e.g. out of Datetime64 range)
                            infer_datetime_format=True)   # Try to infer format from first non-NaN element

    return table
