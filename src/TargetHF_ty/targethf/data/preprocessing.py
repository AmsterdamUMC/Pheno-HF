import html
import pandas as pd
import numpy as np
import re

# Efficiently NULL certain categories from a table
def XXremove_categories(table, categories):
    for col, cat in categories.items():
        table[col] = table[col].cat.remove_categories(cat)

# Reduce datetime accuracy: encode as days/reset time to 00:00:00
def normalize_time(table):
    date_cols = table.select_dtypes("datetime").columns
    for col in date_cols:
        table[col] = table[col].dt.normalize()
    return table

# Fix the encoding issues likely introduced at ZIS-export-level and exacerbated by SQL Server 2016's lack of UTF-8
def fix_hagdb_encoding(table):
    string_cols = table.select_dtypes("string").columns
    # Fix (somehow doubly) escaped characters
    table[string_cols] = table[string_cols].apply(lambda x: html.unescape(html.unescape(x)))
    return table

# Remove categories not complying to a regex pattern
def categorical_regex(series, regex):
    assert series.dtype.name == "category", "Pass the code column as categorical"
    codes = series.cat.categories
    # Identify invalid formats
    invalid_codes = codes[~codes.str.fullmatch(regex)]
    # Filter out
    return series.cat.remove_categories(invalid_codes)

# ICPC codes come in the form
# ABB.CC
# With the following levels
# A: Chapter
# B: Complaint/diagnosis
# C: Specification (optional)
# Warning: This function will take the first pattern match if multiple are available in that line
def parse_ICPC(text, specify_sublevel=False):
    icpc_pattern = r"[ABDFHKLNPRSTUWXYZ](?!00)[0-9]{2}(?:\.(?!00)[0-9]{2})?"
    # Remove trailing/leading spaces
    text = text.strip()
    # Find the first character set that conforms to ICPC formatting
    match = re.search(icpc_pattern, text, re.IGNORECASE)

    # Match found
    if match:
        # Extract matching string
        match = match.group()
        # Make uppercase
        match = match.upper()
        # Optionally add ".00" sublevel and return 
        if specify_sublevel and (len(match)==3):
            return match + ".00"
        else:
            return match
    # No match found
    else:
        return None

# ATC codes come in the form
# ABBCDEE
# With the following levels
# A: Anatomical group
# B: Therapeutic subgroup 
# C: Therapeutic/pharmacological subgroup 
# D: Chemical/therapeutic/pharmacological subgroup
# E: Chemical substance
def parse_ATC(text):
    match = re.search(r"[A-Z][0-9]{2}[A-Z]{2}(?:(?!00)[0-9]{2})?", text.strip().upper(), re.IGNORECASE)
    return match.group() if match else None