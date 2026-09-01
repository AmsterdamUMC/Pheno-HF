# Groupby on a certain column without taking NULL values in another column into account
def groupby_notna(table, by, col, **kwargs):
    return table[table[col].notna()].groupby(by, **kwargs)[col]