import pandas as pd

start_observation = pd.to_datetime("2011-01-01T00:00:00")
stop_observation  = pd.to_datetime("2020-12-31T23:59:59")
min_years_age = 18 # T.Y. modified
min_years_history = 1

COHORT_TY_START = pd.to_datetime("2010-01-01T00:00:00") # 2010
COHORT_TY_START_SUBSAMPLE = pd.to_datetime("2000-01-01T00:00:00") # 1980