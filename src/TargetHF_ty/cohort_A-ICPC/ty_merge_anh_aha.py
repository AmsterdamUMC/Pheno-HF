# Merges the AHA/ANH cohorts into cohort_ty_merged_{target_condition}.parquet,
# consumed downstream by TargetHF_ty/cohort_A-ICPC/02_SimpleSurvivalAnalysis.py
# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
logger = get_default_logger_fn(__file__)
# get namespace
from namespaces import get_ns_name
import namespaces
ns_name = get_ns_name(__file__)
ns = getattr(namespaces, ns_name)
# parse commandline args
cmd_args = parse_commandline_args(verbose=True, required_extra_args=ns.required_extra_args)

IS_DEBUG = cmd_args["IS_DEBUG"]
SUBSAMPLE_DATA = cmd_args["SUBSAMPLE_DATA"]
script_params = { k : cmd_args[k] for k in ns.required_extra_args }
ns.__dict__.update(script_params)

check_if_debugging(IS_DEBUG)
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
from constants import *
from try_utils import *
import numpy as np
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
start_time = logger("Start running...")
# Boilerplate end

USE_ONLY_AFPOS = ns.target_condition == "AF"
USE_ONLY_VHDPOS = ns.target_condition == "VHD"
USE_ONLY_HFPOS = ns.target_condition == "HF"
assert USE_ONLY_AFPOS + USE_ONLY_VHDPOS + USE_ONLY_HFPOS == 1

outcome_column =  {
    "AF" : "atrial_fibrillation",
    "VHD" : "valvular_heart_disease",
    "HF" : "heart_failure"
    }[ns.target_condition]

from TargetHF_ty.targethf.data.tagging import year_diff
import pandas as pd


merged_dfs = None


for db_prefix in ['AHA', 'ANH']:
    c_df = pd.read_parquet(old_data_dir/f"cohort_A-ICPC/cohort_ty_{db_prefix}{subsampled_str}.parquet")
    c_df['heart_failure'] = ~c_df['t_hf'].isna()
    c_df = c_df[c_df[outcome_column] == T]

    # Convert variables to regression-pleasing form
    c_df["male"]        = (c_df["sex"]=="M")
    c_df["decades_age"] = c_df["years_age"]/10

    # Extract censoring/time-to-event (can't be zero, so extremely small)
    c_df["time_to_event"] = year_diff(c_df[["t_hf", "t_max"]].min(axis="columns"), c_df["t_0"])/10
    c_df["time_to_event"] = c_df["time_to_event"].replace(0, 1e-9)
    c_df["event"]         = c_df["t_hf"].notna()


    journals = try_read_pd_df(pqt_dir/f"journals_{db_prefix}.csv", dtypes=CSV_DTYPES['journals'])
    # Only journals from our c_df
    logger(f"Journals in dataset: {len(journals)}")
    journals = journals.join(c_df["t_0"], how="inner", on="person_id")
    logger(f"Journals in c_df:  {len(journals)}")
    journals = journals[journals["journal_datetime"]<journals["t_0"]].drop(columns=["t_0"])
    logger(f"Journals before t0:  {len(journals)}")

    # Save count
    c_df = c_df.join(journals.groupby("person_id").count(), how="left", on="person_id")
    del journals


    merged_dfs = pd.concat([merged_dfs, c_df])

logger(f"Merged anh and aha person cohorts has {len(merged_dfs)} records, with {sum([not pd.isnull(x) for x in merged_dfs['t_hf']])} positive HF cases")
merged_dfs.to_parquet(old_data_dir/f"cohort_A-ICPC/cohort_ty_merged_{ns.target_condition}{subsampled_str}.parquet")