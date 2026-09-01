import sys
#sys.path.append('..')

# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
logger = get_default_logger_fn(__file__)

# get namespace
from namespaces import get_ns_name
import namespaces
ns_name = get_ns_name(__file__)
ns = getattr(namespaces, ns_name)

#ns_name = [ x for x in dir(namespaces) if x == f'ns_{__file__[:-3]}'][0]
#ns = getattr(namespaces, ns_name)
# parse commandline args
cmd_args = parse_commandline_args(verbose=True, required_extra_args=ns.required_extra_args)

IS_DEBUG = cmd_args["IS_DEBUG"]
SUBSAMPLE_DATA = cmd_args["SUBSAMPLE_DATA"]
script_params = { k : cmd_args[k] for k in ns.required_extra_args}
ns.__dict__.update(script_params)

check_if_debugging(IS_DEBUG)
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
from try_utils import *
from constants import *
import numpy as np
import random
pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
start_time = logger("Start running...")
# Boilerplate end


import pandas as pd
from pathlib import Path
import constants
from TargetHF_ty.targethf.data.tagging import year_delta, year_diff
from TargetHF_ty.targethf.definitions import icpc_def, cohort_def


# Local naming note: this script's `data_dir`/`pqt_dir` refer to the older
# extract tree; `constants.data_dir`/`constants.pqt_dir` (the newer extract)
# are used explicitly below where both trees are read.
data_dir = old_data_dir

# Define paths
pqt_dir = data_dir/"parquet"
cht_dir = data_dir/"cohort_A-ICPC"

cht_dir.mkdir(exist_ok=True)


# Load tables
# T.Y. -  to work with all patients we need to use patients_ANH.parquet , patients_AHA.parquet 
# Q1: are they different structurally than persons.parquet?
# A1:quite different.... 
#  persons.parquet has extra features like t_min, t_max, t_atrial_fibrilation, etc... 
# Q2: how did persons.parquet get derived?
# A2: 03_TableDistillation.py
#persons  = pd.read_parquet(pqt_dir/"persons.parquet")

nrows_pats = 10000 if SUBSAMPLE_DATA else None
nrows_eps = 5000000 if SUBSAMPLE_DATA else None
start_time_cohort = cohort_def.COHORT_TY_START
if SUBSAMPLE_DATA:
    start_time_cohort = cohort_def.COHORT_TY_START_SUBSAMPLE

for db_prefix in ['AHA', 'ANH']:
    persons  = try_read_pd_df(constants.pqt_dir/f"persons_ty_{db_prefix}{subsampled_str}.parquet", nrows=nrows_pats)
    logger(f"Read {len(persons)} unique patients")
    journals = try_read_pd_df(constants.pqt_dir/f"journals_{db_prefix}.csv", dtypes=CSV_DTYPES['journals'], nrows=nrows_eps)
        
    # ## Distillation  and tagging (contd.)
    # Add date of birth to journals
    journals = journals.join(persons["t_birth"], how="inner", on="person_id") 
    # Filter out journals where the person is less than min_years_age years old
    # T.Y. - doing so would remove journals we are interested in?, workaround: set min age to 18
    # this keeps only journals where the date of the journal is greater than the date at which the patient turned {min_years_age}
    logger(f"Remove journals where patient was below minimum age ({cohort_def.min_years_age}) at point of contact")
    journals['journal_datetime'] = pd.to_datetime(journals["journal_datetime"])
    journals = journals[journals["journal_datetime"] >= (journals["t_birth"]+cohort_def.min_years_age*year_delta)]
    logger(f"Left with {len(journals)} journal records")

    logger("use first journal after start period as new t_0 ")
    logger(f"{nrow(persons)}: N patients in cohort")
    persons["t_0"] = journals[journals["journal_datetime"] >= start_time_cohort].groupby("person_id", sort=False).apply(lambda x: x.loc[x["journal_datetime"].idxmin()])["journal_datetime"]
    persons = persons.dropna(subset = ['t_0'])
    logger("Define time zero (t_0) as first  journal of person after start_time_cohort")
    logger(f"Drop patients with no journal after {start_time_cohort}")
    logger(f"{nrow(persons)}: left with n patients")

    logger("Determine time spans (in years)")
    persons["years_history"]  = year_diff(persons["t_0"],   persons["t_min"])
    persons["years_followup"] = year_diff(persons["t_max"], persons["t_0"])
    persons["years_age"]      = year_diff(persons["t_0"],   persons["t_birth"])

    extra_condition_factors = ['AF', 'VHD', 'HF']
    extra_condition_factors = extra_condition_factors + [f"text_{x}" for x in extra_condition_factors]
    tags_epi_to_person = list(icpc_def.risk_factors.keys()) + ['icpc_HF'] + extra_condition_factors

    for risk_factor in tags_epi_to_person:
        persons["t_"+risk_factor] = pd.to_datetime(persons["t_"+risk_factor])

    # Determine risk factors occuring before t_0
    for risk_factor in tags_epi_to_person:
        # Time span since diagnosis (in years)
        persons["years_since_"+risk_factor] = year_diff(persons["t_0"], persons["t_"+risk_factor])
        # Occurs before t_0 if span is positive
        persons[risk_factor] = (~persons["t_"+risk_factor].isnull()) # i.e., did they ever get this diagnosis?

    cohort = persons.loc[persons["t_0"].notna()].copy() # seems redundant?
    logger(f"After minimum age, and start period filtering :: cohort from {db_prefix} left with {len(cohort)} persons")
    logger(f"Persons removed due to no jounrals present = {nrow(persons) - len(cohort)} persons")
    # Non-negative follow-up
    if not SUBSAMPLE_DATA: # dont enforce this for subsampling, otherwise cant get any positive cases to compare
        cohort = cohort[cohort["t_max"] > cohort["t_0"]]
        logger(f"After non-negative follow-up filtering :: cohort from {db_prefix} left with {len(cohort)}: records")

    # Read verified file
    # ver_hf_episodes = pd.read_csv(cht_dir/"verification.csv",
    #                             index_col=["person_id", "practice_id", "episode_id", "import_id"],
    #                             parse_dates=["episode_start_date"])

    ver_hf_episodes = pd.read_csv(constants.data_dir/f"cohort/verification_icpc_{db_prefix}.csv",
                                        index_col=["person_id"], parse_dates=["episode_start_date"])

    # remove rows where no HF was determined after adjudication (i.e., remove FPs)
    logger("try_table(ver_hf_episodes[NO_HF])")
    logger(try_table(ver_hf_episodes['NO_HF']))
    ver_hf_episodes = ver_hf_episodes[pd.isnull(ver_hf_episodes['NO_HF'])]
    ver_hf_episodes = ver_hf_episodes[ver_hf_episodes.index.get_level_values("person_id").isin(cohort.index)]
    logger(f"adjudicated records from cohort have {len(ver_hf_episodes)} TPs")


    # Determine date of first heart failure episode
    cohort["t_hf"] = ver_hf_episodes["episode_start_date"].groupby("person_id").min()
    logger(f"{len(cohort)} persons in cohort:")

    logger(f"{sum([ not pd.isnull(x)  for x in cohort['t_hf']])}: HF pos cases (all)")
    logger(f"{nrow(cohort[(cohort['t_hf'] <= cohort['t_0'])])}: HF pos cases before t_0 ")


    # ## Storing
    try_save_parquet(cht_dir/f"cohort_ty_{db_prefix}{subsampled_str}.parquet", cohort)


logger("Done")
print("DONE")

