# Boilerplate start
print('''WHAT THIS 'SCRIPT DOES
adds episosde/journal level tags to patients. Sets t_max, t_min for each patient based on journal times
''')
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
logger = get_default_logger_fn(__file__)
IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
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
from TargetHF_ty.targethf.data.distillation import groupby_notna
from TargetHF_ty.targethf.definitions import icpc_def

nrows_pats = 10000 if SUBSAMPLE_DATA else None
nrows_eps = 5000000 if SUBSAMPLE_DATA else None
# Load tables
# patients.parquet dedup: 1,385,597 rows -> 137,935 after dedup on patient_id (vs. 145,206 after
# dedup on the patient+practice composite key); we go with patient_id-only dedup (see OPEN_QUESTIONS.md)

conds_tags_mapping = {
    "HF" : {"icpc_col_nm" : "icpc_HF", "text_def_nm": "heart_failure" },
    "AF" : {"icpc_col_nm" : "atrial_fibrillation", "text_def_nm": "afib"},
    "VHD" : {"icpc_col_nm" : "valvular_heart_disease", "text_def_nm": "valvhd"}
}
for db_prefix in ['AHA', 'ANH']:
    logger("Load persons and journals")
    patients = try_read_pd_df(pqt_dir/f"patients_{db_prefix}_01Tagging_ty{subsampled_str}.parquet", nrows=nrows_pats)
    journals = try_read_pd_df(pqt_dir/f"journals_{db_prefix}.csv", dtypes=CSV_DTYPES['journals'], nrows=nrows_eps)
    
    # ## Grouping
    # Sort (take care NOT to sort anywhere else) based on import_id (primarily) and registration date (secondarily)
    patients = patients.sort_values(["person_id", "import_id", "reg_date"])

    # ### Persons

    # Generate Person table (patient-across-practices)
    #patients2 = add_composite_key(patients, unit="patient", dedup=T)
    persons = pd.DataFrame(patients["person_id"].drop_duplicates())
    persons = persons.set_index("person_id")
    logger("Persons deduplicated ok")
    if SUBSAMPLE_DATA:
        journals = journals[journals['person_id'].isin((patients.index))]

    logger("Determine t_min & t_max")
    journals_grouped = journals.groupby("person_id", sort=False)
    del journals
    persons["t_min"] = journals_grouped["journal_datetime"].min()
    persons["t_max"] = journals_grouped["journal_datetime"].max()
    del journals_grouped

    persons["t_min"] = pd.to_datetime(persons["t_min"])
    persons["t_max"] = pd.to_datetime(persons["t_max"])

    logger("Compute Demographics (deceased, t_dereg, t_birth, sex, t_death)")
    patients_grouped    = patients.groupby("person_id", sort=False)
    # T.Y. "deceased" not found... you may need to run 01_Tagging.py before this then...
    persons["deceased"] = patients_grouped["deceased"].any()
    persons["t_dereg"]  = patients_grouped.last()["dereg_date"]
    del patients_grouped
    persons["t_birth"]  = groupby_notna(patients, "person_id", "birth_date", sort=False).last()
    persons["sex"]      = groupby_notna(patients, "person_id", "sex",        sort=False).last().astype("category")
    del patients
    persons["t_death"]  = persons.loc[persons["deceased"], ["t_max", "t_dereg"]].max(axis="columns")
    n = nrow(persons)
    logger("Read episodes...")
    episodes = try_read_pd_df(pqt_dir/f"ty/episodes_{db_prefix}{subsampled_str}.parquet", nrows=nrows_eps)
    if SUBSAMPLE_DATA:
        episodes = episodes[episodes['person_id'].isin((persons.index))]
    # Determine time of first risk factor episode
    logger("Add episode level risk factors to person-level...")
    extra_condition_factors = ['AF', 'VHD', 'HF']
    extra_condition_factors = extra_condition_factors + [f"text_{x}" for x in extra_condition_factors]
    tags_epi_to_person = list(icpc_def.risk_factors.keys()) + ['icpc_HF'] + extra_condition_factors
    for risk_factor in tags_epi_to_person:
        t_col = f"t_{risk_factor}"
        persons[t_col] = None
        risk_pos_pats = episodes[episodes[risk_factor] == 1].groupby("person_id", sort=False)["episode_start_date"].min()
        
        
        persons[t_col] = [risk_pos_pats.loc[i] if i in risk_pos_pats else None for i in persons.index]
        is_text_col = 't_text_' in t_col
        npos = len(risk_pos_pats)
        percPos = 100*npos/nrow(persons) 
        if is_text_col:
            logger("*******")
            non_text_col = "t_" + conds_tags_mapping[t_col.replace('t_text_', '')]['icpc_col_nm']
            icpcPos = persons[~persons[non_text_col].isna()].index
            onlyTextPosCount = len(try_sd(risk_pos_pats.index, icpcPos))
            logger(f"{t_col} AND NOT {non_text_col} = {100*onlyTextPosCount/n:.2f}% ({onlyTextPosCount}/{n})")
            logger("*******")
        logger(f"{t_col} = {percPos:.2f}% ({npos}/{n})")


    try_save_parquet(pqt_dir/f"persons_ty_{db_prefix}{subsampled_str}.parquet", persons)
logger("Done")
print("DONE")
