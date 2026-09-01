#!/usr/bin/env python
# coding: utf-8
print('''WHAT THIS 'SCRIPT DOES
adds deceased flag to persons parquet
adds risk factors on episode level (from icpc codes)
adds HF/AF/VHD tags on episode level (from regexes)
''')
# Boilerplate start
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
from TargetHF_ty.targethf.data.tagging import multi_boolex, icpc_match
from TargetHF_ty.targethf.definitions import icpc_def, text_def
import sys

text_def = module_to_dict(text_def)

nrows_pats = 10000 if SUBSAMPLE_DATA else None
nrows_eps = 500000 if SUBSAMPLE_DATA else None
for db_prefix in ['AHA', 'ANH']:
    #:: Patients
    patients = try_read_pd_df(pqt_dir/f"patients_{db_prefix}.parquet", nrows=nrows_pats)
    # Identify records with indication of death
    patients["deceased"]  = (patients["dereg_cause"]=="O")
    # Store and cleanup
    try_save_parquet(pqt_dir/f"patients_{db_prefix}_01Tagging_ty{subsampled_str}.parquet", patients)
    del patients

    #:: Episodes
    episodes = try_read_pd_df(pqt_dir/f"episodes_{db_prefix}.csv", nrows=nrows_eps)
    # Columns for searching
    episodes_text = episodes[["episode_description"]]
    episodes_icpc = episodes[["icpc_episode"]].astype('category')


    #: Risk factors
    for risk_factor, pattern in icpc_def.risk_factors.items():
        logger(f"Risk factor for episdoes [{risk_factor} => {pattern}]")
        episodes[risk_factor] = icpc_match(episodes_icpc, pattern)
    


    episodes["icpc_HF"]  = icpc_match(episodes_icpc,   icpc_def.heart_failure)
    logger("icpc_HF done")
    
    conds_tags_mapping = {
        "HF" : {"icpc_col_nm" : "icpc_HF", "text_def_nm": "heart_failure" },
        "AF" : {"icpc_col_nm" : "atrial_fibrillation", "text_def_nm": "afib"},
        "VHD" : {"icpc_col_nm" : "valvular_heart_disease", "text_def_nm": "valvhd"}
    }

    # --- inline regex tests (see OPEN_QUESTIONS.md: candidate to move to a proper test file) ---
    def test_regex(x, c):
        t_df = episodes_text.head(1)
        t_df.iloc[0,0] = x
        return multi_boolex(t_df, text_def[conds_tags_mapping[c]['text_def_nm']]).to_dict()[0]

    def run_tests():
        test_regex_HF = lambda x: test_regex(x, "HF")
        assert test_regex_HF("hartfalen") == T
        assert test_regex_HF("Thartfalen") == T
        assert test_regex_HF("hey dit is Hartfalen") == T

        assert test_regex_HF("falen") == F
        assert test_regex_HF("angst voor hartfalen") == F
        assert test_regex_HF("a angst voor hartfalen") == F
        assert test_regex_HF("geen hartfalen") == F
        assert test_regex_HF("Angst voor bananen") == F

        assert test_regex_HF("hartfalen") == T
        assert test_regex_HF("hey dit is Hartfalen") == T

        assert test_regex_HF("angst voor hartfalen") == F
        assert test_regex_HF("a angst voor hartfalen") == F
        assert test_regex_HF("geen hartfalen") == F
        assert test_regex_HF("Angst voor bananen") == F

        test_regex_AF = lambda x: test_regex(x, "AF")
        assert test_regex_AF("atrium f") == T
        assert test_regex_AF("atriumf") == T
        assert test_regex_AF("atrium   f") == T
        assert test_regex_AF("Natrium f") == T
        assert test_regex_AF("fibr") == T
        assert test_regex_AF("fibra") == T
        assert test_regex_AF("as980ry8932w04b6 jto04fibrsed opter09m-oyt") == T
        assert test_regex_AF("Angst  afibrali") == T


        assert test_regex_AF("atrium  lepel f") == F
        assert test_regex_AF("Angst voor fibr") == F
        assert test_regex_AF("fibro") == F

        test_regex_VHD = lambda x: test_regex(x, "VHD")
        assert test_regex_VHD("aklep") == T
        assert test_regex_VHD("i like valv") == T
        assert test_regex_VHD("Mitrialis") == T
        assert test_regex_VHD("aortak") == T
        assert test_regex_VHD("aorta   koo") == T
        assert test_regex_VHD("insufficient vespine gas") == T
        assert test_regex_VHD("stenorka") == T

        assert test_regex_VHD("Angst voor stenorka") == F
        assert test_regex_VHD("stent") == F

    run_tests()
    # END TEST REGEXES

    for cond, mapping in conds_tags_mapping.items():
        text_col_nm = f"text_{cond}"
        icpc_col_nm = mapping['icpc_col_nm']
        text_def_nm = mapping['text_def_nm']
        logger(f"************* {cond} ************* : {text_col_nm}  {icpc_col_nm} [risk factor from ICPCs, already done]")
        episodes[text_col_nm] = multi_boolex(episodes_text, text_def[text_def_nm])
        tag_tbl = try_table(episodes[text_col_nm]).to_dict()
        npos = tag_tbl[T] if T in tag_tbl else 0
        prop_pos = npos / nrow(episodes)
        logger(f"{text_col_nm} done ({prop_pos*100:.2f}%)")
        logger(f"episode.{cond} = ? {text_col_nm} OR {icpc_col_nm} ")
        episodes[cond] = episodes[[text_col_nm, icpc_col_nm]].any(axis="columns")



    # Store and cleanup
    # quick sanity check, inspect what episodes are being marked for AF/VHD
    # xxx = episodes[episodes['text_VHD'] == 1]
    # xxx = xxx[ xxx['valvular_heart_disease'] == 0]
    # xxx = xxx[["person_id", risk_factor, "episode_start_date", "episode_description", "icpc_episode"]]
    # vals(xxx.head(10)['episode_description'])
    try_save_parquet(pqt_dir/f"ty/episodes_{db_prefix}{subsampled_str}.parquet", episodes)
    del episodes
logger("done")
print("DONE")
