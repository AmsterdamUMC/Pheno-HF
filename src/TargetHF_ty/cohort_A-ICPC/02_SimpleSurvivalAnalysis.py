#!/usr/bin/env python
# coding: utf-8



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
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
start_time = logger("Start running...")
# Boilerplate end

import pandas as pd

import scipy.stats as st
import sys
#sys.path.append('..')
# pip install lifelines scikit-survival xgboost
import matplotlib.pyplot as plt

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.calibration import survival_probability_calibration
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw, concordance_index_censored

from sklearn import model_selection
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBRegressor
from pathlib import Path
from TargetHF_ty.targethf.data.tagging import year_diff
from TargetHF_ty.targethf.definitions import icpc_def
from TargetHF_ty.targethf.definitions.cohort_def import COHORT_TY_START
from itertools import repeat
from multiprocessing import Pool


# ## Loading
data_dir = old_data_dir
# Define paths
pqt_dir = data_dir/"parquet"
cht_dir = data_dir/"cohort_A-ICPC"

cohort = pd.read_parquet(cht_dir/"cohort_ty_merged.parquet")


cohort["journal_datetime"] = cohort["journal_datetime"].fillna(0)
cohort["avg_use"] = (cohort["journal_datetime"]/cohort["years_history"]).fillna(0)

risk_factors = list(icpc_def.risk_factors.keys())
predictors = ["decades_age", "male"] + risk_factors

cohort = cohort[predictors+["time_to_event", "event"]]

hrs = calc_TARGETHF_scores(cohort)
cohort['target_hf_score'] = hrs
# 30.08.2024. T.Y. update from Jelle:
#  we dont want simple event within 2 years like you did initially.. doing so looses a lot of the details about when exactly the event occurred
# e.g., patient 1 had hf diagnosis after 1 month, patient 2 had hf dianogsis after 23 months
# if we use event_2y binary variable, that will give both patients a value of 1 (true)
# better than this is to just sum-up all the time to events, and divide by the total number of follow-up years of all patients
# call this HF per person year incidence (hfppyi)
# e.g., dataset has 4 patients:
#  patient A - no HF, followed for 3 years
#  patient B -yes HF, after 1 year
#  patient C - no HF, followed for 2 years
#  patient D -yes HF, after 6 months
#  hfppyi = (1 + 0.5) / (3 + 1 + 2 + 0.5) =~ 0.23
#  so this means, on average, per 1 year of follow-up of a patient in ths dataset, we would get 0.23 hf diagnoses, or about 1 in 4 years

# select top 50% of   target-hf risk scores
median_hr = np.quantile(hrs, 0.5)

f_dpi = 96
plt.figure(figsize=(12,12), dpi=f_dpi)
plt.hist(hrs, bins = 100)
plt.axvline(median_hr, color ='red') # add red line for where 50% is
plt.savefig("plots/targethf_cohort_selection/target_hf_hazard_ratios_distr.png")

plt.figure(figsize=(12,12), dpi=f_dpi)
plt.hist(np.log(hrs), bins = 100)
plt.axvline(np.log(median_hr), color ='red') # add red line for where 50% is
plt.savefig("plots/targethf_cohort_selection/target_hf_hazard_ratios_log_distr.png")

cohort['time_to_event'] = cohort['time_to_event'] * 10 # lets keep tte in years, not decades please..
# calc hfppyi
hfppyi = lambda x: sum(x[x['event'] == True]['time_to_event'])/sum(x['time_to_event'])
hfppyi_single = lambda x: 0 if not x['event'] else 1/x['time_to_event']

logger(f"Cohort hfppyi = In a person year we expect {round(hfppyi(cohort), 4)} HF diagnoses")

res_df = pd.DataFrame(columns=["quantile", "score_thresh", "hfppyi", "n_records"])
for i, cur_q in enumerate(np.arange(0.5, 1, 0.005)):
    cur_thresh = np.quantile(hrs, cur_q)
    cur_df = cohort[cohort['target_hf_score'] >= cur_thresh]
    c_hfppyi = hfppyi(cur_df)
    logger(f"In top {cur_q*100}% of scores, hfppyi = In a person year we expect {round(c_hfppyi, 4)} HF diagnoses")
    res_df.loc[i] = [cur_q, cur_thresh, c_hfppyi, len(cur_df)]

res_df.to_excel('excel/cohort_selection_jelle/cohort_selection_jelle.xlsx')

res_df
# optional, select based on thresholds 
tte_thresh = 2 # say two years
cohort['event_2y'] = cohort.apply(lambda x: x['event'] and x['time_to_event'] <= tte_thresh, axis = 1)
cohort['event'].value_counts()
cohort['event_2y'].value_counts()
outcomes = vals(cohort['event_2y'])
baseline_inc = sum(cohort['event_2y']) / len(cohort)
logger(f"Baseline incidence of event in 2 years is {round(baseline_inc,2)}")
hr_threshes = [4.5, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]
for hr_thr in hr_threshes:
    c_set = [i for i,v in enumerate(hrs) if v >= hr_thr]
    c_outs = [v for i,v in enumerate(outcomes) if i in c_set]
    c_prev = np.sum(c_outs) / len(c_outs)
    logger(f"HR thresh of {hr_thr} produces cohort with prevalence of {round(c_prev, 2)} and size {len(c_outs)} ({round(len(c_outs)/len(cohort),2)} out of input set)")

# create a table for Jelle with outcome (binary y/n for years 2018, and 2019)
for start_year in [2018, 2019]:
    cur_bin_outcome = f"HF_yn_{start_year}"
    actual_cohort_year_start = COHORT_TY_START.year + 1 # because COHORT_TY_START is set to 6 months before desired cohort start ...
    diff_chrt_year = start_year - actual_cohort_year_start
    c_cohort = cohort[cohort['time_to_event'] > diff_chrt_year]
    c_cohort[cur_bin_outcome] = c_cohort.apply(lambda x: x['event'] and x['time_to_event'] > diff_chrt_year and x['time_to_event'] <= 1+diff_chrt_year, axis = 1)
    cns(c_cohort)
    c_cohort = c_cohort.drop(columns=[x for x in cns(c_cohort) if x not in ['target_hf_score', cur_bin_outcome, 'time_to_event', 'event']])
    c_cohort.to_csv(f'excel/cohort_selection_jelle/{start_year}_scores_and_cases.csv')
    

# create calibration plot
# 
n_q = 7
cohort['ppy_outcome'] = [hfppyi_single(x) for _,x in cohort.iterrows()]
cohort['risk_quantile'] = pd.qcut(cohort['target_hf_score'], q=n_q, labels = False)
calibration_data = cohort.groupby('risk_quantile').agg(avg_predicted_risk=('target_hf_score', 'mean'),
                                                        observed_ppy=('ppy_outcome', 'mean'))
calibration_data = calibration_data.reset_index()

plt.figure(figsize=(12,12), dpi=f_dpi)
plt.plot(calibration_data['avg_predicted_risk'], calibration_data['observed_ppy'], marker='o')
plt.xlabel('Average target_hf risk score')
plt.ylabel('Average ppy')
plt.show()
plt.savefig(f"plots/targethf_cohort_selection/calibration_target_hf_q{n_q}.png")


plt.figure(figsize=(12,12), dpi=f_dpi)
plt.plot(range(n_q), calibration_data['observed_ppy'], marker='o')
plt.xlabel('Average target_hf risk score (qunatiles)')
plt.ylabel('Average ppy')
plt.show()
plt.savefig(f"plots/targethf_cohort_selection/calibration_target_hf_q_q{n_q}.png")
sys.exit(-1)


#cph_results_l_one = cph_l_one.summary # cannot be trusted  with overriden params
#cph_results_l_one[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]].round(2).join(cph_results_l_one["p"].round(4))

