# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES
# 1. Reads input cohort with gmm clusters results file
# 2. Processes into format used for spider chart (ggradar) plotting in R
# 3. Outputs as csv
'''
)

from try_utils import *
from constants import *
from sys import exit
from sklearn.preprocessing import MinMaxScaler

IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"

check_if_debugging(IS_DEBUG)

outfile_infix = f"{subsampled_str}"
logfile = f'{os.path.basename(__file__)[:-3]}_{outfile_infix}.log'
logger = get_logger_fn(logfile)
logger(f"Starting ...")

import random
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)

# why do we need the model here? so we can map each doc to its label 



import analyse_results_util as ar_util


logger('')


tmp = ar_util.load_cohort(override_saved_file=F, with_gmm_dummies=F)
cohort = tmp['cohort']
# vars_used = [f"{v}_y" if v.startswith("t_") else v for v in tmp['vars_used']] + ['event', 'gmm_cls']
protected_cols = ['event', 'gmm_cls']
vars_used = [ 
		 '0_24_tw_atc_code_cats0_C07A',
		 '0_24_tw_atc_code_cats0_C03C',
		 '0_24_tw_atc_code_cats0_A02B',
		 '0_24_tw_atc_code_cats0_A12A',
		 '0_24_tw_atc_code_cats0_C09A',
		 '0_24_tw_atc_code_cats0_A06A',
		 '0_24_tw_atc_code_cats0_C08C',
		 '0_24_tw_atc_code_cats0_C01D',
		 '0_24_tw_atc_code_cats0_B01A',
		 '0_24_tw_atc_code_cats0_C10A',
		 '0_24_epj_text__t10_twPA_dist',
		 '0_24_epj_text__t24_twPA_dist',
		 '0_24_epj_text__t20_twPA_dist',
		 '0_24_epj_text__t17_twPA_dist',
		 '0_24_epj_text__t16_mx_dist',
		 '0_24_epj_text__t9_twPA_dist',
		 '0_24_epj_text__t4_twPA_dist',
		 '0_24_epj_text__t19_mx_dist',
		 '0_24_epj_text__t21_mx_dist',
		 '0_24_epj_text__t37_mx_dist',
		 '0_24_tw_icpc_ep_cats0_R7',
		 '0_24_tw_icpc_ep_cats0_R2',
		 '0_24_tw_icpc_ep_cats1_*7',
		 '0_24_tw_icpc_ep_cats1_*1',
		 '0_24_tw_icpc_ep_cats0_A5',
		 '0_24_tw_icpc_ep_cats1_*4',
		 '0_24_tw_icpc_ep_cats0_W1',
		 '0_24_tw_icpc_ep_cats0_X1',
		 '0_24_tw_icpc_e_cats0_A5',
		 '0_24_tw_icpc_ep_cats1_*6',
		 'decades_age',
		 't_hypertension_y',
		 't_diabetes_mellitus_y',
		 't_coronary_artery_disease_y',
		 't_atrial_fibrillation_y',
		 't_valvular_heart_disease_y',
		 't_stroke_y',
		 't_copd_y',
		 't_chronic_kidney_disease_y',
		 '0_24_n_eps_'	]   +  protected_cols


cohort = cohort[vars_used]
cohort['gmm_cls'] = cohort['gmm_cls'].astype(str)
cohort['event'] = cohort['event'].astype(float)
gmm_cls_of_interest = [str(x) for x in [2, 1, 0, 8, 11]] + ["ALL"]


# scale vars
df_scaled = cohort.copy()
vars_to_scale = try_sd(vars_used, protected_cols)
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled[vars_to_scale] = scaler.fit_transform(cohort[vars_to_scale])
cohort = df_scaled

# comp group-level aggr vals
gmm_groups = df_scaled.groupby(['gmm_cls'])

cohort_size = pd.Series(nrow(cohort))
cohort_size.index = ["ALL"]
global_mean = cohort.mean(numeric_only=T).to_frame().T
global_mean.index = ["ALL"]
global_median = cohort.median(numeric_only=T).to_frame().T
global_median.index = ["ALL"]


gmm_sizes = pd.concat([gmm_groups.size(),  cohort_size]).to_frame()
gmm_sizes.columns = ['size']

gmm_means = gmm_groups.mean()
gmm_means = pd.concat([global_mean, gmm_means])
gmm_means = pd.concat( [gmm_means, gmm_sizes] , axis = 1)
gmm_means = gmm_means[gmm_means.index.isin(gmm_cls_of_interest)]
gmm_means = gmm_means.sort_values(by=['event'], ascending=F)

gmm_medians = gmm_groups.median()
gmm_medians = pd.concat([global_median, gmm_medians])
gmm_medians = pd.concat( [gmm_medians, gmm_sizes] , axis = 1)
gmm_medians = gmm_medians[gmm_medians.index.isin(gmm_cls_of_interest)]
gmm_medians = gmm_medians.sort_values(by=['event'], ascending=F)

gmm_means.to_csv("gmm_means.csv")
gmm_medians.to_csv("gmm_medians.csv")



    


