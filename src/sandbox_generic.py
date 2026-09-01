from try_utils import *
from constants import *
from sys import exit

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



vars_main = ['age_days', 'atc_code_cats0_A02B', 'atc_code_cats0_A06A', 'atc_code_cats0_A12A', 'atc_code_cats0_B01A', 'atc_code_cats0_C01D', 'atc_code_cats0_C03C', 'atc_code_cats0_C07A', 'atc_code_cats0_C08C', 'atc_code_cats0_C09A', 'atc_code_cats0_C10A', 'deceased', 'eps_', 'follow_up_LAST', 'icpc_e_cats0_A5', 'icpc_ep_cats0_A5', 'icpc_ep_cats0_R2', 'icpc_ep_cats0_R7', 'icpc_ep_cats0_W1', 'icpc_ep_cats0_X1', 'icpc_ep_cats1_*1', 'icpc_ep_cats1_*4', 'icpc_ep_cats1_*6', 'icpc_ep_cats1_*7', 't_atrial_fibrillation', 't_chronic_kidney_disease', 't_copd', 't_coronary_artery_disease', 't_diabetes_mellitus', 't_hypertension', 't_stroke', 't_valvular_heart_disease'] 


vars_sa = ['age_days', 'atc_code_cats0_A02B', 'atc_code_cats0_A06A', 'atc_code_cats0_A12A', 'atc_code_cats0_B01A', 'atc_code_cats0_C01D', 'atc_code_cats0_C03C', 'atc_code_cats0_C07A', 'atc_code_cats0_C08C', 'atc_code_cats0_C09A', 'atc_code_cats0_C10A', 'deceased', 'eps_', 'follow_up_LAST', 'icpc_ep_cats0_A5', 'icpc_ep_cats0_L1', 'icpc_ep_cats0_R2', 'icpc_ep_cats0_R7', 'icpc_ep_cats0_W1', 'icpc_ep_cats0_X1', 'icpc_ep_cats1_*1', 'icpc_ep_cats1_*4', 'icpc_ep_cats1_*6', 'icpc_ep_cats1_*7', 't_atrial_fibrillation', 't_chronic_kidney_disease', 't_copd', 't_coronary_artery_disease', 't_diabetes_mellitus', 't_hypertension', 't_stroke', 't_valvular_heart_disease']

logger('')

def analysed_missed_HF():
    # quick check, are there dup consuslts?? yup! there are..
    # xxx = read_pickle('pats_dict_text.pkl')
    # x = xxx['x']
    # tm_c = xxx['time_cols'][0]
    # tx_c = xxx['text_cols'][0]
    # for pid,pd in x.items():
    #     if pd[tm_c] != []:
    #         break
    #     pd[tx_c]
    tmp = ar_util.load_cohort(override_saved_file=F, with_gmm_dummies=F)
    cohort = tmp['cohort']
    vars_used = tmp['vars_used']
    missed_HF_fn = f"{ar_util.ns.res_mod}_cohort_missed_HF.pkl"
    if pickle_exists(missed_HF_fn):
        cohort = read_pickle(missed_HF_fn)
    else:
        x = read_pickle(ar_util.ns.pats_dict_file)
        df = ar_util.try_dict_to_df(x, ['t_HF', 't_icpc_HF', 't_text_HF'])
        df[ar_util.ns.id_col] = list(x.keys())
        cohort = pd.merge(cohort, df, how='inner', on=ar_util.ns.id_col)
        save_pickle(missed_HF_fn, cohort)
    
    # event_TXT: patient's HF was caught by the text tag before the ICPC tag
    cohort['event_TXT'] = cohort.t_icpc_HF < cohort.t_text_HF
    cohort['event_TXT'] = cohort.event_TXT | (~pd.isnull(cohort.t_text_HF) & pd.isnull(cohort.t_icpc_HF))
    per_clust_missed_count = cohort[['gmm_cls', 'event_TXT']].groupby(['gmm_cls'])['event_TXT'].sum()
    out_fn = f"{ar_util.ns.res_mod}_missed_HF_count_per_cluster.xlsx"
    per_clust_missed_count.to_excel(f"excel/{out_fn}")
    

    return per_clust_missed_count, out_fn

analysed_missed_HF()