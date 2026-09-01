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
import analyse_results_util as ar_util
pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)
cached_call = lambda fn, override_cache=F, **kwargs : try_cached_call(fn, io_r=read_pickle, io_c=pickle_exists, io_w=save_pickle, override_cache=override_cache, **kwargs)

#from Top2Vec import Top2Vec
#umap_file = EMBEDDING_MODEL_METADATA['doc2vec'][f'umap_embeddings_file{subsampled_str}']['0_24_epj_text_'] 
#hdbscan_file = EMBEDDING_MODEL_METADATA['doc2vec'][f'hdbscan_labels_file{subsampled_str}']['0_24_epj_text_']
print("s") #
# why do we need the model here? so we can map each doc to its label 

# [23.Jun.2025]


def analyse_targetHF_score():
    # Base: cluster id, clusters proba, pat_id
    # get cluster ids/probs 
    model, vars_used, gmm_df = cached_call(ar_util.init__gmm_results, override_cache=T)




    tmp = ar_util.load_cohort(override_saved_file=F)
    cohort = tmp['cohort']
    gmm_vars = tmp['vars_used']
    del tmp
    
    clust_stats = ar_util.calc_cluster_stats(cohort)
    baseline_inci = vals(clust_stats[clust_stats['gmm_lab'] == 'ALL']['inci'])[0]
    hi_hf_cluster_cols = vals(clust_stats[clust_stats['inci'] > 2*baseline_inci]['gmm_lab'])


    gmm_labs = try_regex_multi(cns(cohort), 'gmm_cls')
    prob_gmm_labs = try_regex_multi(gmm_labs, '_prob')
    abs_gmm_labs = try_sd(gmm_labs, prob_gmm_labs)

    protected_cols =  [ar_util.ns.id_col, ar_util.ns.outcome_col] #[ar_util.ns.id_col, ar_util.ns.outcome_col, ar_util.ns.target_hf_col] + gmm_labs + gmm_vars
    all_cols = sorted(list(set(protected_cols + prob_gmm_labs + abs_gmm_labs + gmm_vars + [ar_util.ns.target_hf_col] + ar_util.ns.targetHF_cols)))
    cohort = cohort.drop(try_sd(cns(cohort), all_cols), axis =1)
    cohort['t_coronary_artery_disease'] = cohort['coronary_artery_disease']
    cohort['t_chronic_kidney_disease'] = cohort['chronic_kidney_disease']
    cohort['t_copd'] = cohort['copd']
    cohort['t_stroke'] = cohort['stroke']
    cohort['t_diabetes_mellitus'] = cohort['diabetes_mellitus']
    cohort['t_valvular_heart_disease'] = cohort['valvular_heart_disease']
    cohort['t_hypertension'] = cohort['hypertension']
    cohort['t_atrial_fibrillation'] = cohort['atrial_fibrillation']


    cluster_sizes = gmm_df[['id_str', 'gmm_cls']]
    cluster_sizes = cluster_sizes.drop_duplicates(ignore_index=T).groupby(['gmm_cls']).size().reset_index(name='clust_size')
    cluster_sizes = cluster_sizes.to_dict()['clust_size']


    gmm_df['prac_id'] = gmm_df.id_str.apply(lambda x : x.split('|')[1])


    cohort['time_to_event'] = 0.25 # 3 months
    targethf_preds = calc_TARGETHF_scores(cohort)
    cohort['targetHF_score'] = targethf_preds
    out_df = pd.merge(cohort[['id_str', 'targetHF_score', 'event']], gmm_df[["id_str", "gmm_cls"]], on = 'id_str', how = 'inner').reset_index(drop=T)
    per_clust_mean = out_df[['gmm_cls', 'targetHF_score']].groupby(['gmm_cls'])['targetHF_score'].mean()
    per_clust_std = out_df[['gmm_cls', 'targetHF_score']].groupby(['gmm_cls'])['targetHF_score'].std()
    per_clust_inci = out_df[['gmm_cls', 'event']].groupby(['gmm_cls'])['event'].mean()
    out_df = pd.concat([per_clust_mean, per_clust_std, per_clust_inci], axis=1)
    out_df.columns = ['TargetHF_score(mean)', 'TargetHF_score(SD)', 'HF incidence']

    out_df.to_excel("excel/targetHF_score_per_cluster.xlsx")
    return 0

analyse_targetHF_score()