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

import analyse_results_util as ar_util

#from Top2Vec import Top2Vec
#umap_file = EMBEDDING_MODEL_METADATA['doc2vec'][f'umap_embeddings_file{subsampled_str}']['0_24_epj_text_'] 
#hdbscan_file = EMBEDDING_MODEL_METADATA['doc2vec'][f'hdbscan_labels_file{subsampled_str}']['0_24_epj_text_']
print("s") #
# why do we need the model here? so we can map each doc to its label 

# [23.Jun.2025]



def analyse_practice_results():
    tmp = ar_util.load_cohort(override_saved_file=F)
    cohort = tmp['cohort']
    clust_stats = ar_util.calc_cluster_stats(cohort)
    gmm_labs = try_regex_multi(cns(cohort), 'gmm_cls')
    prob_gmm_labs = try_regex_multi(gmm_labs, '_prob')
    abs_gmm_labs = try_sd(gmm_labs, prob_gmm_labs)
    cohort['gmm_cls'] = 0 # assumes cls_0 is the ref cat
    for gmm_l in abs_gmm_labs:
        c_class = int(gmm_l.split('_')[-1])
        cohort.loc[cohort[gmm_l] == T, 'gmm_cls'] = c_class


    cohort['prac_id'] = cohort.id_str.apply(lambda x : x.split('|')[1])

    hf_per_prac = {}
    gr = cohort.groupby(['prac_id'])

    for cgr, crows in gr:
        hf_per_prac[cgr]  = { 'n' : nrow(crows), 'nHFPos' :  sum(crows.event) , 'inci' : sum(crows.event)/nrow(crows) }
        # break
        
    hf_per_prac = pd.DataFrame.from_records(hf_per_prac).T
    hf_per_prac.to_excel('excel/hf_per_prac.xlsx')

    prac_per_clst = cohort.groupby(['prac_id', 'gmm_cls']).size().reset_index(name='prac_count')
    prac_per_clst =  prac_per_clst.sort_values(by=['gmm_cls', 'prac_id'], ascending=F)
    prac_per_cluster_matrix = prac_per_clst.pivot(index='prac_id', columns='gmm_cls', values = 'prac_count').fillna(0)
    cs_df = clust_stats[clust_stats['gmm_lab'] != 'ALL']

    gmm_labs_inci_sorted = [ x.split("_")[-1] for x in try_regex_multi( vals(clust_stats['gmm_lab']), 'gmm_cls_')]
    gmm_labs_inci_sorted = [int(x) if x != 'REF' else 0 for x in gmm_labs_inci_sorted]
    cluster_sizes = { c:s for c,s in zip(gmm_labs_inci_sorted, vals(cs_df['size'])) } # used for column naming
    prac_per_cluster_matrix = prac_per_cluster_matrix[gmm_labs_inci_sorted]
    prac_per_cluster_matrix.columns =   [ f"C{c} n~{(cluster_sizes[c]/1000):.0f}k" for c in vals(prac_per_cluster_matrix.columns)]

    prac_per_cluster_corr = pd.DataFrame(index = prac_per_cluster_matrix.index, columns = cns(prac_per_cluster_matrix))
    n = sum(cluster_sizes.values())
    n_pracs = prac_per_cluster_matrix.sum().sum()
    for clst_idx, c_clst in enumerate(cns(prac_per_cluster_matrix)):
        for c_attr in prac_per_cluster_matrix.index:
            s_a = prac_per_cluster_matrix.loc[c_attr]
            a = s_a[c_clst]
            b = s_a.sum() - a
            c = prac_per_cluster_matrix.iloc[:, clst_idx].sum() - a
            d = n_pracs - s_a.sum() - a
            numerator = (a*d - b*c) 
            denom = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
            c_corr = numerator/denom
            prac_per_cluster_corr.loc[c_attr, c_clst] = c_corr
  
    prac_per_cluster_corr.to_excel("excel/prac_per_cluster_corr.xlsx")
    
    return 0

analyse_practice_results()