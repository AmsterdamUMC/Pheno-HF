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
import analyse_results_util as ar_util
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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
split_atc_rowname = lambda rn: rn.split("_")[-1:][0]

def get_atcs_within_flwp(p):
    dt_atcs = [d for d in [ (m['medication_datetime'], m['atc_code']) for m in p['Medications'] if ar_util.is_med_dt_during_flwp(p, m)] if not pd.isnull(d[1])]
    if dt_atcs == []:
        return []
    dt_atcs = list(dict.fromkeys(dt_atcs)) # dedup
    return dt_atcs#list(list(zip(*dt_atcs))[1])

def init__pats_dict_with_labs_atc(gmm_df, pats_dict_file = "pats_dict_merged.pkl"):
    x = read_pickle(pats_dict_file)
    r_keys = set(gmm_df['id_str'].values)
    p_ids = [ k for k in x.keys() if k in r_keys] # this should be a full overlap, unly in debug otherwise...
    logger(f"{len(p_ids)} patient ids extracted")

    all_HFpos_ids = vals(gmm_df[gmm_df.event == 1].id_str)
    for k in x.keys():
        x[k]['event'] = 0
    for c_id in all_HFpos_ids:
        x[c_id]['event'] = 1

    for k in x.keys():
        x[k]['follow_up_LAST'] = ar_util.apply_flwp_cens_window(k, x[k]['follow_up_LAST'], all_HFpos_ids)

    # 2. extract atc codes only from that range, and dedup
    # deduplicate based on medication_datetime, atc_code (ideally would also do based on import_id, but oh well...)
    x_meds = [get_atcs_within_flwp(x[k]) for k in p_ids]


    logger(f"medication lists of {len(x_meds)} patients extracted")
    empty_idxs = set([i for i,m in enumerate(x_meds) if m == []])
    x_meds = [m for i,m in enumerate(x_meds) if i not in empty_idxs]
    p_ids = [pid for i,pid in enumerate(p_ids) if i not in empty_idxs]
    logger(f"NON-EMPTY medication lists of {len(x_meds)} patients remain")
    for i,p_id in enumerate(p_ids):
        x_meds[i] =  [ {'id_str': p_id, 'atc_code' : med[1], 'medication_datetime': med[0]} for med in x_meds[i] ]
    logger(f"p_id key appended to med lists of {len(x_meds)} patients")

    meds_flat = [ xm for xms in x_meds for xm in xms ]
    del x_meds
    logger(f"{len(meds_flat)} total medication records flattened")
    meds_df = pd.DataFrame.from_records(meds_flat)
    del meds_flat
    meds_df = meds_df[['id_str', 'atc_code', 'medication_datetime']]

    meds_df = pd.merge(meds_df, gmm_df [["id_str", "gmm_cls"]], on = 'id_str', how = 'inner').reset_index(drop=T)
    meds_df_dup_check = meds_df.copy().drop_duplicates(
        subset = ['id_str', 'atc_code', 'medication_datetime'],  # why no id here?
        ignore_index=T
        )
    logger(f"{nrow(meds_df) - nrow(meds_df_dup_check)} duplicate ATC codes removed (on same patient, at same day)")
    logger(f"Left with {nrow(meds_df_dup_check)} ATC code occurrences")
    meds_df = meds_df_dup_check.copy()
    del meds_df_dup_check

    meds_df = meds_df[~pd.isnull(meds_df.atc_code)]
    logger(f"Removed entries with NaN atc_code. Left with {nrow(meds_df)} ATC code occurrences")
    return meds_df, x


def init__mark_used_vars_atc(out_df, vars_used):
    # mark rows with variables that were used in GMM
    atc_vars = [ split_atc_rowname(v) for v  in vals(out_df.index)]
    atc_cats_used = [split_atc_rowname(v) for v in try_regex_multi(vars_used, 'atc')]
    is_in_cats_used = lambda atc : any([atc.startswith(v) for v in atc_cats_used])
    atc_vars_used = sorted(list(set([ v for v in atc_vars if is_in_cats_used(v)])))
    
    logger(f"{len(atc_vars_used)} unique ATC codes used from {len(atc_cats_used)} atc cats selected")
    
    out_df['used_by_gmm'] = "NO"
    out_df['used_by_gmm'].loc[atc_vars_used] = "YES"
    return out_df, atc_vars_used

def analyse_ATC_results():
    model, vars_used, gmm_df = cached_call(ar_util.init__gmm_results, override_cache=T)

    meds_df, x = cached_call(init__pats_dict_with_labs_atc, override_cache=T, gmm_df = gmm_df)
    


    meds_per_pat = meds_df.copy().groupby(['id_str', 'atc_code']).size().reset_index(name='ATC_count')
    meds_per_pat = meds_per_pat.sort_values(by=['ATC_count'], ascending=F)
    meds_1pat = [v for v in x['1058|1']['Medications'] if ar_util.is_med_dt_during_flwp(x['1058|1'], v)]
    meds_1pat = pd.DataFrame.from_records(meds_1pat) # already sorted by time
    
    #meds_per_pat.to_csv("excel/ATC_per_pat_counts.csv")
    #meds_1pat.to_excel("excel/meds_1pat.xlsx")
    # what is the deal with 10x4 clust 0?? check out the M01 codes for example..
    if 1 == 2:
        xxx = meds_df[meds_df.gmm_cls == 0]
        meds_counts = xxx.copy().groupby(['atc_code']).size().reset_index(name='ATC_count')
        meds_counts = meds_counts[meds_counts['atc_code'].str.startswith('M01A')]
        #
        meds_counts.sum() # = 12809  /5070
        yyy = meds_df[meds_df.gmm_cls == 6]
        meds_counts = yyy.copy().groupby(['atc_code']).size().reset_index(name='ATC_count')
        meds_counts = meds_counts[meds_counts['atc_code'].str.startswith('M01A')]
        meds_counts.sum() # = 2104 / 2900

        meds_per_pat = xxx.copy().groupby(['id_str', 'atc_code']).size().reset_index(name='ATC_count')
        meds_per_pat = meds_per_pat[meds_per_pat['atc_code'].str.startswith('M01A')]
        meds_per_pat = meds_per_pat.sort_values(by=['ATC_count', 'atc_code'], ascending=F)
        meds_per_pat.head()

        # seems they do have more prescriptions, but only like 2-4 times more than normal, not 24 times...
        # why is this 24 times there??
        # only other explanation - they had these prescriptions very recently
        cohort =ar_util.load_cohort(with_gmm_dummies=F)['cohort']
        xxx = cohort[cohort.gmm_cls == 0]
        xxx['0_24_tw_atc_code_cats0_M01A'].round(3).value_counts()
        xxx['0_24_tw_atc_code_cats0_M01A'].mean()
        cohort['0_24_tw_atc_code_cats0_M01A'].mean()


    meds_per_cluster = meds_df.groupby(['gmm_cls', 'atc_code']).size().reset_index(name='ATC_count')
    meds_per_cluster = meds_per_cluster.drop_duplicates(ignore_index=T) 
    logger(f"{len(set(meds_per_cluster.atc_code.values))/model.n_components:.0f} unique ATC codes remaining")
    
    cluster_sizes = gmm_df[['id_str', 'gmm_cls']]
    cluster_sizes = cluster_sizes.drop_duplicates(ignore_index=T).groupby(['gmm_cls']).size().reset_index(name='clust_size')
    cluster_sizes = cluster_sizes.to_dict()['clust_size']

    # row = ATC code , col = cluster , value_ij = number times ATC code i  was seen in cluster j 
    meds_per_cluster_matrix = meds_per_cluster.pivot(index='atc_code', columns='gmm_cls', values = 'ATC_count').fillna(0)

    meds_per_cluster_corr = pd.DataFrame(index = meds_per_cluster_matrix.index, columns = cns(meds_per_cluster_matrix))
    n = sum(cluster_sizes.values())
    n_meds = meds_per_cluster_matrix.sum().sum()
    for c_clst in cns(meds_per_cluster_matrix):
        for c_atc in meds_per_cluster_matrix.index:
            s_a = meds_per_cluster_matrix.loc[c_atc]
            a = s_a[c_clst]
            b = s_a.sum() - a
            c = meds_per_cluster_matrix.iloc[:, c_clst].sum() - a
            d = n_meds - s_a.sum() - a
            numerator = (a*d - b*c) 
            denom = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
            c_corr = numerator/denom
            meds_per_cluster_corr.loc[c_atc, c_clst] = c_corr

    meds_per_cluster_corr = meds_per_cluster_corr[[2, 1, 0, 8, 11, 9, 5, 6, 3, 10, 7, 4]]         
    meds_per_cluster_corr.columns =   [ f"C{c} n~{(cluster_sizes[c]/1000):.0f}k" for c in meds_per_cluster_corr.columns.values]


    meds_per_cluster_corr, atc_vars_used = cached_call(init__mark_used_vars_atc, override_cache=T,
            out_df = meds_per_cluster_corr,
            vars_used = vars_used
            )


    meds_per_cluster_corr.to_excel("excel/meds_per_cluster_corr.xlsx")
    return 0

analyse_ATC_results()


logger("DONE")
print("DONE")