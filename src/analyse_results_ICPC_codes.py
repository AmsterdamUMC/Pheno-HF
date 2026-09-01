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
# Top-level util functions (called only once)
# @Cacheable
def init__pats_dict_with_labs(gmm_df, pats_dict_file = "pats_dict_merged.pkl"):
    p_icpcs = {"icpc_ep" : {} ,
                "s" : {}, 
                "o" : {}, 
                "e" : {}, 
                "p" : {}, 
                "x" : {}
                }
    x = read_pickle(pats_dict_file)
    # add HFpos outcome of interest to pat dict items
    all_HFpos_ids = vals(gmm_df[gmm_df.event == 1].id_str)
    for k in x.keys():
        x[k]['event'] = 0
    for c_id in all_HFpos_ids:
        x[c_id]['event'] = 1

    # set dict to only use pats that were in experiments 
    x = { k:x[k] for k in vals(gmm_df['id_str'])}
    r_keys = set(gmm_df['id_str'].values)
    p_ids = [ i for j,i in enumerate(x.keys()) if i in r_keys] # this should be a full overlap, unly in debug otherwise...
    logger(f"{len(p_ids)} patient ids extracted")

    # remove episodes/journals outside of flwp time
    # for each pat,
    # 1. extract their last_flwp date, and derive the start date of the flwp,
    # note: need to subtract FOLLOW_UP_HFPOS_CENS_WINDOW in HF+ cases! 
    
    for k in x.keys():
        x[k]['follow_up_LAST'] = ar_util.apply_flwp_cens_window(k, x[k]['follow_up_LAST'], all_HFpos_ids)

    # 2. extract icpc codes only from that range, and dedup
    # deduplicate based on ep_start_date, icpc_ep (ideally would also do based on import_id, but oh well...)
    for pid,p in list(x.items()):
        
        p_icpcs['icpc_ep'][pid] = get_ep_icpcs_within_flwp(p)
        soepx_j_icpcs = get_j_icpcs_within_flwp(p)
        for k in soepx_j_icpcs.keys():
            p_icpcs[k][pid] = soepx_j_icpcs[k]

    # remove empty entries
    for icpc_cat in p_icpcs.keys():
        p_icpcs[icpc_cat] = {k:v for k,v in p_icpcs[icpc_cat].items() if v != []}
    logger(f"icpc episode+SOEPX codes of patients extracted")
    return p_icpcs

# @Cacheable
def init__mark_used_vars(out_df, vars_used, infiles_code_cats = [f'icpc_cats{i}.json' for i in [1,2]]):
    # mark rows with variables that were used in GMM
    code_mapping_todo_bad_refactor =  {
                "ep" : "icpc_ep",
                "s" : "s", 
                "o" : "o", 
                "e" : "e", 
                "p" : "p", 
                "x" : "x" 
                }
    mapping_icpc_to_cat, mapping_cat_to_icpc = get_icpc_cats(infiles_code_cats)

    icpc_vars = [ split_icpc_rowname(v) for v  in vals(out_df.index)]
    icpc_cats_used = [ f"{extr_icpc_cat_nm(v)}_{extr_icpc_soepxed(v)}" for v  in vars_used if 'icpc' in v]

    is_icpc_var_used = lambda icpc_var, icpc_soepx: f"{mapping_icpc_to_cat[icpc_var]}_{icpc_soepx}" in icpc_cats_used if icpc_var in mapping_icpc_to_cat else F
    icpcs_vars_used = sorted(list(set([f"{icpc[1]}_{icpc[0]}" for icpc in icpc_vars if is_icpc_var_used(icpc[0], icpc[1])])))
    
    logger(f"{len(icpcs_vars_used)} unique ICPC codes used from {len(icpc_cats_used)} icpc cats selected")
    
    out_df['used_by_gmm'] = "NO"
    out_df['used_by_gmm'].loc[icpcs_vars_used] = [mapping_icpc_to_cat[v.split("_")[-1:][0]] for v in icpcs_vars_used]    

    return out_df, icpcs_vars_used

# Second-level util functions (called multiple times with different inputs, non-cacheable)



extr_icpc_cat_nm = lambda v: "_".join(v.split("_")[-2:])
code_mapping_todo_bad_refactor =  {
            "ep" : "icpc_ep",
            "s" : "s", 
            "o" : "o", 
            "e" : "e", 
            "p" : "p", 
            "x" : "x" 
            }

extr_icpc_soepxed = lambda v: code_mapping_todo_bad_refactor["_".join(v.split("_")[-3:-2])]
split_icpc_rowname = lambda rn: ar_util.split_code_rowname(rn, 'icpc_ep')


def get_ep_icpcs_within_flwp(p):
    dt_icpcs = [d for d in [ (ep['episode_start_date'], ep['icpc_episode']) for ep in p['Episodes'] if ar_util.is_ep_dt_during_flwp(p, ep)] if not pd.isnull(d[1])]
    if dt_icpcs == []:
        return []
    dt_icpcs = list(dict.fromkeys(dt_icpcs)) # dedup
    return list(list(zip(*dt_icpcs))[1])

def get_j_icpcs_within_flwp(p):
    dt_icpcs_d = {
            "s" : [], 
            "o" : [], 
            "e" : [], 
            "p" : [], 
            "x" : []
            }
    eps = [ep for ep in p['Episodes'] if ar_util.is_ep_dt_during_flwp(p, ep) and ep['JOURNALS'] != []]
    js = [j for ep in eps for j in ep['JOURNALS'] if ar_util.is_dt_during_flwp(p, j['journal_datetime'])]
    if js == []:
        return dt_icpcs_d
    # dedup journals based on date and icpc

    for icpc_cat in list("soepx"):
        dt_icpcs = [(j['journal_datetime'], j[f"icpc_{icpc_cat}"]) for j in js if not pd.isnull(j[f"icpc_{icpc_cat}"])]
        if dt_icpcs == []:
            dt_icpcs_d[icpc_cat] = []
            continue    
        dt_icpcs = list(dict.fromkeys(dt_icpcs)) # dedup
        dt_icpcs_d[icpc_cat] = list(list(zip(*dt_icpcs))[1])

    return dt_icpcs_d

def get_corr_out_df(p_icpcs, vars_used, model, gmm_df):
    flatten_ = lambda k,vs : [(k, v) for v in vs]
    flat_dfs = {}

    for icpc_cat in p_icpcs.keys():
        flat = [flatten_(k,v) for k,v in p_icpcs[icpc_cat].items()]
        flat = [i for o in flat for i in o]
        flat_df = pd.DataFrame(flat)
        flat_df.columns = ['id_str', f'icpc_{icpc_cat}']
        flat_df = pd.merge(flat_df, gmm_df[[ 'id_str', 'gmm_cls']], on ='id_str', how = 'left')
        flat_dfs[icpc_cat] = flat_df

    cluster_sizes = gmm_df[['id_str', 'gmm_cls']]
    cluster_sizes = cluster_sizes.drop_duplicates(ignore_index=T).groupby(['gmm_cls']).size().reset_index(name='clust_size')
    cluster_sizes = cluster_sizes.to_dict()['clust_size']
    out_df = None
    for icpc_cat in list(p_icpcs.keys())[::-1]:
        c_df = flat_dfs[icpc_cat]
        c_icpc_col_nm = cns(c_df)[1]
        icps_per_cluster = c_df.groupby(['gmm_cls', c_icpc_col_nm]).size().reset_index(name=f"{c_icpc_col_nm}_count")
        icps_per_cluster = icps_per_cluster.drop_duplicates(ignore_index=T) 
        icps_per_cluster[c_icpc_col_nm] = [f"{icpc_cat}_{v}" for v in vals(icps_per_cluster[c_icpc_col_nm])]
        logger(f"{len(set(icps_per_cluster[c_icpc_col_nm].values))/model.n_components:.0f} unique {c_icpc_col_nm} codes remaining")
        # row = ICPC code , col = cluster , value_ij = number times ICPC code i  was seen in cluster j 
        icps_per_cluster_matrix = icps_per_cluster.pivot(index=c_icpc_col_nm, columns='gmm_cls', values = f"{c_icpc_col_nm}_count").fillna(0)
        #icpc_code_names = [ f"{icpc_cat}_{v}" for v in icps_per_cluster_matrix.index ]
        icps_per_cluster_corr = pd.DataFrame(index = icps_per_cluster_matrix.index, columns = cns(icps_per_cluster_matrix))
        n = sum(cluster_sizes.values())
        n_icps = icps_per_cluster_matrix.sum().sum()
        expected_clusters = sorted(list(gmm_df.gmm_cls.unique()))
        for c_clst in expected_clusters:
            is_clst_present = c_clst in cns(icps_per_cluster_matrix)
            for c_icpc in icps_per_cluster_matrix.index:
                c_corr = np.nan
                if is_clst_present:
                    s_a = icps_per_cluster_matrix.loc[c_icpc]
                    a = s_a[c_clst]
                    b = s_a.sum() - a
                    clst_idx = cns(icps_per_cluster_matrix).index(c_clst)
                    c = icps_per_cluster_matrix.iloc[:, clst_idx].sum() - a
                    d = n_icps - s_a.sum() - a
                    numerator = (a*d - b*c) 
                    denom = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
                    c_corr = numerator/denom
                icps_per_cluster_corr.loc[c_icpc, c_clst] = c_corr
        icps_per_cluster_corr = icps_per_cluster_corr[[2, 1, 0, 8, 11, 9, 5, 6, 3, 10, 7, 4]]         
        icps_per_cluster_corr.columns =  [ f"C{c} n~{(cluster_sizes[c]/1000):.0f}k" for c in icps_per_cluster_corr.columns.values]
        out_df = pd.concat([icps_per_cluster_corr, out_df], axis=0)
    return out_df

# utils end

def analyse_ICPC_results():
    # init
    model, vars_used, gmm_df = cached_call(ar_util.init__gmm_results, override_cache=F)
    p_icpcs = cached_call(init__pats_dict_with_labs, override_cache=T, gmm_df=gmm_df)

    # great! now we need to see how these ICPC codes look per cluster 
    out_df = cached_call(get_corr_out_df, 
            override_cache=T,
            p_icpcs=p_icpcs,
            vars_used=vars_used,
            model=model,
            gmm_df=gmm_df
            )

    out_df, icpcs_vars_used = cached_call(init__mark_used_vars,
            override_cache=T,
            out_df = out_df,
            vars_used = vars_used
            )

    # add defs 
    icpc_defs = pd.read_excel("excel/icpc_defs.xlsx")
    icpc_def_exct_mch = lambda icpc_code : ";".join(vals(icpc_defs[icpc_defs['ICPC code'] == icpc_code]['ICPC tekst']))
    icpc_def_nexct_mch = lambda icpc_code : icpc_def_exct_mch(icpc_code.split(".")[0])
    is_two_level_icpc = lambda icpc_code : len(icpc_code.split(".")[0]) == 2
    find_icpc_def = lambda icpc_code: f"{icpc_def_exct_mch(icpc_code)} || {icpc_def_nexct_mch(icpc_code)}" if is_two_level_icpc(icpc_code) else icpc_def_exct_mch(icpc_code)

    out_df['icpc_def'] = [find_icpc_def(v.split("_")[-1:][0]) for v in vals(out_df.index)]
    out_df.to_excel("excel/icps_per_cluster_corr.xlsx")

    icpcs_missing_defs = sorted(list(set([split_icpc_rowname(v)[0] for v in vals(out_df[out_df['icpc_def'] == ""].index)])))
    logger(f"{len(icpcs_missing_defs)} ICPC code defs missing! Namely:")
    try_print_list(icpcs_missing_defs, logger)

    return 0
    

analyse_ICPC_results()

logger("DONE")
print("DONE")