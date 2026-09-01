# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
from namespaces import get_ns_name, parse_ns_val_bool
import namespaces
ns_name = get_ns_name(__file__)
ns = getattr(namespaces, ns_name)
cmd_args = parse_commandline_args(verbose=True, required_extra_args=ns.required_extra_args)
IS_DEBUG = cmd_args["IS_DEBUG"]
SUBSAMPLE_DATA = cmd_args["SUBSAMPLE_DATA"]
script_params = { k : parse_ns_val_bool(cmd_args[k]) for k in ns.required_extra_args }
ns.__dict__.update(script_params) # init input args into namespace
logger = get_default_logger_fn(__file__, override=False)
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
full_filename_pkl = lambda fname : f"{fname}{subsampled_str}.pkl"
full_filename_tsv = lambda fname, batch_n: f"{fname}{subsampled_str}b{batch_n}.tsv"
from try_utils import *
from constants import *
import numpy as np
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)
delete_pickle = lambda f: try_delete_pickle(f, subsampled=SUBSAMPLE_DATA)
# Boilerplate end
import pandas as pd
from functools import reduce
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD


infiles_code_cats = [f'icpc_cats{i}.json' for i in [1,2]]
mapping_icpc_to_cat, mapping_cat_to_icpc = get_icpc_cats(infiles_code_cats)

def apply_icpc_grouping(v):
    v = 'NaN' if pd.isnull(v) else v
    tv = v.split(".")[0]
    if tv in mapping_icpc_to_cat:
        return mapping_icpc_to_cat[tv]

    return f'UNDEFINED_{v}'
    

def group_icps(df, code_lookup_dicts, keep_undefined=F):
    data_cns = cns(df)
    icpc_cols = filter_strings_regex(data_cns, 'icpc')     
    if nrow(df) == 0:
        return df
    for icpc_col in icpc_cols:
        df[icpc_col[1]] = df[icpc_col[1]].apply(apply_icpc_grouping) 
        if not keep_undefined and nrow(df) > 0:
            df = df[~df[icpc_col[1]].str.startswith('UNDEFINED')]
    return df



# Description of ATC coding system
# example code: C07AD01 , has 5 levels
# 1st: C  [single letter]
# 2nd: 07 [two digits]
# 3rd: A  [single letter]
# 4th: D  [single letter]
# 5th: 01 [two digits]
max_atc_level = 3
atc_level_lens = [1, 2, 1, 1, 2]
atc_max_len = sum(atc_level_lens)
atc_truncate_after_n_chars = sum(atc_level_lens[:max_atc_level])

def apply_atc_grouping(v):
    if pd.isnull(v):
        return 'UNDEFINED_ATC'
    return v[:atc_truncate_after_n_chars]


def dedup_atcs(df):
    if nrow(df) == 0:
        return df

    n_b4dd = nrow(df)
    df = df.drop_duplicates(
        subset = ['p_id', 'atc_code', 'medication_datetime'], 
        ignore_index=T
        )
    logger(f"[ATC] Before deduplication N={n_b4dd}; After deduplication N={nrow(df)}")
    return df

def group_atcs(df, dedup=F, keep_undefined=F):
    if nrow(df) == 0:
        return df
    if dedup:
        df = dedup_atcs(df)

    data_cns = cns(df)
    atc_cols = filter_strings_regex(data_cns, 'atc')   
    for atc_col in atc_cols:
        df[atc_col[1]] = df[atc_col[1]].apply(apply_atc_grouping) 
        if not keep_undefined and nrow(df) > 0:
            df = df[~df[atc_col[1]].str.startswith('UNDEFINED')]
    return df

def dim_red__expert_knowledge_icpc(code_lookup_dicts, data, data_cns,
    infiles_code_cats = [f'icpc_cats{i}.json' for i in [1,2]],
    outcome_of_interest = None,
    return_as = "csr", # "df"
    remove_empty_cols = F,
    silent=F
 ):
    c_log = lambda s, t=None: logger(s, t) if not silent else None
    n_cols_start = data.shape[1]
    data_cns = [v for _,v in data_cns]
    icpc_cols = filter_strings_regex(data_cns, 'icpc_\w+_\d+')
    icpc_codes = [(c_col, one_hot_decode_varname(c_col, code_lookup_dicts)) for _,c_col in icpc_cols if c_col != outcome_of_interest[1] ]
    # ICPC coding system explained:
    # {LETTER}{NUMBER}[optional]{.NUMBER}
    icpc_letters, icpc_numbers = [x[0] for _,x in icpc_codes], [x[1:] for _,x in icpc_codes]
    mapping_icpc_to_cat, mapping_cat_to_icpc = get_icpc_cats(infiles_code_cats)
    
    c_log(try_table(["_".join(c.split("_")[3:-1]) for c in data_cns if 'icpc' in c]))

    time_periods = uniq(["_".join(x.split("_")[0:2]) for x,_ in icpc_codes])
    icpc_var_names = uniq(["_".join(x.split("_")[2:5]) for x,_ in icpc_codes])
    n_df = None
    total_n_iterations = len(mapping_cat_to_icpc) * len(time_periods) * len(icpc_var_names)
    iter_counter = 0
    t0 = c_log("Starting dim_red__expert_knowledge_icpc ...")
    t1 = t0
    log_every_n = 1000 if SUBSAMPLE_DATA else 100
    for n_code, old_codes in mapping_cat_to_icpc.items(): # n_code, old_codes = list(code_cat_mapping.items())[4]
        for t_period in time_periods: 
            for icpc_var_name in icpc_var_names: # icpc_var_name = icpc_var_names[0]
                iter_counter += 1
                c_icpc_codes = [(col_name, i_code) for col_name, i_code in icpc_codes if i_code in old_codes]
                c_icpc_codes = [(a,b) for a,b in c_icpc_codes if a.startswith(t_period)]
                c_icpc_codes = [(a,b) for a,b in c_icpc_codes if "_".join(a.split("_")[2:5]) == icpc_var_name]
                c_idxs = [i for i,v in icpc_cols if v in [x for x,_ in c_icpc_codes]]
                old_vals = data[:, c_idxs].toarray()
                n_col = f"{t_period}_{icpc_var_name}_{n_code}"
                if iter_counter % log_every_n == 0:
                    t1 = c_log(f"Done with {n_col} ({iter_counter}/{total_n_iterations})", t1) 
                n_val = pd.DataFrame(np.sum(old_vals, axis=1), columns=[n_col])
                n_df = pd.concat([n_df, n_val], axis=1)
    
    data, data_cns = remove_csr_columns(icpc_cols, data, data_cns)
    if remove_empty_cols:
        cols_to_remove = [c for c in cns(n_df) if len(n_df[c].value_counts()) <= 1]
        n_df = n_df.drop(cols_to_remove, axis=1)
    if ncol(n_df) != 0:
        data = hstack([data, csr_matrix(n_df)])
        data_cns += list(n_df.columns)
    c_log(f"Dimensionality reduction of icpc codes via expert knowledge of {n_cols_start} down to {data.shape[1]}", t0)
    c_log(try_table(["_".join(c.split("_")[3:-1]) for c in data_cns if 'icpc' in c]))
    if return_as == "df":
        ret = pd.DataFrame(data.todense())
        ret.columns = data_cns
        return ret
    return (data, data_cns)