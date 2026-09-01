# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES 
# Perform dimensionality reduction on feature space (using either expert knowledge or autoencoder) (only for icpc codes and postcodes, not text)
# remove last category from each ohe-var 
# Note: need to have run A_runner before this!
'''
)
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
logger = get_default_logger_fn(__file__) # init logger
check_if_debugging(IS_DEBUG)
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
start_time = logger("Start running...")
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
# note: do NOT import * , messes up the namespace
from dim_reduce_utils import dim_red__expert_knowledge_icpc, atc_level_lens, atc_max_len, atc_truncate_after_n_chars


infile = full_filename_pkl(ns.infile)
infile_colnames = full_filename_pkl(ns.infile_colnames)
infile_code_lookups = full_filename_pkl(ns.infile_code_lookups)




dim_red_technique = 'expert_knowledge'

oufile = full_filename_pkl(ns.outfile)


# Mains
# : reduce dimensionality (Autoencoder) 
# # Implemented so far: truncatedSVD
def dim_red__autoenc(data, data_cns, outcome_of_interest = None):
    data_cns = [v for _,v in data_cns]
    icpc_cols = filter_strings_regex(data_cns, 'icpc_\w+_\d+')
    col_idxs = [i for i,v in icpc_cols]
    cols_to_keep_idxs = [i for i in range(data.shape[1]) if i in col_idxs]
    icpc_data = data[:, cols_to_keep_idxs]
    icpc_data_cns = [v for i,v in icpc_cols] 
    # X_train, X_test = train_test_split(icpc_data, test_size=0.2, random_state=42)
    n_embeddings = 100
    autoencoder = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('svd', TruncatedSVD(n_components=n_embeddings)),
    ])
    X = icpc_data
    X_encoded = autoencoder.fit_transform(X)
    logger(f"Dimensionality reduction via autoencoder of {X.shape[1]} down to {n_embeddings}")
    # X_reconstructed = autoencoder.inverse_transform(X_encoded)
    X_encoded = csr_matrix(X_encoded)
    data, data_cns = remove_csr_columns(icpc_cols, data, data_cns) # remove columns where DR was done
    # add embeddings
    data = hstack([data, X_encoded])
    data_cns += [f'm_icpc_emb_{x}' for x in range(n_embeddings)]
    return (data, data_cns)



def first_alpha(s):
    matches = re.findall('\d*([a-zA-Z])+', s)
    if matches == []:
        return ''
    return matches[0]

def dem_red__expert_knowledge_contact_type(data, data_cns): 
    cns_idxs = llz(data_cns,0)
    data_cns = llz(data_cns,1)
    n_cols_start = data.shape[1]
    ctyp_cols = filter_strings_regex(data_cns, '_ctyp_\d+') # get ctyp cols and ther indexes
    time_bins = get_time_quantized_period_names(TIME_BINS_DAYS[:-1])
    # get their actual code values 
    ctyp_codes = [(c_col, one_hot_decode_varname(c_col, code_lookup_dicts)) for _,c_col in ctyp_cols]
    
    # group based on first char of ctyp 
    first_char = uniq([first_alpha(x) for _,x in ctyp_codes if first_alpha(x) != ''])
    n_column_vals = None
    n_column_nms = []
    for tb in time_bins:
        for char_group in first_char: # char_group = first_char[0]
            c_codes = [ i for i,v in ctyp_codes if first_alpha(v) == char_group and i.startswith(tb) ]

            c_idx = [i for i,v in ctyp_cols if v in c_codes]
            c_matrix = data[:, c_idx]
            n_column_vals = hstack([n_column_vals, c_matrix.sum(axis=1)]).tocsr() # e.g. tb 0_48m, cg c,  c2 0.3 , c24 0.1 => c 0.4
            n_column_nms.append(f"{tb}_ctyp_{char_group}")

    # remove old columns 
    cols_to_keep = [x for x in range(data.shape[1]) if x not in [i for i,_ in ctyp_cols ]]
    data = data[:, cols_to_keep]
    data_cns = [x for x in data_cns if x not in [v for _,v in ctyp_cols ]]

    # add new ones 
    data = hstack([data, n_column_vals])
    data_cns += n_column_nms
    data_cns = tet(data_cns)
    logger(f"Dimensionality reduction of ctyps via expert knowledge of {n_cols_start} down to {data.shape[1]}")
    return (data, data_cns)

def dim_red__expert_knowledge_atc(data, data_cns, max_atc_level=3):
    '''
    
    '''
    rev_lookups = code_lookup_dicts['rev-lookup']['atc_code']
    n_cols_start = data.shape[1]
    if type(data_cns[0]) == tuple: # in case they are pairs
        data_cns = [v for _,v in data_cns]
    atc_cols = filter_strings_regex(data_cns, 'atc_code_\d+')
    atc_codes = [(c_col, one_hot_decode_varname(c_col, code_lookup_dicts)) for _,c_col in atc_cols]

    time_periods = uniq(["_".join(x.split("_")[0:2]) for x,_ in atc_codes])
    atc_var_names = uniq(["_".join(x.split("_")[2:5]) for x,_ in atc_codes])

    # about 40 atc codes in the system do NOT have the max length, solution for now: pad them !
    non_complete_atc_codes = [ atc_code for c_col, atc_code in atc_codes if len(atc_code) != atc_max_len ]
    atc_codes = [(c_col, atc_code.ljust(atc_max_len,'_') ) for c_col, atc_code in atc_codes]
    assert all([ len(atc_code) == atc_max_len for c_col, atc_code in atc_codes]) # all atc codes have exactly 7 characters

    trunkated_atc_codes = [(c_col, atc_code[:atc_truncate_after_n_chars]) for c_col, atc_code in atc_codes]
    all_atc_codes = [s for _, s in atc_codes]
    all_tatcs = [s for _,s in trunkated_atc_codes]
    mapping_cat_to_atc = {f"cats0_{tatc}" : uniq([atc for atc in all_atc_codes if atc.startswith(tatc)]) for tatc in all_tatcs}

    n_df = None
    total_n_iterations = len(mapping_cat_to_atc) * len(time_periods) * len(atc_var_names)
    iter_counter = 0
    t0 = logger("Starting dim_red__expert_knowledge_atc ...")
    t1 = t0
    log_every_n = 1000 if SUBSAMPLE_DATA else 100
    for n_code, old_codes in mapping_cat_to_atc.items(): # n_code, old_codes = list(mapping_cat_to_atc.items())[0]
        for t_period in time_periods:  # t_period = time_periods[0]
            for atc_var_name in atc_var_names: # atc_var_name = atc_var_names[0]
                iter_counter += 1
                c_atc_codes = [(col_name, i_code) for col_name, i_code in atc_codes if i_code in old_codes]
                c_atc_codes = [(a,b) for a,b in c_atc_codes if a.startswith(t_period)]
                c_atc_codes = [(a,b) for a,b in c_atc_codes if "_".join(a.split("_")[2:5]) == atc_var_name]
                c_idxs = [i for i,v in atc_cols if v in [x for x,_ in c_atc_codes]]
                old_vals = data[:, c_idxs].toarray()
                n_col = f"{t_period}_{atc_var_name}_{n_code}"
                if iter_counter % log_every_n == 0:
                    t1 = logger(f"Done with {n_col} ({iter_counter}/{total_n_iterations})", t1) 
                n_val = pd.DataFrame(np.sum(old_vals, axis=1), columns=[n_col])
                n_df = pd.concat([n_df, n_val], axis=1)

    data, data_cns = remove_csr_columns(atc_cols, data, data_cns)
    data = hstack([data, csr_matrix(n_df)])
    data_cns += list(n_df.columns)
    logger(f"Dimensionality reduction of atc codes (max_atc_level = {max_atc_level}) via expert knowledge of {n_cols_start} down to {data.shape[1]}", t0)

    return (data, data_cns)

def dim_red__expert_knowledge(data, data_cns, outcome_of_interest = None):
    """
    Orchestrator function, calls dim reduction techniques for needed data types.
    """
    # data, data_cns = dem_red__expert_knowledge_contact_type(data, data_cns)
    data, data_cns = dim_red__expert_knowledge_icpc(
            code_lookup_dicts,
            data,
            data_cns,
            outcome_of_interest = outcome_of_interest, 
            remove_empty_cols = F
            )
    if len(try_match_strings(data_cns, '_atc_code')) > 0:
        data, data_cns = dim_red__expert_knowledge_atc(data, data_cns)
    return (data, data_cns)
    
def __wrapper_dim_red__expert_knowledge(params):
    return dim_red__expert_knowledge(data = params['data'],
                data_cns = params['data_cns'], 
                outcome_of_interest = params['outcome_of_interest'])

def dim_red__expert_knowledge_multiprocess(data, data_cns, outcome_of_interest = None,
                                    n_processes = 1 if SUBSAMPLE_DATA else 10): 
    # n_processes = 1
    rows_per_worker = determine_n_records_per_split(data.shape[0], n_processes)
    logger(f"Going to run on {n_processes} cores, will use {rows_per_worker[0]} records per job")
    start, inputs = 0, []
    for c_rows_per_worker in rows_per_worker:
        c_data = data[start:(start+c_rows_per_worker),]
        c_params = { 'data' : c_data, 'data_cns' : data_cns, 'outcome_of_interest' : outcome_of_interest }
        inputs += [c_params]
        start += c_rows_per_worker
    
    results = try_run_multiprocess(items = inputs, worker_fn=__wrapper_dim_red__expert_knowledge, n_processes=n_processes)
    return (vstack([x[0] for x in results]), results[0][1])



def remove_ohe_redundant_cats(data, data_cns):
    logger(f"OHE_REMOVE_FIRST_CATEGORY = {OHE_REMOVE_FIRST_CATEGORY}")
    if OHE_REMOVE_FIRST_CATEGORY:
        # remove last category from each var 
        vars_without_t_periods = uniq(["_".join(x.split("_")[2:]) if len(re.findall('\d+_\d+_\w+', x)) != 0 else x for x in data_cns ])
        vars_with_cat_nums = uniq([x for x in vars_without_t_periods if len(re.findall(r'.*_\d+$', x)) != 0])
        vars_without_cat_nums = uniq(["_".join(x.split("_")[:-1]) for x in vars_with_cat_nums])
        col_idxs = set()
        should_skip_var = lambda v: 'postcode' in v or '_ctyp_' in v
        for c_var in vars_without_cat_nums: # c_var = 'm_ctyp' # c_var =vars_without_cat_nums[10] 
            if should_skip_var(c_var):
                continue
            matches = [x for x in vars_with_cat_nums if len(re.findall(c_var,x))>0]
            if len(matches) < 2:
                continue
            last_cat_num = sorted([int(x.split("_")[-1]) for x in matches])[-1]
            last_cat = f"{c_var}_{last_cat_num}"
            all_cols_matching = [x for x in data_cns if x.endswith(last_cat)]
            if not all_cols_matching:
                continue
            logger(f"Dropping columns {all_cols_matching}")
            col_idxs = col_idxs.union(set( [i for i,v in enumerate(data_cns) if v in all_cols_matching]))
        col_idxs = list(col_idxs)
        cols_to_keep_idxs = [i for i in range(data.shape[1]) if i not in col_idxs]
        data = data[:, cols_to_keep_idxs]
        data_cns = [v for i,v in enumerate(data_cns) if i in cols_to_keep_idxs] 
    return data,data_cns


MAPPING_DIM_REDUC_TECHNIQUE = {
                "autoencoder": dim_red__autoenc,
                "expert_knowledge" : dim_red__expert_knowledge_multiprocess 
                }

# Read input
t0 = logger(f"loading data from {infile}, {infile_colnames}, {infile_code_lookups}")
data = read_pickle(infile)
data_cns = read_pickle(infile_colnames)
code_lookup_dicts = read_pickle(infile_code_lookups)
logger(f"loaded data with dimensionality of {data.shape[0]}:{data.shape[1]}", t0)
try_print_list(data_cns, logger)

# find last time period -> split dataset into X and y
time_groups = sorted(uniq(["_".join(x.split("_")[0:2]) for x in data_cns if try_regex('\d+_\d+_\w+', x)]))


outcome_of_interest = (data_cns.index('y_HF'), 'y_HF')
logger(f"Using {outcome_of_interest[1]} as outcome of interest (idx = {outcome_of_interest[0]})")


# split dataset into data_to_fit_on and Y
#id_col = [(i,x) for i,x in enumerate(data_cns) if x == 'id'][0]
#fit_cns = try_sd(range(len(data_cns)), [id_col[0]])
fit_cns = try_sd(range(len(data_cns)), [outcome_of_interest[0]])
data_to_fit_on = data[:, fit_cns ] # X
fit_cns = [(i, data_cns[i]) for i in fit_cns]
Y = data[:,  outcome_of_interest[0] ].todense() 
del data
Y = np.squeeze(np.asarray(Y))
red_fn = MAPPING_DIM_REDUC_TECHNIQUE[dim_red_technique]
logger(f"Using DIM_REDUC_TECHNIQUE:= {dim_red_technique}")
data_to_fit_on, fit_cns = red_fn(data_to_fit_on, fit_cns, outcome_of_interest)
data_to_fit_on, fit_cns = remove_ohe_redundant_cats(data_to_fit_on, fit_cns)

data_to_fit_on.shape
#[vals(X[X['id'] == convert_pat_id_str2float('86938|12') ][c]) for c in cns(X) if 'icpc_HF' in c] 
# prepare data for clustering
X = pd.DataFrame(data_to_fit_on.toarray())
del data_to_fit_on
X.columns = fit_cns

protected_cols = ['id', VAR_FOLLOW_UP_DATE]
unprotected_cols = try_sd(cns(X), protected_cols)

# remove any informationless columns
single_val_cols = [c for c in unprotected_cols if len(set(X[c].values.tolist())) < 2]
logger(f"Removing ({len(single_val_cols)}) columns with only single value: {single_val_cols}")
X  = X[try_sd(cns(X), single_val_cols)]

def is_sufficient_contrast(vals, Y, contrast_thresh = 50 if not SUBSAMPLE_DATA else 1):
    vals_y1 = [v for v,y in zip(vals, Y) if y ==1]
    vals_y0 = [v for v,y in zip(vals, Y) if y ==0]
    return len(vals_y0) > contrast_thresh  and len(vals_y1) > contrast_thresh

unprotected_cols = try_sd(cns(X), protected_cols)
insufficient_contrast_cols = [c for c in unprotected_cols if not is_sufficient_contrast(X[c].values.tolist(), Y)] 
logger(f"Removing ({len(insufficient_contrast_cols)}) columns with insufficient contrast: {insufficient_contrast_cols}")
X  = X[try_sd(cns(X), insufficient_contrast_cols)]

dist_cols = llz(filter_strings_regex(unprotected_cols, '_dist$'), 1)
dist_cols_to_keep = llz(filter_strings_regex(unprotected_cols, '(_twPA_|_mx_)dist$'), 1)
dist_cols_to_drop = try_sd(dist_cols, dist_cols_to_keep)
logger(f"Removing ({len(dist_cols_to_drop)}) unnecesarry columns from topic-distance metrics: {dist_cols_to_drop}")
X  = X[try_sd(cns(X), dist_cols_to_drop)]
unprotected_cols = try_sd(cns(X), protected_cols)
X = X[protected_cols + unprotected_cols]
logger(f"X.shape = {dim(X)}")
model_inputs = {
    "X" : X,
    "Y" : Y,
    "outcome_of_interest" : outcome_of_interest
}

# Save data
save_pickle(oufile, model_inputs)
logger("DONE", start_time)
print("DONE")
