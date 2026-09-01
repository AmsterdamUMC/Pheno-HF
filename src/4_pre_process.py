# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES 
# reads all saved batches from 3-pre-process.py, concatenates them into a single sparse matrix format
# removes duplicate text columns (bug from last step)
# adds boolean chronic flags (erroneously removed in last step)
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
import os
import gc
from scipy.sparse import csr_matrix, vstack, hstack, coo_matrix


#f'cluster_fs1{subsampled_str}_n_batches{suffix}.pkl'
n_batches = read_pickle(full_filename_pkl(ns.infile_nbatches))["n_batches"]
n_batches_append = read_pickle(full_filename_pkl(ns.infile_nbatches))["n_batches"]

infiles = [ full_filename_tsv(ns.infile, b) for b in range(n_batches) ]

append_attributes = ns.append_attributes.split(';')
infiles_append = [ full_filename_tsv(ns.infile_append, b) for b in range(n_batches_append) ] if ns.infile_append != '' else []

logger(f"Going to read {n_batches} batches, 0 till {n_batches-1}")

outfile = full_filename_pkl(ns.outfile) 

outfile_colnames = full_filename_pkl(f"{ns.outfile}_colnames")  #{ infix: f'cluster_fs1_column_names{infix}{suffix}.pkl'  for infix in infixes}  

n_processes = min(n_batches, 2) if SUBSAMPLE_DATA else max(os.cpu_count() - 3, 1)
if IS_DEBUG:
    n_processes = 1
logger(f"Going to run on {n_processes} cores")

def remove_dup_cols_txt(df, log=logger):
    cols = cns(df)
    txt_cols = [ x for x in cols if try_regex('_text', x) and not try_regex('\.1$', x)]
    dup_cols = [ x for x in cols if try_regex('_text', x) and try_regex('\.1$', x)]
    assert all([ tc == dc[:-2] for tc,dc in zip(txt_cols,dup_cols)])
    for tc,dc in zip(txt_cols,dup_cols):
        assert all(df[tc] == df[dc])

    df = df[try_sd(cols, dup_cols)]
    log(f"Removed {len(dup_cols)} duplicate text cols")
    log(f"df left with shape {dim(df)}")
    return df

def tte_to_binary(tte_data, censor_threshold):
    return [not pd.isnull(t) and t <= ct for t,ct in zip(tte_data, censor_threshold) ]

def get_tags_for_pats(pat_ids, flwp_ends):
    assert len(pat_ids) == len(flwp_ends)
    rel_items = { k:pats_dict[k] for k in pat_ids}
    res = pd.DataFrame.from_dict(rel_items).T
    for c in cns(res):
        res[c] = tte_to_binary(res[c].values.tolist(), flwp_ends)
        # logger(f"{c}:=")
        # logger(f"{try_table(res[c])}")

    res['id'] = res.index.values.tolist()
    res.reset_index(drop=T, inplace=T)
    res = res[ [ 'id' ] +  try_sd( cns(res), ['id'])   ] # make id first column for convention
    
    return res

def add_tags_for_pats(df):
    tags = get_tags_for_pats(df.id.values.tolist(), df.follow_up_LAST.values.tolist())
    df = pd.merge(df, tags, on ='id', how = 'inner')
    return df



def load_in_files(item):
    worker_num, infiles, append_attributes, infiles_append, out_file_colnames = item
    spm3x = None
    dm3x = None

    protected_cols = [ 'y_HF', 'age_days']
    init_dense_cols = ['id', VAR_FOLLOW_UP_DATE] 
    first_columns = init_dense_cols + protected_cols 
    sparse_cols = None
    dense_cols = None
    # pre-load infiles_append into x_app_all
    # why? we are not guaranteed that batch_i from infile will be the same ids as batch_i from infile_append
    x_app_all = None
    empty_fn = lambda s: None
    if infiles_append is not None:
        for in_file_append in infiles_append:
            if in_file_append is not None:
                x_app = try_read_pd_df(in_file_append)
                x_app = remove_dup_cols_txt(x_app, empty_fn)
                aa_cols = ['id'] + [ c for c in cns(x_app) for aa in append_attributes if aa in c ]
                x_app = x_app[aa_cols]
                x_app_all = pd.concat([x_app_all, x_app], axis=0)
    has_append = x_app_all is not None
    for f_idx, in_file in enumerate(infiles):
        log_once = lambda s: logger(s) if f_idx == 0 else None
        gc.collect()
        t0 = logger(f"Worker {worker_num} reading file {in_file}...", flush=T)
        x = try_read_pd_df(in_file)
        x = remove_dup_cols_txt(x, log_once) # fix duplicate text columns
        if has_append:
            remove_attributes = [ c for c in cns(x) for aa in append_attributes if aa in c ]
            aa_cols = ['id'] + [ c for c in cns(x_app) for aa in append_attributes if aa in c ]
            x = x[try_sd(cns(x), remove_attributes)]
            x = pd.merge(x, x_app_all, on='id', how='left').reset_index(drop=T)

        # remove any unwanted index columns
        idx_cols = [c for c in cns(x) if try_regex('(index|level|Unnamed)',c)]
        x  = x[try_sd(cns(x), idx_cols)]
        x = add_tags_for_pats(x)
        # add chronic illness tags (cause you were too quick to remove them!)
        try_print_col_counts_by_type(x, log_once)
        
        x['id'] = x['id'].apply(convert_pat_id_str2float)
        # dense + protected columns first !!!!
        cols_ordered = first_columns + try_sd(cns(x), first_columns)
        x = x[ cols_ordered ]

        if sparse_cols is None:
            sparse_cols = sorted([x for x in cols_ordered if try_regex('(^t_|_\d+$)', x) and not try_regex('_tw', x)])
        if dense_cols is None:
            dense_cols = try_sd(cols_ordered, sparse_cols)
            dense_cols = first_columns + sorted(try_sd(dense_cols, first_columns))

        if worker_num == 0 and f_idx == 0:
            logger(f"Saving {len(cns(x))} colnames...", flush=T)
            save_pickle(out_file_colnames, dense_cols+sparse_cols)

        c_spm3x =  hstack([csr_matrix(x[col].values.reshape(-1,1)) for col in sparse_cols]) 
        
        c_dm3x = x[dense_cols].values
        if spm3x is not None:
            spm3x = vstack([spm3x, c_spm3x])
            dm3x = np.vstack([dm3x, c_dm3x ])
        else:
            spm3x = c_spm3x
            dm3x = c_dm3x
        logger(f"... Done. Current df ({spm3x.shape[0]}; {spm3x.shape[1]}) ", t0, flush=T)
    final_ma3x = hstack([dm3x, spm3x])
    return final_ma3x



pats_dict = read_pickle(full_filename_pkl(ns.infile_pats_dict))

chronic_condition_keys = [
    'death',
    'cvd_in_family',
    'coronary_artery_disease',
    'atrial_fibrillation',
    'heart_murmur',
    'valvular_heart_disease',
    'hypertension',
    'stroke',
    'copd',
    'diabetes_mellitus',
    'chronic_kidney_disease',
    'alcohol_abuse',
    'tobacco_use',
    'obesity',
    'material_deprivation',
    'AF',
    'VHD'
]
cc_t_cols = [f"t_{c}" for c in chronic_condition_keys]

def filter_pat_item_chronic_tags_only(pat):
    return { k:pat[k] for k in cc_t_cols }
pats_dict = {k:filter_pat_item_chronic_tags_only(v) for k,v in pats_dict.items() }


files_per_split = round_down(n_batches/n_processes)
files_per_split = [files_per_split]*n_processes
for remainder_files in range(n_batches % n_processes):
    files_per_split[remainder_files] += 1

files_per_split_append = round_down(n_batches_append/n_processes)
files_per_split_append = [files_per_split_append]*n_processes
for remainder_files in range(n_batches_append % n_processes):
    files_per_split_append[remainder_files] += 1

start = 0
start_append = 0
files_split = []
files_append_split = []
for i in range(n_processes):
    files_split.append(infiles[start:(start+files_per_split[i])])
    files_append_split.append(infiles_append[start_append:(start_append+files_per_split_append[i])] if infiles_append != [] else None)
    start += files_per_split[i]
    start_append += files_per_split_append[i]
    t1 = logger(f"Going to start {n_processes} processes for loading files...")
    items = [(i, fs, append_attributes, files_append_split[i], outfile_colnames) for i,fs in enumerate(files_split)]
    results = try_run_multiprocess(worker_fn=load_in_files, items=items, n_processes=n_processes)
    all_keys = try_read_pickle(f'all_keys{subsampled_str}.pkl')  


    data = vstack(results)
    data_df = pd.DataFrame(data.todense())
    data_df['id'] = data_df[0].apply(convert_pat_id_float2str).values
    x_ids = set(data_df['id'].values)
    all_keys = [k for k in all_keys if k in x_ids]
    data_df = data_df.set_index('id')
    # restore order in data rows as was before shuffle in previous script
    data_df = data_df.loc[all_keys].reset_index(drop=T)
    data = coo_matrix(data_df.values)
    logger(f"Running {n_processes} processes for loading files...done", t1)
    del results
    save_pickle(outfile, data)
logger("DONE", start_time)
print("DONE")

# sanity check, subsampled patient 58608|9 should have 57 journals in 0_48



