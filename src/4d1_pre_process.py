# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES 
# scale values using MaxAbsScaler (not really essential for RF, but whatever)
# transforms age_days based on last followup date. Before this step age_days actually represents birth_date
# remove nan values via nan_to_num
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
plot_hist_vals = lambda vals, **kwargs: try_plot_hist_vals(vals, outpath="plots/4d1_pre_process/", subsampled=SUBSAMPLE_DATA, **kwargs)
# Boilerplate end
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import MaxAbsScaler

in_file = full_filename_pkl(ns.infile)
infile_colnames = full_filename_pkl(f"{ns.infile}_colnames")

outfile = full_filename_pkl(ns.outfile)
outfile_scale_multipliers = full_filename_pkl(f"{ns.outfile}_scale_multipliers")


PLOT_HISTS = ns.plot_histograms
plot_fn = lambda vals, **kwargs:  plot_hist_vals(vals, **kwargs) if PLOT_HISTS else None

t0 = logger(f"loading data from {in_file}, {infile_colnames}")
# Read input
data = read_pickle(in_file)
data_cns = read_pickle(infile_colnames)
logger(f"loaded data with dimensionality of {data.shape[0]}:{data.shape[1]}", t0)
# xxx = pd.DataFrame(data.todense(), columns = data_cns)
# len([c for c in unprotected_cols if len(set(xxx[c].values.tolist())) < 2])

data = data.tocsr()
protected_cols = ['id', VAR_FOLLOW_UP_DATE, 'y_HF']
protected_cols_idxs = [i for i,c in enumerate(data_cns) if c in protected_cols]
protected_cols_ixnms = [(i,c) for i,c in enumerate(data_cns) if c in protected_cols]

unprotected_cols = try_sd(data_cns, protected_cols)
unprotected_cols_ixnms = [(i,c) for i,c in enumerate(data_cns) if c in unprotected_cols]
unprotected_cols_idxs = [i for i,c in enumerate(data_cns) if c in unprotected_cols]

scale_multipliers = {}
np.nan_to_num(data.data, copy=F)
log_every_n = 100
logger(f"transform age_days based on last followup date")
age_idx = [ i for i,c in unprotected_cols_ixnms if c=='age_days'][0]
last_flwp_idx = [ i for i,c in protected_cols_ixnms if c==VAR_FOLLOW_UP_DATE][0]

age_vals = data[:,age_idx].todense()
last_flwp_vals = data[:,last_flwp_idx].todense()
age_vals  = last_flwp_vals+age_vals # since we stored -1*days_since_birth
# tmp = np.asarray(age_vals).reshape(-1)
# np.quantile(tmp/365, [0, 0.25, 0.5, 0.75, 1])
data = data.tolil()
data[:, age_idx] = age_vals
data = data.tocsr()


for i,ixnm in enumerate(unprotected_cols_ixnms):
    c_idx, c_nm = ixnm
    if i % log_every_n == 0:
        logger(f"Done with {i}/{len(unprotected_cols_ixnms)} columns...")

    plot_fn(data[:,c_idx].toarray().reshape(-1), title=c_nm, outfile=f"hist_{c_nm}_noscale")
    
    scale_multipliers[c_nm] = np.max((np.abs(data[:,c_idx])))
#sum([scale_multipliers[c] for c in unprotected_cols if len(set(xxx[c].values.tolist())) < 2])
#tmp =sorted([(c,str(scale_multipliers[c])) for c in unprotected_cols if len(set(xxx[c].values.tolist())) >= 2], key=lambda x:x[1])
#try_print_list(["; ".join(tp) for tp in tmp], logger)
save_pickle(outfile_scale_multipliers, scale_multipliers)

# perform scaling 
scaler = MaxAbsScaler()
scaled_data = scaler.fit_transform(data[:,unprotected_cols_idxs]) 
scaled_data = hstack([data[:,protected_cols_idxs], scaled_data])

# plotting histograms
for i,c_nm in unprotected_cols_ixnms:
    plot_fn(scaled_data[:,i].toarray().reshape(-1), title=c_nm, outfile=f"hist_{c_nm}_scale")

# xxx = pd.DataFrame(scaled_data.todense(), columns = data_cns)
# len([c for c in unprotected_cols if len(set(xxx[c].values.tolist())) < 2])
save_pickle(outfile, scaled_data)


logger("DONE", start_time)
print("DONE")