# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES 
# scale values using RobustScaler
# Plots histograms of variable values (before and after scaling)
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
plot_hist_vals = lambda vals, **kwargs: try_plot_hist_vals(vals, outpath="plots/GMM_preprocess/", subsampled=SUBSAMPLE_DATA, **kwargs)
# Boilerplate end
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import RobustScaler

in_file = full_filename_pkl(ns.infile)
infile_colnames = full_filename_pkl(f"{ns.infile}_colnames")
infile_code_lookups = full_filename_pkl(ns.infile_code_lookups)

outfile = full_filename_pkl(ns.outfile)
outfile_scale_multipliers = full_filename_pkl(f"{ns.outfile}_scale_multipliers")
PLOT_HISTS = ns.plot_histograms

plot_fn = lambda vals, **kwargs:  plot_hist_vals(vals, **kwargs) if PLOT_HISTS else None

t0 = logger(f"loading X from {in_file}, {infile_colnames}, {infile_code_lookups}")
# Read input
tmp = read_pickle(in_file)
X = tmp["X"]
cns_X = cns(X)
code_lookup_dicts = read_pickle(infile_code_lookups)
# 3_pre_process.py was last (re-)run only for ATC, so we reuse the older lookup dicts for every other var (see OPEN_QUESTIONS.md)
cld_atc = read_pickle('cluster_fs1_lookup_dicts.pkl')
code_lookup_dicts['lookup']['atc_code'] = cld_atc['lookup']['atc_code']
code_lookup_dicts['rev-lookup']['atc_code'] = cld_atc['rev-lookup']['atc_code']
logger(f"loaded X with dimensionality of {dim(X)}", t0)
Y = tmp["Y"]
outcome_of_interest = tmp["outcome_of_interest"]
protected_cols = ['id', VAR_FOLLOW_UP_DATE, 'deceased_1']
del tmp

scalable_cols = try_sd(cns_X, protected_cols)
log_every_n = 100
for i,c_nm in enumerate(scalable_cols):
    if i % log_every_n == 0:
        logger(f"Done with {i}/{len(scalable_cols)} columns...")
    col_name = one_hot_decode_varname(c_nm, code_lookup_dicts)
    plot_fn(X[c_nm].values, title=col_name, outfile=f"hist_{c_nm}_noscale")

# perform scaling 
scaler = RobustScaler()
scaled_vals = scaler.fit_transform(X[scalable_cols]) 
save_pickle(outfile_scale_multipliers, scaler)

X = X[protected_cols]
X_scaled = pd.DataFrame(scaled_vals, columns = scalable_cols)
X = pd.concat([X, X_scaled], axis=1)
del X_scaled

for i,c_nm in enumerate(scalable_cols):
    col_name = one_hot_decode_varname(c_nm, code_lookup_dicts)
    plot_fn(X[c_nm].values, title=col_name, outfile=f"hist_{c_nm}_scale")

save_obj = {"X" : X,
            "Y": Y,
            "outcome_of_interest" : outcome_of_interest
            }

save_pickle(outfile, save_obj)


logger("DONE", start_time)
print("DONE")