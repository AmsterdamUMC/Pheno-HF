print(
    '''
# WHAT THIS SCRIPT DOES:
# 1. Calls variable selection with different input data and config combinations 
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
from VariableSelection import run_it


res_filenames = []
infile = full_filename_pkl(ns.infile)
hp_params = {}

OVERRIDE_RES_FILE = T

if not SUBSAMPLE_DATA:
    # RF
    hp_params = {
        "n_estimators": [5000], 
        "max_depth": [5],
        "min_samples_split": [100] ,
        "max_leaf_nodes": [200]
                }


else:
    hp_params = {
        "n_estimators": [100, 200],
        "max_depth": [2],
        "min_samples_split": [20],
        "max_leaf_nodes": [20]
                }


logger(f"Running for {infile} ...")
res_filename = run_it(infile, hp_params, ns.outfile, override_res_file=OVERRIDE_RES_FILE, use_text_vars=ns.use_text_vars)
logger(f"Results saved in {res_filename}")

logger("DONE", start_time)
print("DONE")