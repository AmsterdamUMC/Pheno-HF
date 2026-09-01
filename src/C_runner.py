print(
    '''
# WHAT THIS SCRIPT DOES:
# 1. Runs StepMix (GMM) with different number of target clusters, using input file(s) from GMM_preprocess.
# 2. Identifies combination of clusters/input files that produces clusters with largest incidence and sufficient mass
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
from GaussMMStepMix import run_it
from sys import exit

logger = get_default_logger_fn(__file__)
start_time = logger("Start running...")
logger(f"Starting ... RANDOM_SEED = {RANDOM_SEED}")
res_filenames = []
hp_params = {
        "use_outcome" : [-1] ,
        "use_only_text_icpc" : [F]
        }
 
infile = full_filename_pkl(ns.infile)
outfile = full_filename_pkl(ns.outfile)
best_metrics = -1

logger(f"Using input file {infile}")

hp_params_to_try = expand_hp_params(hp_params)
vars_to_try = {"non-nested" : {'vars' :[], 'n_components' : list(range(3,30))
 }}
for c_params_to_try in hp_params_to_try:
    for vn,vtt in vars_to_try.items():
        logger(f"Trying hp_params  {c_params_to_try}")
        c_filenames = run_it(infile, c_params_to_try, f"{vn}_{outfile}", use_only_vars=vtt['vars'], n_components = vtt['n_components'])
        res_filenames += c_filenames
        logger(f"Results saved in {c_filenames}")
    break



filenames_str = "',\n\t\t '".join(res_filenames)
filenames_str = "'" + filenames_str + "'"
logger(f"Use following list of files as inputs for next step: \n\t\t[{filenames_str}\n\t\t]")
logger("DONE")


logger("DONE", start_time)
print("DONE")