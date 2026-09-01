# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES:
# 1. Merges the two patient-level dicts from ANH and AHA
'''
)
# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
logger = get_default_logger_fn(__file__)

# get namespace
from namespaces import get_ns_name, parse_ns_val_bool
import namespaces
ns_name = get_ns_name(__file__)
ns = getattr(namespaces, ns_name)
# parse commandline args
cmd_args = parse_commandline_args(verbose=True, required_extra_args=ns.required_extra_args)

IS_DEBUG = cmd_args["IS_DEBUG"]
SUBSAMPLE_DATA = cmd_args["SUBSAMPLE_DATA"]
script_params = { k : parse_ns_val_bool(cmd_args[k]) for k in ns.required_extra_args }
ns.__dict__.update(script_params) # init input args into namespace

check_if_debugging(IS_DEBUG)
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
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
# Boilerplate end

db_suffixes = ["ANH", "AHA"]
full_filename_pkl = lambda fname : f"{fname}{subsampled_str}.pkl"
in_files = [full_filename_pkl(x) for x in ns.infiles.split(';')]
outfile = full_filename_pkl(ns.outfile)

pat_ds = {}
for i,db_suffix in enumerate(db_suffixes):
    pat_ds[db_suffix] = read_pickle(in_files[i])
    
keys_anh = list(pat_ds['ANH'].keys())
keys_aha = list(pat_ds['AHA'].keys())
ks_present_in_both = set(keys_anh).intersection(set(keys_aha))
assert len(ks_present_in_both) == 0 

logger(f"AHA with {len(keys_aha)} records")
logger(f"ANH with {len(keys_anh)} records")

t0 = logger("Merging two dbs..")
pat_ds['ANH'].update(pat_ds['AHA'])

len(pat_ds['ANH'].keys())
merged_d = pat_ds['ANH']
# del pat_ds
_ = logger("Merging two dbs..done", t0)
logger(f"merged dict size = {len(merged_d)}")

save_pickle(outfile, merged_d)

logger("DONE", start_time)
print("DONE")