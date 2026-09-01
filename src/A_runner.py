print(
    '''
# WHAT THIS SCRIPT DOES:
# 1. Invokes 2.2-preprocess-text.py (now called textEmbeddingT2V.py) with specific hyperparameter values 
    hyperparams suppoerted: embedding_model, concat_modifiers
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
from try_utils import *
from constants import *
import numpy as np
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
start_time = logger("Start running...")
cached_call = lambda fn, override_cache=F, **kwargs : try_cached_call(fn, io_r=read_pickle, io_c=pickle_exists, io_w=save_pickle, override_cache=override_cache, **kwargs)
# Boilerplate end
from textEmbeddingT2V import run_it

full_filename_pkl = lambda fname : f"{fname}{subsampled_str}.pkl"
in_file = full_filename_pkl(ns.infile)
out_file = full_filename_pkl(ns.outfile)

res_filenames = []
emb_models_to_try = ['doc2vec'] #if not SUBSAMPLE_DATA else ['RobertaForMaskedLM_embedding_model']
#xxx = try_read_pickle(EMBEDDING_MODEL_METADATA['RobertaForMaskedLM_embedding_model']['doc_embeddings_file'])
for embedding_model in emb_models_to_try:
    logger(f"Running for {embedding_model} ...")
    res_filename = run_it(
        logger, 
        embedding_model,
        d2v_dm,
        ns.include_medications_text,
        ns.include_measurements_text,
        in_file,
        out_file,
        ns.interim_out_file
        )
    res_filenames+= [res_filename]
    logger(f"Results saved in {res_filename}")

filenames_str = "',\n\t\t '".join(res_filenames)
filenames_str = "'" + filenames_str + "'"
logger(f"Use following list of files as inputs for next step: \n\t\t[{filenames_str}\n\t\t]")
logger("DONE")
try_log("DONE")


# hacky hack hack # kept running out of memory on hdbscan, 
# in future, I suggest: compute doc embeddings + save to disk; 
# compute umap embs + save to disk + unload doc embeddings;
# compute hdbscan labels + save to disk + unload umap embeddings; (modify min_cluster_size, min_samples, core_dist_n_jobs for oom-issues)
# load doc embeddings + compute topic vectors etc.
# but hdbscan only needs umap embeddings, which are tiny, also lower min_samples helps (default = min_cluster_size)
# notes further: a lot of docs get into the -1 (noise) label, could try to improve by:
# 1) reducing min_samples, or other hp-params of hdbscan
# 2) changing UMAP to higher dimension (5 seems a bit low to me... maybe try 10? or even 20, or 50?)
# idea for reducing number of topics - modify eps / min_sample for dbscan run on topic vectors
if 1 == 2:
    umap_embedding = try_read_pickle(EMBEDDING_MODEL_METADATA['doc2vec']['umap_embeddings_file']['0_48_text_edsoepx'])['UMAP']
    import hdbscan
    hdbscan_args = {'min_cluster_size': 300, 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'min_samples': 150,
                        'core_dist_n_jobs': -2} # default is 4, but much faster if you run it on all #cores -1 (downside, uses more memory....)
    cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_embedding)
    labels = cluster.labels_
    try_save_pickle(f=f"try_hdbscan_labs_{dt.now()}.pkl", 
        o = {"labels": labels})