print(
    '''
    # WHAT THIS SCRIPT DOES:
    # takes only the [ episode / consult / medication ] dates and sets the flwp start times for the control patients 
    # Author: T.Y.
'''
)

# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn

# get namespace
from namespaces import get_ns_name
import namespaces
ns_name = get_ns_name(__file__)
ns = getattr(namespaces, ns_name)

# parse commandline args
cmd_args = parse_commandline_args(verbose=True, required_extra_args=ns.required_extra_args)

IS_DEBUG = cmd_args["IS_DEBUG"]
SUBSAMPLE_DATA = cmd_args["SUBSAMPLE_DATA"]
script_params = { k : cmd_args[k] for k in ns.required_extra_args }
ns.__dict__.update(script_params)

logger = get_default_logger_fn(f"{__file__}")

check_if_debugging(IS_DEBUG)
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
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
cached_call = lambda fn, override_cache=T, cache_fn=None, **kwargs : try_cached_call(fn, io_r=read_pickle, io_c=pickle_exists, io_w=save_pickle, override_cache=override_cache, cache_fn=cache_fn, **kwargs)
cached_name = lambda cache_nm, parent_file=__file__: parent_file.split('/')[-1:][0].split('.')[0] + "." + f"{cache_nm}{subsampled_str}"
start_time = logger("Start running...")
# Boilerplate end
from try_utils import __init_pats_dict_time_unbiased
import gc
db_suffixes = ["AHA", "ANH"]
# Note: only the AF/VHD paths are exercised below, so the exact provenance of the HF flag here doesn't matter for this script
tagged_ICPC_HF_AF_VHD_files = [f"hf_af_vhd_tagged_{db_suffix}{subsampled_str}.pkl" for db_suffix in db_suffixes]

in_file_patients_d = {
    "AHA" : "patients_AHA.parquet",
    "ANH" : "patients_ANH.parquet"
}
in_file_episodes_d = { 
    "AHA": "ty/episodes_AHA.csv",
    "ANH": "ty/episodes_ANH.csv"
}
in_file_journals_d = {
    "AHA": "ty/journals_AHA.csv",
    "ANH": "ty/journals_ANH.csv"
}

pat_ids = None 
pat_tags = None

skip_follow_up_time_filtering = F
out_file_d = {
        "AHA" : f"pats_dict_AHA{subsampled_str}_time_biases.pkl",
        "ANH" : f"pats_dict_ANH{subsampled_str}_time_biases.pkl"
}

nrows_pat = 50000 if SUBSAMPLE_DATA else None 
nrows_ep = 50000 if SUBSAMPLE_DATA else None 
nrows_j = 50000 if SUBSAMPLE_DATA else None
batch_size = STEP1_BATCH_SIZE
episode_cols = EPISODE_COLS
journal_cols = JOURNAL_COLS

# CHECKPOINT - continue to journals
def merge_episodes_journals_on_patient_level(promise_patients, promise_episodes, promise_journals, tag):
    # :: Step 4 - merge records into a single object with patient as the first-class citizen
    t0 = logger(f'starting pats dict building (batch_size={batch_size})....')
    p_key = MAPPING_COMPOSITE_KEYS['patient']['key_nm']
    # fetch patients,  recall on this level pat_id is unique on its own!
    pat_ids, pat_tags, patients = cached_call(promise_patients, override_cache=T, cache_fn = cached_name(f'promise_patients_{tag}')) # pat_ids here follow order as in patients df 
    pat_cols = PATIENT_COLS + pat_tags
    pat_ids_nonempty_eps, episodes, outcome_df = cached_call(promise_episodes, pat_ids = pat_ids, override_cache=T, cache_fn = cached_name(f'promise_episodes_{tag}'))
    episodes = episodes[[  'episode_start_date', 'ptnt_prc_id', 'adj_HF_diag', 'ptnt_prc_epi_id' ]]
    episode_cols = try_si(EPISODE_COLS, cns(episodes))
    pat_ids_nonempty_js, journals = cached_call(promise_journals, pat_ids= pat_ids_nonempty_eps, outcome_df = outcome_df, override_cache=T, cache_fn = cached_name(f'promise_journals_{tag}'))
    journals = journals[[ 'journal_id', 'journal_datetime', 'ptnt_prc_id', 'adj_HF_diag', 'ptnt_prc_epi_id', 'ptnt_prc_jrnl_id' ]]
    journal_cols = try_si(JOURNAL_COLS, cns(journals))
    assert try_sdui(outcome_df[p_key], pat_ids) == []

    # xx = read_pickle(f'___ch_0_pre_process.__init_time_unbiased_{tag}.pkl')
    # strip only to date data

    # add patient outcome and follow-up date
    patients = pd.merge(patients, outcome_df, on = p_key, how='left').reset_index(drop=T)

    #  *************** episodes with journals  ************
    episodes_dict = {}
    e_id_vals = vals(episodes['ptnt_prc_epi_id']) if len(episodes) > 0 else []
    n_batches = round_up(len(e_id_vals) / batch_size )
    t0 = None
    for c_batch in range(n_batches):
        batch_start = c_batch*batch_size
        batch_end = batch_start+batch_size
        e_ids = e_id_vals[batch_start:batch_end] # batch 1, e idx 859 , ANH , '86938|12|7812'
        t0 = logger(f"processing episodes {batch_start} to {batch_end}... ({c_batch}/{n_batches})", t0)
        # get episodes from current e_ids
        c_eps = episodes[episodes['ptnt_prc_epi_id'].isin(e_ids)]
        # 
        p_ids = uniq(vals(c_eps['ptnt_prc_id']))

        if nuniq(c_eps.ptnt_prc_epi_id) != len(e_ids):
            raise Exception("nuniq(c_eps.ptnt_prc_epi_id) != len(e_ids):")
        episodes.drop(c_eps.index, inplace=True)

        if nrow(journals) > 0:
            if SUBSAMPLE_DATA:
                logger(f"DEBUG SUBSAMPLE: assigning jounrnals to random episodes available (useful when sampling small data)")
                journals['ptnt_prc_epi_id'] = random.choices(e_ids, k = nrow(journals))
            c_js = journals[journals['ptnt_prc_epi_id'].isin(e_ids)] #   journals[journals['ptnt_prc_epi_id'] == '86938|12|7812'] , Ok!
            #j_e_ids = uniq(vals(c_js['ptnt_prc_epi_id']))
            journals.drop(c_js.index, inplace=True)
            # go over each pat_id from current batch of episodes...
            for c_i in range(len(p_ids)):
                p_id = p_ids[c_i]
                c_c_eps = c_eps[c_eps['ptnt_prc_id'].isin([p_id])] # get episodes for p_id
                c_e_ids = uniq(vals(c_c_eps['ptnt_prc_epi_id'])) # get ep_ids for p_id  # "8976|2|1050498" in c_e_ids
                c_c_eps_props = c_c_eps[episode_cols + ['ptnt_prc_epi_id'] ].to_dict('records') # make  list of episode dicts for p_id
                
                for c_j in range(len(c_e_ids)): # go over each ep_id of current p_id
                    e_id = c_e_ids[c_j] 
                    c_c_js = c_js[c_js['ptnt_prc_epi_id'] == e_id] # get journals for current ep_id 
                    j_props = c_c_js[journal_cols].to_dict('records') # make them into list of dicts 
                    j_props = sorted(j_props, key= lambda item: item['journal_datetime'], reverse=F)
                    c_e_idx = [i for i,v in enumerate(c_c_eps_props) if v['ptnt_prc_epi_id'] == e_id][0]
                    c_c_eps_props[c_e_idx]['JOURNALS'] = j_props # add them to current ep_id entry 
                for i, _ in enumerate(c_c_eps_props):
                    if 'ptnt_prc_epi_id' in c_c_eps_props[i]:
                        del c_c_eps_props[i]['ptnt_prc_epi_id']
                    else:
                        logger(f'WARNING: ptnt_prc_epi_id NOT in c_c_eps_props[i] = {c_c_eps_props[i]}')

                prev_ep_props = episodes_dict[p_id] if p_id in episodes_dict else []
                # sort episodes based on start_date 
                c_c_eps_props = sorted(prev_ep_props + c_c_eps_props, key= lambda item: item['episode_start_date'], reverse=T)
                episodes_dict[p_id] = c_c_eps_props

    p_ids_with_episodes = list(episodes_dict.keys())
    pats_dict = {}
    p_id_vals = vals(patients['ptnt_prc_id'])
    n_batches = round_up(len(p_id_vals) / batch_size )
    t0 = None
    logger(f"Begin processing patients + episodes/journals (n_pat_ids =  {len(p_id_vals)})")
    #  *************** patients + episodes/journals  ************
    for c_batch in range(n_batches):
        batch_start = c_batch*batch_size
        batch_end = batch_start+batch_size
        p_ids = p_id_vals[batch_start:batch_end]
        t0 = logger(f"processing patient {batch_start} to {batch_end}... ({c_batch}/{n_batches})", t0)

        c_pats = patients[patients['ptnt_prc_id'].isin(p_ids)]
        if nuniq(c_pats.ptnt_prc_id) != len(p_ids):
            raise Exception("nuniq(c_pats.ptnt_prc_id) != len(p_ids)")

        patients.drop(c_pats.index, inplace=True)
        p_props = c_pats[pat_cols].to_dict('records')
        del c_pats
    
        for c_i in range(len(p_ids)):
            p_id = p_ids[c_i]
            if p_id not in p_ids_with_episodes:
                continue
            pats_dict[p_id] = p_props[c_i]
            pats_dict[p_id]['Episodes'] = episodes_dict[p_id]

    
    return pats_dict

t0 = logger('Starting to read input files....')
for db_suffix in db_suffixes:
    gc.collect()
    out_file = out_file_d[db_suffix]
    in_file_patients = in_file_patients_d[db_suffix]
    in_file_episodes = in_file_episodes_d[db_suffix]
    in_file_journals = in_file_journals_d[db_suffix]
    promise_read_patients = lambda in_file: lambda : read_patients(in_file, 
                                                    nrows = nrows_pat,
                                                    pat_ids = pat_ids,
                                                    pat_tags = pat_tags
                                                    )
    promise_read_episodes = lambda in_file: lambda pat_ids=[]: read_episodes(in_file,
                                                    nrows = nrows_ep,
                                                    pat_ids = pat_ids,
                                                    skip_follow_up_time_filtering=skip_follow_up_time_filtering,
                                                    is_debug = SUBSAMPLE_DATA
                                                    )

    promise_read_journals = lambda in_file: lambda pat_ids=[], outcome_df=None : read_journals(in_file,
                                                    nrows = nrows_j,
                                                    pat_ids = pat_ids,
                                                    outcome_df = outcome_df,
                                                    skip_follow_up_time_filtering=skip_follow_up_time_filtering,
                                                    is_debug = SUBSAMPLE_DATA
                                                    )


    pats_dict = cached_call(
    merge_episodes_journals_on_patient_level, 
        promise_patients = promise_read_patients(pqt_dir/in_file_patients), 
        promise_episodes = promise_read_episodes(pqt_dir/in_file_episodes),
        promise_journals = promise_read_journals(pqt_dir/in_file_journals),
        tag = db_suffix,
    override_cache=T, cache_fn = cached_name(f'merge_eps_js_ps_{db_suffix}')
    )


    pats_dict = cached_call(
    __init_pats_dict_time_unbiased, 
    x = pats_dict,
    MATCH_DATE_THRESHOLD = 90 if not SUBSAMPLE_DATA else 9000,
    IS_DEBUG = SUBSAMPLE_DATA,
    override_cache=T, cache_fn = cached_name(f'__init_time_unbiased_{db_suffix}')
    )


    save_pickle(out_file, pats_dict)

_ = logger('pats dict write to file....DONE ', t0)
print("DONE")
