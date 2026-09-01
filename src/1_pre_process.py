print(
    '''
    # WHAT THIS SCRIPT DOES:
    # 1. Reads raw csv input for episodes/journals and parquet input for patient/practices (both ANH and AHA)
    # 2. Marks any patient record which had HF diagnoses at any point in their history 
    # 4. Merges all records on patient level, 
    #      one patient record := patient-level attributes + [episodes]
    #      one episode record := episode-level attributes + [journals]
    #      one journal record := journal-level attributes
    # Outputs: "pats_dict_{AHA/ANH}{subsampled_str}.pkl"
    # WARNING: 62GiB RAM is unsufficient, ensure there are at least 30GiB of additional swapspace available!  
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

logger = get_default_logger_fn(f"{__file__}{ns.target_condition}_t{ns.tagging_mode}")

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
cached_call = lambda fn, override_cache=F, cache_fn=None, **kwargs : try_cached_call(fn, io_r=read_pickle, io_c=pickle_exists, io_w=save_pickle, override_cache=override_cache, cache_fn=cache_fn, **kwargs)
cached_name = lambda cache_nm, parent_file=__file__: parent_file.split('/')[-1:][0].split('.')[0] + "." + f"{cache_nm}{subsampled_str}"
cached_name_0_pre = lambda cache_nm: cached_name(cache_nm, parent_file='0_pre_process.py')
start_time = logger("Start running...")
# Boilerplate end
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

USE_ONLY_AFPOS = ns.target_condition == 'AF'
USE_ONLY_VHDPOS = ns.target_condition == 'VHD'
USE_ONLY_HFPOS = ns.target_condition == 'HF'
TAGGING_MODE = ns.tagging_mode == "T"

assert USE_ONLY_AFPOS + USE_ONLY_VHDPOS + USE_ONLY_HFPOS <= 1
if TAGGING_MODE:
    assert USE_ONLY_AFPOS + USE_ONLY_VHDPOS + USE_ONLY_HFPOS == 0


def get_COND_pos_pat_ids(target_outcome):
    '''
    taret_outcome = one of atrial_fibrillation, valvular_heart_disease, or heart_failure
    '''
    import pandas as pd
    dfs = []
    pat_tags = [ [f't_{x}', f't_text_{x}', f'text_{x}', x] for x in ['AF', 'VHD', 'HF']] 
    pat_tags = ['pat_prac_id'] + [ i for o in pat_tags for i in o]  
    for c_file in tagged_ICPC_HF_AF_VHD_files:
        c_df = read_pickle(c_file)
        c_df['patient_id'] = vals(c_df.index)
        dfs+= [c_df]

    tdf = pd.concat(dfs, ignore_index=T, sort=F)
    tdf['heart_failure'] = ~tdf['t_hf'].isnull() # note: for HF we are using the adjudicated HF tag, not the icpc one
    #try_table(tdf['atrial_fibrillation'])
    pos_df = tdf[tdf[target_outcome]]
    logger(f"{target_outcome} N pos = {nrow(pos_df)}")
    recent_pos_df = pos_df[pos_df[f"t_{target_outcome}"] >= ADJ_TIME_START_TIMESTAMP]
    logger(f"{target_outcome} N pos after {ADJ_TIME_START_YEAR} = {nrow(recent_pos_df)}")
    pat_ids = vals(recent_pos_df['pat_prac_id'])
    logger(f"@get_COND_pos_pat_ids({target_outcome}): N patients = {len(pat_ids)}")
    return pat_ids, recent_pos_df[pat_tags]

pat_ids = None # read_pickle('ids_of_interest.pkl') # 
pat_tags = None
if USE_ONLY_HFPOS:
    _, pat_tags = get_COND_pos_pat_ids('HF')
    lukas_adj_df = read_pickle(infile_Lukas_adjudicated)
    pat_ids, _ = get_HF_adj_pos_pat_ids(lukas_adj_df)

if USE_ONLY_AFPOS:
    pat_ids, pat_tags = get_COND_pos_pat_ids('AF')

if USE_ONLY_VHDPOS:
    pat_ids, pat_tags = get_COND_pos_pat_ids('VHD')

#pat_ids = read_pickle('ids_of_interest.pkl') # 
skip_follow_up_time_filtering = F
custom_infix = f"_{ns.target_condition}"   # to denote this is a custom subste of patients, not all
out_file_d = {
        "AHA" : f"pats_dict_AHA{custom_infix}{subsampled_str}.pkl",
        "ANH" : f"pats_dict_ANH{custom_infix}{subsampled_str}.pkl"
}

nrows_pat = 100000 if SUBSAMPLE_DATA else None 
nrows_ep = 200000 if SUBSAMPLE_DATA else None 
nrows_j = 200000 if SUBSAMPLE_DATA else None


batch_size = STEP1_BATCH_SIZE

def tag_outcome(promise_patients, db_prefix):
    # :: Step 4 - merge records into a single object with patient as the first-class citizen
    t0 = logger(f'starting pats dict tag_outcome....')

    p_key = MAPPING_COMPOSITE_KEYS['patient']['key_nm']
    # fetch patients,  recall on this level pat_id is unique on its own!
    pat_ids, _, patients = promise_patients() # pat_ids here follow order as in patients df 
    # patch practice_id onto patients (see OPEN_QUESTIONS.md re: extracting this into its own step)
    pat_prac_ids = { split_pat_prac_id(x)['pat_id'] : split_pat_prac_id(x)['prac_id']  for x in pat_ids}
    c_df = pd.read_parquet(old_data_dir/f"cohort_A-ICPC/cohort_ty_{db_prefix}{subsampled_str}.parquet")
    prac_ids = []
    iter_pat_ids_order = []
    for row in c_df.itertuples(): 
        pat_id = row.Index
        iter_pat_ids_order += [pat_id]
        if pat_id in pat_prac_ids:
            prac_ids += [pat_prac_ids[pat_id]]  
        else:
            prac_ids.append(None)
    if not SUBSAMPLE_DATA:
        assert [x for x in prac_ids if x is None] == []

    c_df['pat_prac_id'] = [f'{pat_id}|{prac_id}' for pat_id, prac_id in zip(vals(c_df.index), prac_ids)]
    assert [ i  for i,x in enumerate(zip(vals(c_df.index), c_df['pat_prac_id'])) if split_pat_prac_id(x[1])['pat_id'] != x[0]] == []
    save_pickle(f'hf_af_vhd_tagged_{db_prefix}{subsampled_str}.pkl', c_df)

# CHECKPOINT - continue to journals
def merge_episodes_journals_on_patient_level(promise_patients, promise_episodes, promise_journals, tag):
    # :: Step 4 - merge records into a single object with patient as the first-class citizen
    t0 = logger(f'starting pats dict building (batch_size={batch_size})....')
    p_key = MAPPING_COMPOSITE_KEYS['patient']['key_nm']
    # fetch patients,  recall on this level pat_id is unique on its own!
    pat_ids, pat_tags, patients = cached_call(promise_patients, override_cache=F, cache_fn = cached_name_0_pre(f'promise_patients_{tag}')) # pat_ids here follow order as in patients df 
    pat_cols = PATIENT_COLS + pat_tags
    pat_ids_nonempty_eps, episodes, outcome_df = cached_call(promise_episodes, pat_ids = pat_ids, override_cache=F, cache_fn = cached_name_0_pre(f'promise_episodes_{tag}'))
    pat_ids_nonempty_js, journals = cached_call(promise_journals, pat_ids= pat_ids_nonempty_eps, outcome_df = outcome_df, override_cache=F, cache_fn = cached_name_0_pre(f'promise_journals_{tag}'))
    time_biases = read_pickle(f'pats_dict_{tag}{subsampled_str}_time_biases.pkl')
    pat_start_time = lambda pat: pat['follow_up_LAST'] - pat['days_flwp']
    pat_start_times = [(pid, pat_start_time(time_biases[pid])) for pid in time_biases.keys()]
    df_start_times = pd.DataFrame(pat_start_times)
    df_start_times.columns = ['ptnt_prc_id', 'start_time']
    patients = patients.merge(df_start_times, on = 'ptnt_prc_id')
    episodes = episodes.merge(df_start_times, on = 'ptnt_prc_id')
    journals = journals.merge(df_start_times, on = 'ptnt_prc_id')
    outcome_df = outcome_df.merge(df_start_times[['ptnt_prc_id']], on = 'ptnt_prc_id')
    pat_ids = vals(patients.ptnt_prc_id)
    
    #pat_ids_tbs 
    assert try_sdui(outcome_df[p_key], pat_ids) == []
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
        # 21.FEB THIS IS WHERE IT BECOMES CLEAR SOMETHING IS NOT RIGHT. we see only 19 episodes with K78, 
        # whilst we selected only patients with AF, thus there should be at LEAST one episode with K78 per p_id
        # right now we get 19 out of 140 p_ids... 
        # why could the rest be missing?
        # -> is this the fault of 03_TableDistillation? 
        # No! 
        # All that does it tag the patients from the risk factos already assigned in episodes!
        # So where do the episodes risk factore get tagged?

        #assert len([x for x in vals(c_eps['icpc_episode']) if type(x) == str and x.startswith("K78")]) >= len(p_ids)

        if nuniq(c_eps.ptnt_prc_epi_id) != len(e_ids):
            raise Exception("nuniq(c_eps.ptnt_prc_epi_id) != len(e_ids):")
        episodes.drop(c_eps.index, inplace=True)
        #e_props = c_eps[EPISODE_COLS].to_dict('records')
        # del c_eps
        # get journals from current e_ids
        if nrow(journals) > 0:
            if SUBSAMPLE_DATA:
                logger("DEBUG SUBAMPLED: setting ptnt_prc_epi_id of journals to match current patients")
                journals['ptnt_prc_epi_id'] = random.choices(e_ids, k = nrow(journals))
            c_js = journals[journals['ptnt_prc_epi_id'].isin(e_ids)] #   journals[journals['ptnt_prc_epi_id'] == '86938|12|7812'] , Ok!
            #j_e_ids = uniq(vals(c_js['ptnt_prc_epi_id']))
            journals.drop(c_js.index, inplace=True)
            # go over each pat_id from current batch of episodes...
            for c_i in range(len(p_ids)):
                p_id = p_ids[c_i]
                c_c_eps = c_eps[c_eps['ptnt_prc_id'].isin([p_id])] # get episodes for p_id
                c_e_ids = uniq(vals(c_c_eps['ptnt_prc_epi_id'])) # get ep_ids for p_id  # "8976|2|1050498" in c_e_ids
                c_c_eps_props = c_c_eps[EPISODE_COLS + ['ptnt_prc_epi_id'] ].to_dict('records') # make  list of episode dicts for p_id
                
                for c_j in range(len(c_e_ids)): # go over each ep_id of current p_id
                    e_id = c_e_ids[c_j] 
                    c_c_js = c_js[c_js['ptnt_prc_epi_id'] == e_id] # get journals for current ep_id 
                    j_props = c_c_js[JOURNAL_COLS].to_dict('records') # make them into list of dicts 
                    j_props = sorted(j_props, key= lambda item: item['journal_datetime'], reverse=F)
                    c_e_idx = [i for i,v in enumerate(c_c_eps_props) if v['ptnt_prc_epi_id'] == e_id][0]
                    c_c_eps_props[c_e_idx]['JOURNALS'] = j_props # add them to current ep_id entry 
                for i, _ in enumerate(c_c_eps_props):
                    if 'ptnt_prc_epi_id' in c_c_eps_props[i]:
                        del c_c_eps_props[i]['ptnt_prc_epi_id']
                    else:
                        logger(f'WARNING: ptnt_prc_epi_id NOT in c_c_eps_props[i] = {c_c_eps_props[i]}')

                # add episode + journal entires to episodes dict  (with p_id as key)
                # 26.Mar.2025 - terrible bug found
                # you can end up overriding episodes_dict for a patient if said patient has episodes in multiple batches! (very likely...)
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

    if TAGGING_MODE:
        tag_outcome(promise_read_patients(pqt_dir/in_file_patients), db_suffix)

    else:
        pats_dict = cached_call(
        merge_episodes_journals_on_patient_level, 
            promise_patients = promise_read_patients(pqt_dir/in_file_patients), 
            promise_episodes = promise_read_episodes(pqt_dir/in_file_episodes),
            promise_journals = promise_read_journals(pqt_dir/in_file_journals),
            tag = db_suffix,
        override_cache=T, cache_fn = cached_name(f'merge_eps_js_ps_{db_suffix}')
        )

        save_pickle(out_file, pats_dict)

_ = logger('pats dict write to file....DONE ', t0)
print("DONE")
