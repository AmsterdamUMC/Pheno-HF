# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES 
# Takes the list of patient-centered dicts, filters out patients younger than min age at followup start,
# converts remaining patient dicts into a flattened dataframe representing the feature space (fs).
# Filters out ICPC codes that occurr less than OHE_MIN_OCCURRENCES times 
# writes dataframe into multiple batches of tsv files
# creates aggregates for time-dependant variables on jounral/episode level (quantization based on TIME_BINS_DAYS)
# removes variables from last time-bin interval (one used for prediction) 
# Note: time-quantization of text embeddings from Top2Vec already handled in 2.1-pre-process-text.py
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
pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)
delete_pickle = lambda f: try_delete_pickle(f, subsampled=SUBSAMPLE_DATA)
cached_call = lambda fn, override_cache=F, **kwargs : try_cached_call(fn, io_r=read_pickle, io_c=pickle_exists, io_w=save_pickle, override_cache=override_cache, **kwargs)
# Boilerplate end
from time import sleep as wait
import pandas as pd
import os
from functools import reduce
from itertools import chain
import gc
import copy
from dim_reduce_utils import group_atcs

import analyse_results_util as ar_util
from try_utils import __init_pats_dict_flwp_filtered

idx = 0 if SUBSAMPLE_DATA else 1
full_filename_pkl = lambda fname : f"{fname}{subsampled_str}.pkl"
in_files = [full_filename_pkl(ns.infile)] 

out_files = [full_filename_pkl(ns.outfile)]
out_file_code_lookup_dicts = f'cluster_fs1_lookup_dicts{subsampled_str}.pkl'
out_file_n_batches = f'cluster_fs1_n_batches{subsampled_str}.pkl'

period_start_end_times = TIME_BINS_DAYS
period_names = get_time_quantized_period_names(TIME_BINS_DAYS)

OHE_MIN_OCCURRENCES = [5, 100][idx] # MINIMUM proportion of times a ICPC/ATC/NHG code should occur before being considered


BATCH_SIZE_FS = [100, 1000][idx] # FEATURE SPACE building batch size

cat_vars = [
    ('patient', 'sex'),
    ('patient', 'deceased'),
    ('episode', 'icpc_episode'),
    ('episode', 'episode_status'),
    ('journal', 'contact_type'),
    # ('journal', 'icpc_journal'),
    ('journal', 'icpc_s'),
    ('journal', 'icpc_o'),
    ('journal', 'icpc_e'),
    ('journal', 'icpc_p'),
    ('journal', 'icpc_x')
    ]

transform_mapper = "Uninitialized"
modalities = []


def _count_c2uces(ep_end_date): 
    return np.isnan(ep_end_date) # if end date is nan, then its an unclosed episode

def _agg_c2uces(vs):
    return np.nanmean(vs)

def _agg_atc_code(ts):
    ts = [t for t in ts if not np.isnan(t)]
    return sum(ts)
# m_ mean
# mx_ max
# n_ number of
# default transform = identity, defualt agg = sum
derived_attrs = {
    # Episode level
    'n_eps': {}, # episodes                         
    'p_c2uces': {'transform': _count_c2uces, 'agg': _agg_c2uces}, # closed to non-closed episodes 
    'tw_icpc_ep': {}, # each icpc will be its own variable # decide how to weigh/count them (maybe incorporate duration?)
    # Journal level
    'n_js':  {}, # journals 
    'tw_icpc_s':  {}, # ..
    'tw_icpc_o':  {}, # ..
    'tw_icpc_e':  {}, # ..
    'tw_icpc_p':  {}, # ..
    'tw_icpc_x':  {}, # ..
    'tw_icpc_j':  {}, # ..
    'tw_atc_code' : {'agg': _agg_atc_code},
    'tw_nhgnummer': {}
    } 
if ns.include_measurements:
    modalities += ["Measurements"]
    cat_vars += [('measurement', 'nhgnummer')]
if ns.include_medications:
    modalities += ["Medications"]
    cat_vars += [('medication', 'atc_code')]

spec_attrs = []
# used if you do not wish to output all the features but some subset, see keys of derived_attrs
if ns.derive_specific_attrs != '':
    spec_attrs = ns.derive_specific_attrs.split(';')
    derived_attrs = { k : derived_attrs[k] for k in spec_attrs  if k in derived_attrs } 

def _tte_notnull(x):
    return [int(not pd.isnull(xi)) for xi in x] 

def _kv_transform(t_fn, k, v, col_nam="var"):
    return pd.DataFrame(list(zip(*[k,t_fn(v)])), columns=['id', col_nam])


def _calc_ncats_per_var(cat_vars, x): 
    t0 = logger("_calc_ncats_per_var ...")
    n_cats_per_var = {}
    lookup_cats_per_var = {}
    rev_lookup_cats_per_var = {}
    for var_lvl, var_nm in cat_vars:

        if not ns.include_measurements and var_lvl == 'measurement':
            continue
        if not ns.include_medications and var_lvl == 'medication':
            continue
        all_vals = None
        if var_lvl == 'patient':
            all_vals = [i[var_nm] for i in x.values()]
        elif var_lvl == 'episode':
            all_vals = [i[var_nm] for e in x.values() for i in e['Episodes']]
        elif var_lvl == 'medication':
            all_vals = [i[var_nm] for e in x.values() for i in e['Medications']]
        elif var_lvl == 'measurement':
            all_vals = [i[var_nm] for e in x.values() for i in e['Measurements']]
        elif var_lvl == 'journal':
            all_vals = [j[var_nm] for j in list( chain(*[i['JOURNALS'] for e in x.values() for i in e['Episodes']]))]
        vcs = pd.Series(all_vals).value_counts()
        vals_with_sufficient_counts = list(vcs[vcs>OHE_MIN_OCCURRENCES].keys())
        cats_to_remove = try_sd(vcs.index.to_list(), vals_with_sufficient_counts)
        logger(f"Removing categories {cats_to_remove} from {var_lvl}-level var ({var_nm}) due to having less than {OHE_MIN_OCCURRENCES} occurrences.")
        include_nan = len(vcs) <= 2 # larger categories like icpc or atc codes can be missing very often, and the missingness is not really useful to include..
        uvals = uniq(vals_with_sufficient_counts, include_nan=include_nan ) #[x for x in c_uniq(all_vals) if x in vals_with_sufficient_counts]
        n_cats_per_var[var_nm] = len(uvals)
        lookup_cats_per_var[var_nm] = { k:v for k,v in zip(uvals, range(len(uvals)))}
        rev_lookup_cats_per_var[var_nm] = { v:k for k,v in zip(uvals, range(len(uvals)))}
        logger(f"mapping for {var_nm}:=\n\t\t{lookup_cats_per_var[var_nm]}")
    logger("_calc_ncats_per_var done", t0)
    return (n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var)

def _extract_time_attrs_df(k, v, level, attrs):
    id_cntr = 1
    id_cols = {'E': ['ep_id'], 'J': ['ep_id', 'j_id'], 'MED': ['med_id'], 'MEAS': ['meas_id'] }[level]
    accessor = {'E': 'Episodes', 'J': 'JOURNALS', 'MED': 'Medications', 'MEAS': 'Measurements'}[level]
    c_p_rows = []
    p_cols = ['p_id'] + id_cols + attrs
    for c_pk, c_pv in zip(k,v): 
        for c_p in c_pv[accessor]: 
            if level == 'J':
                c_p_row =  [*c_pk, id_cntr] + [c_p[k] for k in attrs]
            else:
                c_p_row =  [c_pk, id_cntr] + [c_p[k] for k in attrs]
            c_p_rows.append(c_p_row)
            id_cntr += 1
    p_df = pd.DataFrame(c_p_rows, columns=p_cols) 
    if level == 'J': # sort journal entries by time  (already sorted for episodes)
        p_df = p_df.sort_values('journal_datetime')
    return p_df

def _nan_safe_mean(x):
    x = [v for v in x if v is not None]
    return np.nanmean(x) if len(x) > 0  else np.nan

def _offset_pat_dt0(t_df, df_k, pat_df, dt_cols, offset):
    """
    Per patient - offset the time of each event such that the last one begins at t = offset
    """
    cols = ['p_id', df_k] + dt_cols
    tmp = t_df[cols].merge(pat_df[['id', 'follow_up_LAST']], left_on ='p_id', right_on='id')
    for dt_col in dt_cols:
        tmp[dt_col] = tmp[dt_col] - tmp['follow_up_LAST'] + offset # assume only 1 time bin
        tmp[dt_col] = tmp[dt_col].apply(lambda x : x if x <= offset else np.nan ) # removes episode end dates after flwp time

    t_df = t_df.drop(dt_cols, axis=1)
    cols =  [df_k] + dt_cols
    t_df = t_df.merge(tmp[cols], on=df_k)
    return t_df

def _handle_merge_outer_left(left, right, id_left='id', id_right='id', keep_id="left", allow_left_missing=F, allow_right_missing=T):
    left_ids_missing = try_sd(right[id_right], left[id_left])
    assert left_ids_missing == [] or allow_left_missing
    if left_ids_missing != []:
        right = right[(~right[id_right].isin(left_ids_missing))]

    ids_missing_data = try_sd(left[id_left], right[id_right])
    if ids_missing_data != []:
        if not allow_right_missing:
            logger(f"Removing {len(ids_missing_data)} patients due to no data during time period")
        left = left[(~left[id_left].isin(ids_missing_data))]
        if not allow_right_missing:
            logger(f"Left with {nrow(left)} patients ")
    assert all([v1 == v2 for v1,v2 in zip(right[id_right], left[id_left]) ]) 
    right = right.reset_index()
    left = left.reset_index() # left[left[id_left] == '9942|2'][[i for i in cns(left) if 'HF' in i] + [id_left]]
    if keep_id=="right":
        left = left.drop(id_left, axis=1)
    else:
        right = right.drop(id_right, axis=1)
    left = pd.concat([left, right], axis=1, sort=F)
    if allow_right_missing:
        missing_right = left[left['id'].isin([])].copy()
        missing_right["id"] = ids_missing_data
        left = pd.concat([left, missing_right], axis=0, sort=F)
    if allow_left_missing:
        return left,right
    return left

def _make_feature_space(
    x, 
    transform_mapper, 
    n_cats_per_var, 
    lookup_cats_per_var, 
    rev_lookup_cats_per_var, 
    batch_id=0, 
    append_text_vars=F,
    verbose=F
    ):
    """
    Build patient-level feature space (fs) to be used for downstream experiments
    """
    global derived_attrs
    res = []
    # some of these attrs  will be per unit time period  (e.g., from 0 to 3months, from 3 to 6 months, etc.)
    # ideally smaller periods in more recent time, larger units in earlier time
    # maybe something like 4 x 6months,  2 x 1 year, 2 x 2 year, 1 x 5 year, n x 10 year
    # starting from most recent time point, going all the way backwards to the first 
    # Currently  just using one single time bin (2 years)
    time_bin_intervals = TIME_BINS_DAYS
    time_bin_col_prefixes = get_time_quantized_period_names(TIME_BINS_DAYS)
    if not verbose:
        logger = lambda *kwargs:  -1
    log_once = lambda s: logger(s) if batch_id == 0 else None

    text_cols = [i for i in list(list(x.values())[0].keys()) if try_regex(TOPIC_DISTANCE_COLUMN_REGEX, i)]
    # col format [interval_string col_name]
    # i-space = input space
    # f-space = feature space 
    p_level_attrs = [i for i,m in transform_mapper.items() if m is None or callable(m)] # i-space == f-space
    e_level_attrs = [i for i,m in transform_mapper['Episodes'].items() if m is None or callable(m)] # in i-space
    j_level_attrs = [i for i,m in transform_mapper['Episodes']['JOURNALS'].items()] # in i-space
    med_level_attrs = [i for i,m in transform_mapper['Medications'].items() if m is None or callable(m)] if ns.include_medications else [] # in i-space
    meas_level_attrs = [i for i,m in transform_mapper['Measurements'].items() if m is None or callable(m)] if ns.include_measurements else []# in i-space
    fspace_varname = [f"{int_str}_{c_name}" for int_str in time_bin_col_prefixes for c_name in names(derived_attrs) ] # in f-space
    # create tall-narrow dfs (id, i-spacefeat+) for each feature, finally column-bind them all into merged_df
    # * Patient-level
    for p_attr in p_level_attrs:
        mapping = transform_mapper[p_attr]
        c_s = mapping(list(x.keys()), [v[p_attr] for v in x.values()])
        res.append(c_s)

    merged_df = reduce(lambda left, right: pd.merge(left,right, on ='id', how ='inner'), res) # use this at the end to make into a  single df 
    # log_once(f"patient-level features extracted: {try_print_list(cns(merged_df))}")
    # extract a pandas series for each time_sensitive attribute
    #  columns = [id, ts, val+]
    # * Episode-level
    log_once('start extract a pandas series for each time_sensitive attribute (Episode-level)')
    pe_df = _extract_time_attrs_df(list(x.keys()), list(x.values()), 'E', e_level_attrs)
    # make all dates start from 0
    if e_level_attrs != []:
        pe_df = _offset_pat_dt0(pe_df, 'ep_id', merged_df, ['episode_start_date', 'episode_end_date'], time_bin_intervals[0][1])
    complex_val_accessors_ep = {
        'icpc_episode' : ['episode_start_date']
    }

    complex_val_accessors_j = {
        # 'icpc_journal' : ['journal_datetime'],
        'icpc_s' : ['journal_datetime'],
        'icpc_o' : ['journal_datetime'],
        'icpc_e' : ['journal_datetime'],
        'icpc_p' : ['journal_datetime'],
        'icpc_x' : ['journal_datetime']
    }

    complex_val_accessors_med = {
        'atc_code' : ['medication_datetime']
    }
    get_comp_ids = lambda id_cols, df: [ "_".join([str(v) for v in cmpst.values.tolist()]) for i,cmpst in df[id_cols].iterrows()]
    dd_pe_df = pe_df.drop_duplicates(subset=try_sd(cns(pe_df), ['ep_id']), keep='last', ignore_index=T)
    ep_comp_ids = get_comp_ids(['p_id' ,'ep_id'], pe_df)
    dd_ep_comp_ids = get_comp_ids(['p_id' ,'ep_id'], dd_pe_df)
    dup_ids_ep = try_sd(ep_comp_ids, dd_ep_comp_ids) # eps which were not present after dedup

    for e_attr in e_level_attrs:
        mapping = transform_mapper['Episodes'][e_attr] # performs the ohe
        c_attrs = [e_attr] + complex_val_accessors_ep[e_attr] if e_attr in complex_val_accessors_ep else [e_attr]
        c_s = mapping(vals(dd_pe_df['ep_id']), vals(dd_pe_df[c_attrs]))
        if len(c_s.columns.intersection(dd_pe_df.columns)) > 0: # skip the indentity mapping for now..
            continue
        c_s = c_s.drop(columns='id', axis=1)
        dd_pe_df = dd_pe_df.drop(columns=e_attr, axis=1)
        dd_pe_df = pd.concat([dd_pe_df, c_s], axis=1, sort=F) #pe_df.merge(c_s, left_on='ep_id', right_on='id', how='inner')        

    # log_once(f"episode-level features extracted: {try_print_list(cns(dd_pe_df))}")
    log_once('end extract a pandas series for each time_sensitive attribute (Episode-level)')
    # * Journal-level
    grouped_pe_df = pe_df.groupby('p_id')
    pe_df = dd_pe_df
    del dd_pe_df
    pj_dfs = []
    if j_level_attrs != []:
        log_once('start extract a pandas series for each time_sensitive attribute (Journal-level)')
        # all_pid_groups = [xk for xk,_ in grouped_pe_df]

        for xk,xv in x.items(): # xk,xv = [ (k,v) for k,v in x.items() if v['Episodes'][0]['JOURNALS'] != [] ][0]
            if xk not in grouped_pe_df.groups:
                continue
            ep_ids = vals(grouped_pe_df.get_group(xk)['ep_id'])
            # pe_df = _extract_time_attrs_df(list(x.keys()), list(x.values()), 'E', e_level_attrs)
            tmp = _extract_time_attrs_df(zip([xk]*len(xv['Episodes']), ep_ids), xv['Episodes'], 'J', j_level_attrs)
            tmp = tmp.drop_duplicates(subset=try_sd(cns(tmp), ['j_id', 'ep_id']), keep='last', ignore_index=T)    
            pj_dfs.append(tmp)
        del grouped_pe_df
        pj_df = pd.concat(pj_dfs, axis=0, sort=F, ignore_index=T)
        del pj_dfs
        pj_df = pj_df.drop_duplicates(subset=try_sd(cns(pj_df), ['j_id', 'ep_id']), keep='last', ignore_index=T)    
        log_once(f'correct date calling...')
        pj_df = _offset_pat_dt0(pj_df, 'j_id', merged_df, ['journal_datetime'], time_bin_intervals[0][1])
    for j_attr in j_level_attrs:
        # log_once(j_attr)
        mapping = transform_mapper['Episodes']['JOURNALS'][j_attr] # performs the ohe
        c_attrs = [j_attr] + complex_val_accessors_j[j_attr] if j_attr in complex_val_accessors_j else [j_attr]
        c_s = mapping(vals(pj_df['j_id']), vals(pj_df[c_attrs]))
        if len(c_s.columns.intersection(pj_df.columns)) > 0: # skip the indentity mapping for now..
            continue
        c_s = c_s.drop(columns='id', axis=1)
        pj_df = pj_df.drop(columns=j_attr, axis=1)
        pj_df = pd.concat([pj_df, c_s], axis=1, sort=F) #pj_df.merge(c_s, left_on='j_id', right_on='id', how='inner')
    
        # log_once(f"journal-level features extracted: {try_print_list(cns(pj_df))}")
        log_once('end extract a pandas series for each time_sensitive attribute (Journal-level)')

    # * Medication level
    pmed_df = None
    if med_level_attrs != []:
        log_once('start extract a pandas series for each time_sensitive attribute (Medication-level)')
        if 'tw_atc_code' in derived_attrs:
            pmed_df = _extract_time_attrs_df(list(x.keys()), list(x.values()), 'MED', med_level_attrs)

            pmed_df = _offset_pat_dt0(pmed_df, 'med_id', merged_df, ['medication_datetime'], time_bin_intervals[0][1])
            pmed_df = pmed_df[~pd.isnull(pmed_df['medication_datetime'])] #null time means it happened outside of cohort interval
            pmed_df = pmed_df[pmed_df['medication_datetime'] >= 0] # same for negative time


            for med_attr in med_level_attrs:
                c_attrs = [med_attr] + complex_val_accessors_med[med_attr] if med_attr in complex_val_accessors_med else [med_attr]
                mapping = transform_mapper['Medications'][med_attr] # performs the ohe
                c_s = mapping(vals(pmed_df['med_id']), vals(pmed_df[c_attrs]))
                if len(c_s.columns.intersection(pmed_df.columns)) > 0: # skip the indentity mapping for now..
                    continue
                c_s = c_s.drop(columns='id', axis=1)
                pmed_df = pmed_df.drop(columns=med_attr, axis=1)
                pmed_df = pd.concat([pmed_df, c_s], axis=1, sort=F) 
            log_once('end extract a pandas series for each time_sensitive attribute (Medications-level)')

            # log_once(f"medications-level features extracted: {try_print_list(cns(pmed_df))}")
    
    pmeas_df = None
    if ns.include_measurements:
        log_once('start extract a pandas series for each time_sensitive attribute (Measurements-level)')
        pmeas_df = _extract_time_attrs_df(list(x.keys()), list(x.values()), 'MEAS', meas_level_attrs) 
        for meas_attr in meas_level_attrs:
            mapping = transform_mapper['Measurements'][meas_attr] # performs the ohe
            c_s = mapping(vals(pmeas_df['meas_id']), vals(pmeas_df[meas_attr]))
            if len(c_s.columns.intersection(pmeas_df.columns)) > 0: # skip the indentity mapping for now..
                continue
            c_s = c_s.drop(columns='id', axis=1)
            pmeas_df = pmeas_df.drop(columns=meas_attr, axis=1)
            pmeas_df = pd.concat([pmeas_df, c_s], axis=1, sort=F) 
        log_once('end extract a pandas series for each time_sensitive attribute (Measurements-level)')
    
    # construct the time-quantiezed df 
    t1 = logger('Start construct the time-quantiezed df...')
    fs_cols = {'e': {}, 'j': {}, 'epj': {}, 'med' : {}, 'meas' : {}}
    ep_level_features = ['eps', 'epsacc', 'ep_sts', 'p_c2uces', 'icpc_ep']
    j_level_features = ['js', 'jsacc', 'ctyp', 'icpc_j', 'icpc_s', 'icpc_o', 'icpc_e', 'icpc_p', 'icpc_x']

    measurement_level_features = ['nhgnummer']
    medication_level_features = ['atc_code']
    lookup_shorthand_colnames = {
        'ep_sts' : 'episode_status',
        'ctyp' : 'contact_type',
        'icpc_ep': 'icpc_episode',
        # 'icpc_j' : 'icpc_journal'
    }
    i = 0
    all_col_names = []
    pid_vals = { 'e' : {}, 'j' : {}, 'epj' : {}, 'med' : {}, 'meas' : {}}
    partial_fs_cols = { 'e' : {}, 'j' : {}, 'epj' : {}, 'med': {}, 'meas' : {}}
    # HEART OF IT ALL INSIDE HERE
    for i_time_bin, t_begin_end in enumerate(time_bin_intervals): # t_begin, t_end = time_bin_intervals[0]
        t_begin, t_end = t_begin_end
        for c_name in names(derived_attrs): # c_name = list(names(derived_attrs))[0]
            c_transform = derived_attrs[c_name]['transform'] if 'transform' in derived_attrs[c_name] else identity
            agg_fn = derived_attrs[c_name]['agg'] if 'agg' in derived_attrs[c_name] else None
            je_attr = fspace_varname[i]
            i+=1
            is_mean = c_name[0:2] == 'm_'
            is_count = c_name[0:2] == 'n_'
            is_max = c_name[0:3] == 'mx_'
            is_tw = c_name[0:3] == 'tw_'
            is_anything = any([is_mean, is_count, is_max, is_tw])
            if agg_fn is None:
                if is_mean:
                    agg_fn = _nan_safe_mean
                elif is_count:
                    agg_fn = len
                elif is_max:
                    agg_fn = lambda x: np.nanmax(x) if len(x) > 0 else np.nan
                else:
                    agg_fn = sum
            
            col_nm = c_name[2:] if is_anything else c_name # todo: make this nicer
            col_nm = col_nm[1:] if is_max or is_tw else col_nm
            is_ep_level = col_nm in ep_level_features
            is_j_level = col_nm in j_level_features
            is_medication_level = col_nm in medication_level_features
            is_measurement_level = col_nm in measurement_level_features
            c_groups = None
            is_last_time_bin = i_time_bin == len(time_bin_intervals) - 1
            if is_last_time_bin:
                # logger(f"Skipping column {col_nm} for last time bin...")
                continue
            #log_once(f"{c_name}. {t_begin}:{t_end}", t0)
            if is_ep_level and e_level_attrs != []:
                c_groups = pe_df[(pe_df['episode_start_date']>=t_begin) & (pe_df['episode_start_date']<=t_end)]
                c_groups = c_groups.groupby('p_id')
            if is_j_level and j_level_attrs != []:
                c_groups = pj_df[(pj_df['journal_datetime']>=t_begin) & (pj_df['journal_datetime']<=t_end)]
                c_groups = c_groups.groupby('p_id')
            if is_medication_level and med_level_attrs != []:
                c_groups = pmed_df[(pmed_df['medication_datetime']>=t_begin) & (pmed_df['medication_datetime']<=t_end)]
                c_groups = c_groups.groupby('p_id')
            if is_measurement_level and ns.include_measurements:
                c_groups = pmeas_df[(pmeas_df['measurement_datetime']>=t_begin) & (pmeas_df['measurement_datetime']<=t_end)]
                c_groups = c_groups.groupby('p_id')

            if (is_medication_level and med_level_attrs == []) or (is_measurement_level and not ns.include_measurements):
                continue
            attr_key = 'e' if is_ep_level else 'j' if is_j_level else  'med' if is_medication_level else 'meas' if is_measurement_level else 'epj'
            if c_groups is None:
                continue
            c_pid_vals = c_groups['p_id'].value_counts().index.to_list()
            if t_begin not in pid_vals[attr_key]:
                pid_vals[attr_key][t_begin] = c_pid_vals
            c_col_names = [col_nm] # help to distinguish multi-column from single column variables
            if col_nm in lookup_shorthand_colnames:
                long_nm = lookup_shorthand_colnames[col_nm]
                c_col_names = [f'{long_nm}_{c}' for c in names(rev_lookup_cats_per_var[long_nm])]

            if col_nm in rev_lookup_cats_per_var:
                c_col_names = [f'{col_nm}_{c}' for c in names(rev_lookup_cats_per_var[col_nm])]

            available_cols = c_groups.obj.columns.to_list()
            shorthand_cols = ['eps', 'epsacc', 'p_c2uces', 'js', 'jsacc']
            c_col_names = [c for c in c_col_names if c in available_cols or c in shorthand_cols]
            if len(c_col_names) == 0:
                log_once(f"!>>>>skipping {col_nm}, no values found...")
                continue

            inner_fs_cols = [[] for _ in range(len(c_col_names))]
            clms_now = [ f"{je_attr}_{ix if ix != 0 else ''}" for ix in range(len(c_col_names)) ]
            all_col_names += clms_now # deprecated, replaced by partial_fs_cols
            if t_begin not in partial_fs_cols[attr_key]:
                partial_fs_cols[attr_key][t_begin] = []
            partial_fs_cols[attr_key][t_begin] += clms_now
            if col_nm.startswith('icpc_'):
                agg_fn = 'mean' if is_mean or is_tw else 'max' if is_max else None # only support is_mean/max now
                c_res = c_groups.agg({ ccn : agg_fn for ccn in c_col_names}).reset_index()               
                inner_fs_cols = c_res.T.values.tolist()[1:]
            else:
                for c_group, c_rows in c_groups: #c_group, c_rows = (list(c_groups.groups.keys())[0], c_groups.get_group(list(c_groups.groups.keys())[0]))
                    c_vls = []
                    if col_nm in ['eps', 'epsacc'] :
                        c_vls.append([st for st in vals(c_rows['episode_start_date'])])
                    elif col_nm == 'p_c2uces':
                        c_vls.append([ed for ed in vals(c_rows['episode_end_date'])])
                    elif col_nm in ['js', 'jsacc']:
                        c_vls.append([jd for jd in vals(c_rows['journal_datetime'])])
                    else:
                        if not any([x in c_col_names for x in cns(c_rows)]):
                            continue
                        c_vls = c_rows[c_col_names].T.values.tolist()
                    c_idx = 0
                    for c_col_name in c_col_names: # HEART OF IT ALL
                        vals_rdy = [c_transform(c) for c in c_vls[c_idx]]
                        agg_val = agg_fn(vals_rdy)
                        inner_fs_cols[c_idx].append(agg_val) # agg_val = single f-space column for a single patient
                        c_idx += 1
            
            if t_begin not in fs_cols[attr_key]:
                fs_cols[attr_key][t_begin] = []
            for fs_col in inner_fs_cols:
                fs_cols[attr_key][t_begin].append(fs_col)
            
               
    logger('End construct the time-quantiezed df...', t1)
    t_quant_df = None 
    for attr_key in pid_vals.keys():
        for t_begin in pid_vals[attr_key].keys():
            c_pid_vals = pid_vals[attr_key][t_begin]
            if len(fs_cols[attr_key]) == 0:
                continue
            c_fs_cols = fs_cols[attr_key][t_begin]
            c_col_names = partial_fs_cols[attr_key][t_begin]
            assert len(uniq([len(c_fs_cols[i]) for i in range(len(c_fs_cols))])) == 1
            assert len(c_pid_vals) == len(c_fs_cols[0])
            c_quant_df = pd.DataFrame([c_pid_vals] + c_fs_cols).T
            c_quant_df.columns = ['p_id'] + c_col_names 
            c_quant_df = c_quant_df.round(4)
            c_quant_df = c_quant_df.fillna(0)
            if t_quant_df is None:
                t_quant_df = c_quant_df
            else:
                t_quant_df = pd.merge(t_quant_df, c_quant_df, on = "p_id", how = 'outer')

    merged_df = merged_df.sort_values(by='id')
    if t_quant_df is not None:
        t_quant_df = t_quant_df.round(4)
        t_quant_df = t_quant_df.fillna(0)
        t_quant_df = t_quant_df.sort_values(by='p_id')
        merged_df = _handle_merge_outer_left(merged_df, t_quant_df, id_right='p_id', allow_left_missing=F)

    if append_text_vars:
        text_emb_df = None
        text_emb_records = []
        for k,v in x.items():
            n_r = {"id": k}
            n_r.update({k1:v1 for k1,v1 in v.items() if k1 in text_cols })
            text_emb_records.append(n_r)
        text_emb_df = pd.DataFrame.from_records(text_emb_records)
        text_emb_df = text_emb_df.sort_values(by='id')

        merged_df, text_emb_df = _handle_merge_outer_left(merged_df, text_emb_df, allow_left_missing=T)
        if 'level_0' in merged_df:
            merged_df = merged_df.drop("level_0", axis=1)
        if 'index' in merged_df:
            merged_df = merged_df.drop("index", axis=1)
        merged_df = merged_df.reset_index()
        text_emb_df = text_emb_df.reset_index()
        merged_df = pd.concat([merged_df, text_emb_df], axis=1, sort=F)

    # try_print_col_counts_by_type(merged_df, log=log_once)

    return merged_df 

def _make_ohe_fn_partial_partial(col_nm, lookup_triplet, transform_fn=identity, special_vals_fn=None):
    if special_vals_fn is not None:
        return lambda k,v: try_ohe(k, transform_fn(v), col_nm, lookup_triplet, special_vals_fn=special_vals_fn)
    return lambda k,v: try_ohe(k, transform_fn(v), col_nm, lookup_triplet)

def _time_weigh_single(t, max_t=FOLLOW_UP_PERIOD_DAYS-FOLLOW_UP_HFPOS_CENS_WINDOW, min_w = 0.5, max_w = 1): #
    w_range = max_w-min_w
    res = (t/max_t)*w_range + min_w
    return round(res, 4)

def _time_weigh_icpc(vs): # v[0] = ICPC code, v[1] = time , note time is already ranging from 0 to 730
    # note - since we filter only on journal start time, episodes might have negative time (i.e., started before t0)
    # to avoid issues with negative time, we set those times to 0 here
    res = [(v[0], _time_weigh_single(max(v[1], 0))) for v in vs]
    return res

def _time_weigh_count(t): # t = time
    # note - since we filter only on journal start time, episodes might have negative time (i.e., started before t0)
    # to avoid issues with negative time, we set those times to 0 here
    return _time_weigh_single(max(t, 0))

def _calc_acceleration(ts): # t[0] = time
    # note - since we filter only on journal start time, episodes might have negative time (i.e., started before t0)
    # to avoid issues with negative time, we set those times to 0 here
    if len(ts) < 3:
        return 0 # cant derive acceleration with too few points
    ts = sorted(ts)
    u = 0 # initial velocity
    a = None # initial acceleration
    S = 0 # initial distance traveled
    prev_t = None # last time point
    # x = []
    # y_s = []
    # y_u = []
    y_a = []
    delta_ts = []
    for t,c in pd.Series(ts).value_counts(sort=F).to_dict().items():
        S = S + c
        if prev_t is None: # first time point can only calc velocity
            prev_t = t
            # logger(f"S = {S}; t = {t}; a = NA; u={u:0.4f}; delta_u= NA ")
            continue
        u = S/t # we've traveled i+1 distnace by time t (i.e., we've had i+1 consults by time t)
        delta_S = c # we've traveled this distance since last time
        delta_t = t-prev_t
        delta_ts += [delta_t]
        delta_u = (delta_S/delta_t) - u # what velocity did we have during this last interval?
        a = 2000*(delta_S-delta_u*delta_t)/(delta_t**2)
        prev_t = t
        u = S/t 
        # x+= [t]
        # y_s += [S]
        # y_u += [u]
        y_a += [a]
        # logger(f"S = {S}; t = {t}; a = {a:0.4f}; u={u:0.4f}; delta_u={delta_u:0.4f} ")
    # x = np.array(x)
    # y_u = np.array(y_u)
    # y_a = np.array(y_a)
    # y_s = np.array(y_s)
    # plt.figure(figsize=(10,6))
    # plt.plot(x, y_u*250, color='green', marker='o', label='vel')
    # plt.plot(x, y_a*1000, color='blue', linestyle='--', marker='s', label='acc')
    # plt.scatter(x, y_s, color='red', label='S', zorder=5)
    # plt.legend()
    # plt.savefig('plots/3_pre_process/test.png')
        
    delta_ts = np.array(delta_ts) / sum(delta_ts)
    # delta_ts used to weigh-in the contribution of each acc value based on its time-span covered
    # time_weighing so that acceleration values earlier in time are less important
    res = sum([_time_weigh_single(max(t, 0))*a*delta_t for t,a,delta_t in zip(ts[1:], y_a, delta_ts)])
    return res

def _nan_to_empty_str(x):
    return x if type(x) == str else ""

def _nans_to_empty_str(xs):
    return [(_nan_to_empty_str(x[0]), x[1]) for x in xs]

def _kv_identity(col_nm):
    return lambda x,y: _kv_transform(identity, x, y, col_nam=col_nm)

def _reduce_single_el_list(xs):
    return [x[0] for x in xs]


def _build_transform_mapper(n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var):
    global derived_attrs
    def make_ohe_fn_partial(col_nm, transform_fn=identity, special_vals_fn=None):
        lookup_triplet = (n_cats_per_var[col_nm], lookup_cats_per_var[col_nm], rev_lookup_cats_per_var[col_nm])
        return _make_ohe_fn_partial_partial(col_nm, lookup_triplet, transform_fn, special_vals_fn=special_vals_fn)

    def make_ohe_fn_partial_icpc(col_nm):
        return make_ohe_fn_partial(col_nm,
                                    transform_fn = _nans_to_empty_str,
                                    special_vals_fn=_time_weigh_icpc)


    derived_attrs['n_eps'] = {'transform' : _time_weigh_count, 'agg': sum }
    derived_attrs['n_js'] =  {'transform' : _time_weigh_count, 'agg': sum }
    derived_attrs['n_epsacc'] = {'agg' : _calc_acceleration }
    derived_attrs['n_jsacc'] = {'agg' : _calc_acceleration }
    

    make_ohe_fn_partial_atc = make_ohe_fn_partial_icpc
    transform_mapper = {
        'follow_up_LAST' : _kv_identity('follow_up_LAST'),
        'age_days' : _kv_identity('age_days'),
        'sex': make_ohe_fn_partial('sex'),
        'deceased': make_ohe_fn_partial('deceased'),
        't_HF': lambda x,y: _kv_transform(_tte_notnull, x, y, col_nam='y_HF'), # patient had HF if their time to HF is not missing 
        'Episodes': {
            'episode_start_date' : _kv_identity('episode_start_date'),
            'episode_end_date': _kv_identity('episode_end_date'), 
            'icpc_episode' : make_ohe_fn_partial_icpc('icpc_episode'),
            'episode_status': make_ohe_fn_partial('episode_status', transform_fn = _reduce_single_el_list), 
            'JOURNALS': {
                'journal_datetime': _kv_identity('journal_datetime'), 
                'contact_type': make_ohe_fn_partial('contact_type', transform_fn = _reduce_single_el_list), 
                # 'icpc_journal': make_ohe_fn_partial_icpc('icpc_journal'),
                'icpc_s': make_ohe_fn_partial_icpc('icpc_s'), 
                'icpc_o': make_ohe_fn_partial_icpc('icpc_o'),
                'icpc_e': make_ohe_fn_partial_icpc('icpc_e'),
                'icpc_p': make_ohe_fn_partial_icpc('icpc_p'),
                'icpc_x': make_ohe_fn_partial_icpc('icpc_x'), 
                }
        } 
    }
    if ns.include_measurements:
        transform_mapper['Measurements'] = {
            'measurement_datetime': lambda x,y: _kv_transform(identity, x, y, col_nam='measurement_datetime'),
            'nhgnummer' : make_ohe_fn_partial('nhgnummer'),
        }
    if ns.include_medications:
        transform_mapper['Medications'] = {
            'medication_datetime' : _kv_identity('medication_datetime'),
            'atc_code' : make_ohe_fn_partial_atc('atc_code')  # trim ATC (and group ICPC) codes happnes in script 5_dim_reduce.py
        }
    return transform_mapper

def _init_make_feature_space_batched(x_file, batch_size, reuse_batchfile_if_exist=F):
    x =  read_pickle(x_file)["x"]
    logger(f"Read {len(x)} records")
    n_eps = [len([len(j['JOURNALS']) for j in x[p1]['Episodes'] if j['JOURNALS'] != []]) for p1 in  list(x.keys()) if [ 1 for j in x[p1]['Episodes'] if j['JOURNALS'] != [] ] != [] ]
    n_js = [sum([len(j['JOURNALS']) for j in x[p1]['Episodes']  if j['JOURNALS'] != []]) for p1 in list(x.keys()) if [ 1 for j in x[p1]['Episodes'] if j['JOURNALS'] != [] ] != [] ]
    sum(n_js)
    sum(n_eps)
    n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var = _calc_ncats_per_var(cat_vars, x)
    code_lookup_dicts = { 'lookup': lookup_cats_per_var, 'rev-lookup': rev_lookup_cats_per_var}
    save_pickle(out_file_code_lookup_dicts, code_lookup_dicts)
    transform_mapper = _build_transform_mapper(n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var)
    logger("_init_make_feature_space_batched calculate number of batches needed...")
    n = len(x)
    all_keys = names(x)
    # store ordering of patient ids to be used in next script
    try_save_pickle(f'all_keys{subsampled_str}.pkl', all_keys)
    shuffle_keys_idx = list(range(0, len(all_keys)))
    random.shuffle(shuffle_keys_idx) # patients earlier have much less data, leads in uneven process usage, shuffle for better utilization of CPU time
    all_keys_shuffled = [all_keys[i] for i in shuffle_keys_idx]
    n_batches = round_up(n / batch_size)
    save_pickle(out_file_n_batches, {"n_batches": n_batches})
    logger(f"Need {n_batches} batches...")
    batch_filename_prefix = x_file.split('.')[0]
    batched_filenames = []
    t1 = logger("_init_make_feature_space_batched split pickle file into separate files...")
    for c_batch in range(n_batches):
        c_fn = f"{batch_filename_prefix}_{batch_size}_b{c_batch}.pkl"
        batched_filenames.append(c_fn)
        if reuse_batchfile_if_exist and pickle_exists(c_fn):
            logger(f"batch = {c_batch} already found in {c_fn}. Going to reuse existing file!")
            continue

        t0 = logger(f"batch = {c_batch}")
        start = c_batch*batch_size
        end = min(start + batch_size, n)
        x_batch = {k: x[k] for k in all_keys_shuffled[start:end]}
        wait(random.randint(1, 10)) # hopefully saves deadlocking
        save_pickle(c_fn, x_batch)
        _ = logger(f"batch = {c_batch} done", t0)
    logger("_init_make_feature_space_batched split pickle file into separate files...done", t1)
    return  transform_mapper, n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, batched_filenames

def _make_feature_space_batched(x_file, out_file, batch_size=BATCH_SIZE_FS, reuse_batchfile_if_exist=F):
    global transform_mapper
    transform_mapper, n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, batched_filenames = _init_make_feature_space_batched(x_file,
                                                                                                             batch_size, 
                                                                                                             reuse_batchfile_if_exist)
    n_batches = len(batched_filenames)
    batch_ids = list(range(n_batches))
    t0 = logger(f"starting _make_feature_space_batched n_batches={n_batches}...")
    t1 = t0
    # Note: making this multiprocess could cause OOM...
    n_processes = 1
    batches_per_proc = determine_n_records_per_split(n_batches, n_processes)
    items = []
    start = 0
    for i in range(n_processes):
        c_item = { "batch_ids" : batch_ids[start:(start+batches_per_proc[i])],
                    "batched_filenames": batched_filenames,
                    "n_cats_per_var": n_cats_per_var,
                    "lookup_cats_per_var": lookup_cats_per_var,
                    "rev_lookup_cats_per_var": rev_lookup_cats_per_var,
                    "worker_id": i+1,
                    "out_file": out_file
                }
        items += [c_item]
        start+=batches_per_proc[i]
    try_run_multiprocess(items = items, worker_fn=__make_cluster_fs1_batched_worker_fn, n_processes=n_processes)
    return None

def __batch_inner_batch(item):
    x_batch = item[0]
    n_cats_per_var = item[1]
    lookup_cats_per_var = item[2]
    rev_lookup_cats_per_var = item[3]
    i_batch = item[4]
    transform_mapper = _build_transform_mapper(n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var)

    # 1 item = [x_batch, transform_mapper
    # c_df = _make_feature_space(x_batch, transform_mapper, n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, batch_id = i_batch, append_text_vars=T)
    # chunk up transform mapper, avoid too much memory usage
    tm_copy = copy.deepcopy(transform_mapper)
    tm_eps = { k:v for k,v in tm_copy.items() if k in ['Episodes', 'age_days', 'follow_up_LAST', 'sex', 'deceased', 't_HF']}
    tm_eps['Episodes']['JOURNALS'] = {}
    tm_eps['Medications'] = {}

    tm_copy = copy.deepcopy(transform_mapper)
    tm_js = { k:v for k,v in tm_copy.items() if k in ['Episodes', 'age_days', 'follow_up_LAST', 'sex', 'deceased', 't_HF']}
    tm_js['Episodes'] = {k:v for k,v in tm_copy['Episodes'].items() if k in ['JOURNALS', 'episode_start_date', 'episode_end_date']} 
    tm_js['Medications'] = {}

    tm_copy = copy.deepcopy(transform_mapper)
    tm_meds = { k:v for k,v in tm_copy.items() if k in ['age_days', 'follow_up_LAST', 'sex', 'deceased', 't_HF', 'Medications']}
    tm_meds['Episodes'] = { 'JOURNALS' : {} } 

    tm_copy = copy.deepcopy(transform_mapper)
    tm_txt = { k:v for k,v in tm_copy.items() if k in ['age_days', 'follow_up_LAST', 'sex', 'deceased', 't_HF']}
    tm_txt['Episodes'] = { 'JOURNALS' : {} } 
    tm_txt['Medications'] = {}

    c_df_eps = _make_feature_space(x_batch,
                tm_eps,
                n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, 
                batch_id = i_batch,
                append_text_vars=F
            )
    c_df_js = _make_feature_space(x_batch,
                tm_js,
                n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, 
                batch_id = i_batch,
                append_text_vars=F
            )
    c_df_meds = _make_feature_space(x_batch,
                tm_meds,
                n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, 
                batch_id = i_batch,
                append_text_vars=F
            )
    c_df_txt = _make_feature_space(x_batch,
                tm_txt,
                n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, 
                batch_id = i_batch,
                append_text_vars=T
            )
    dup_cols_to_remove = ['index', 'age_days', 'sex_1', 'sex_2', 'deceased_1', 'deceased_2', 'y_HF', 'follow_up_LAST']
    cols_eps = try_si(cns(c_df_eps), dup_cols_to_remove)
    cols_js = try_si(cns(c_df_js), dup_cols_to_remove)
    cols_meds = try_si(cns(c_df_js), dup_cols_to_remove)
    if cols_eps != []:
        c_df_eps = c_df_eps.drop(cols_eps, axis=1)
    if cols_js != []:
        c_df_js = c_df_js.drop(cols_js, axis=1)
    if cols_meds != [] and cols_meds in cns(c_df_meds):
        c_df_meds = c_df_meds.drop(cols_meds, axis=1)
    c_df_new = pd.merge(c_df_eps, c_df_js, how='left', on='id', suffixes=(None, "_duplicate"))
    c_df_new = c_df_new.drop([c for c in cns(c_df_new) if '_duplicate' in c], axis=1)
    c_df_new = pd.merge(c_df_new, c_df_meds, how='left', on='id', suffixes=(None, "_duplicate"))
    c_df_new = c_df_new.drop([c for c in cns(c_df_new) if '_duplicate' in c], axis=1)
    c_df_new = pd.merge(c_df_new, c_df_txt, how='left', on='id', suffixes=(None, "_duplicate"))
    c_df_new = c_df_new.drop([c for c in cns(c_df_new) if '_duplicate' in c], axis=1)
    return c_df_new

def __make_cluster_fs1_batched_worker_fn(item):
    batch_ids = item["batch_ids"]
    batched_filenames = item["batched_filenames"]
    n_cats_per_var = item["n_cats_per_var"]
    lookup_cats_per_var = item["lookup_cats_per_var"]
    rev_lookup_cats_per_var = item["rev_lookup_cats_per_var"]
    worker_id = item["worker_id"]
    out_file = item["out_file"]
    t0 = logger(f"Worker {worker_id} starting to run for {len(batch_ids)} batches...")
    t1 = t0
    n_batches = len(batch_ids)
    n_inner_bathes = 14
    for i_batch, c_batch in enumerate(batch_ids):
        # break
        c_out_file = f"{out_file.split('.')[0]}b{c_batch}.tsv"
        x_batch = read_pickle(batched_filenames[c_batch])

        pids = list(x_batch.keys())
        inner_batch_size = round_down(len(x_batch) / n_inner_bathes)
        last_batch_size_diff = len(x_batch) % n_inner_bathes
        inner_start = 0
        x_inner_batches = []
        for c_inner_batch in range(n_inner_bathes):
            c_end = inner_start+inner_batch_size
            if c_inner_batch == n_inner_bathes - 1:
                c_end = len(x_batch)
            c_x = {pid:x_batch[pid] for pid in pids[inner_start:c_end]}

            c_item = [c_x, n_cats_per_var, lookup_cats_per_var, rev_lookup_cats_per_var, i_batch]
            inner_start = c_end
            x_inner_batches = x_inner_batches + [c_item]


        n_processes = min(14, n_inner_bathes)
        t1 = logger(f"******start running on = {n_processes} cpus")
        tmp = try_run_multiprocess(worker_fn=__batch_inner_batch, items = x_inner_batches, n_processes=n_processes)
        c_df_new = pd.concat(tmp, axis=0)# ...
        logger(f"******End running on {n_processes} cpus",t1)
        # c_df_new = __batch_inner_batch(x_inner_batches[0])

        try_save_df(c_out_file, c_df_new) # note: c_Df does have text variables in here # [vals(c_df[c_df['id'] == '86938|12' ][c]) for c in cns(c_df) if 'icpc_HF' in c] 
        t0 = logger(f"Worker {worker_id} batch {c_batch} done... ({round((100*(i_batch+1))/n_batches)}%)", t0)
        gc.collect()
    _ = logger(f"worker {worker_id} done _make_feature_space_batched...", t1)
    logger("Cleaning up...")
    wait(random.randint(1, 60 if not SUBSAMPLE_DATA else 2))
    for c_fn in [batched_filenames[i] for i in batch_ids]:
        delete_pickle(c_fn)
        wait(random.randint(1, 6)) # hopefully enough to prevent strange deadlock...
        



gc.collect()
for i, in_file in enumerate(in_files):
    _make_feature_space_batched(in_file, out_files[i], reuse_batchfile_if_exist=ns.reuse_batchfiles)
logger("Done...")
try_log("DONE")

