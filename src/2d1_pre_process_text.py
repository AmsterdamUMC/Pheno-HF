# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES:
# 1. Extracts all relevant text fields per patient (icpc text, episode descriptions) into document collections.
    Defines 3 document collections [DCs]: [WIP]
    DC1 = medications text  [very repetitive, short, a lot of medication names and dosages, most number of]
    DC2 = episodes text     [very short least number of]
    DC3 = journals text     [varying length (can be very long), more free text, high number of] 
# 2. Removes text information from last time bin (one used for prediction)
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
logger = get_default_logger_fn(__file__)
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
start_time = logger("Start running...")
# Boilerplate end

from collections import OrderedDict

full_filename_pkl = lambda fname : f"{fname}{subsampled_str}.pkl"
in_file = full_filename_pkl(ns.infile)
out_file = full_filename_pkl(ns.outfile)

x = read_pickle(in_file)
# we want to:
# 1. extract all the text per patient 
# recall, x looks like x[id][demo_var] OR x[id]['Episodes'][ep_idx][ep_var] OR x[id]['Episodes'][ep_idx]['JOURNALS'][j_idx]
text_cols = []
time_cols = []
txt_vars_j = ['text_s', 'text_o', 'text_e', 'text_p', 'text_x'] #[x for x in JOURNAL_COLS if try_regex('.*text.*', x)]
txt_vars_ep = ['episode_description']
txt_vars_meds = ['medication_txt']
txt_vars_msrs = ['measurement_txt']
txt_vars_all = txt_vars_ep + txt_vars_j + txt_vars_meds + txt_vars_msrs


period_start_end_times = TIME_BINS_DAYS
period_names = get_time_quantized_period_names(TIME_BINS_DAYS)
# Don't use last time period (one used for prediction)
period_start_end_times = period_start_end_times[:-1]
period_names = period_names[:-1]

txt_per_patient = {}
txt_dict = {} # will mimic x
log_every_n = 100 if SUBSAMPLE_DATA else 10000
n_tot = len(x.keys())
modalities = []
if ns.include_measurements:
    logger(f"ERROR: include_measurements not implemented!")
    exit(-1)
    modalities += ["Measurements"]
if ns.include_medications:
    modalities += ["Medications"]


# logging 
logger(f':: Init :: Going to quantize {", ".join(modalities) + " modalities and" if modalities else "" } text for periods:') 


def quantize_txt_patient(c_patient_txt, txt_vars, tag, txt_var_with_parents = []):
    all_vars = txt_vars + txt_var_with_parents
    all_txt_ids = list(set([txt_id for txt_var in all_vars for txt_id in c_patient_txt[txt_var].keys() ]))
    all_txt_ids = sorted(all_txt_ids, key = lambda x: -int(x.split('_')[1])) 
    all_txt_ids_dt_pairs = [ (id, int(id.split('_')[1])) for id in all_txt_ids]
    # text_cols = []
    # time_cols = []
    
    for i, period_nm in enumerate(period_names): # i, period_nm = list(enumerate(period_names))[0]
        bin_start, bin_end = period_start_end_times[i]
        bin_dur = bin_end-bin_start
        c_txt_ids_dts = [(id,round((dt-bin_start+1)/bin_dur,4)) for id,dt in all_txt_ids_dt_pairs if dt >= bin_start and dt < bin_end]
        txt_var_nm = f'{period_nm}_{tag}_text_'
        time_var_nm = f'{period_nm}_{tag}_time_'

        if txt_var_nm not in c_patient_txt:
            c_patient_txt[txt_var_nm] = []
        if time_var_nm not in c_patient_txt:
            c_patient_txt[time_var_nm] = []

        for txt_id, txt_dt in c_txt_ids_dts:
            c_txt_var_vals = [c_patient_txt[txt_var][txt_id] if txt_id in c_patient_txt[txt_var] else "" for txt_var in txt_vars]
            c_txt_doc = " ".join(c_txt_var_vals).strip()
            c_patient_txt[txt_var_nm] += [c_txt_doc]
            c_time_doc = txt_dt if len(c_txt_doc) > 0 else 0
            c_patient_txt[time_var_nm] += [c_time_doc]

        for txt_id, txt_dt in c_txt_ids_dts:
            c_txt_var_vals = []
            par_txt_val = ""
            for txt_var in txt_var_with_parents:
                c_txt_val = ""
                if txt_id in c_patient_txt[txt_var]:
                    c_txt_val = c_patient_txt[txt_var][txt_id]['txt']
                    par_txt_val = c_patient_txt[txt_var][txt_id]['parent_txt']
                c_txt_var_vals += [c_txt_val]

            c_txt_doc = par_txt_val + " ".join(c_txt_var_vals) 
            c_patient_txt[txt_var_nm] += [c_txt_doc]
            c_time_doc = txt_dt if len(c_txt_doc) > 0 else 0
            c_patient_txt[time_var_nm] += [c_time_doc]

    # # remove non-quantized text vars
    for text_var in txt_vars+txt_var_with_parents:
        del c_patient_txt[text_var]
    return c_patient_txt

for i,_ in enumerate(period_names): 
    bin_start, bin_end = period_start_end_times[i]
    period_nm = period_names[i]
    logger(f"\t\tDays [{bin_start}; {bin_end}) ({period_nm} months)")    

logger(f':: Run :: Start text (and optional modalities) processing of {len(x.keys())} patient records') 
if SENS_ANALYSIS_UNBIAS_FLWP_START:
    time_biases = { pid: pval for tag in ['AHA', 'ANH'] for pid,pval in read_pickle(f'pats_dict_{tag}{subsampled_str}_time_biases.pkl').items() } 
    get_flwp_dts = lambda p: [ j['journal_datetime'] for e in p['Episodes'] for j in e['JOURNALS'] ]
    def is_flwp_in_range(p, days_flwp):
        if days_flwp <= 0:
            return F
        lb = p['follow_up_LAST'] - days_flwp
        return [d for d in get_flwp_dts(p) if d < lb] == []

    assert try_sd(x.keys(), time_biases.keys()) == [] 

    flwp_dates = { pid : get_flwp_dts(v) for pid,v in x.items() }
    needs_correction = [k for k,p in x.items() if not is_flwp_in_range(x[k], time_biases[k]['days_flwp'])]
    for p_idx, pid in enumerate(needs_correction):
        p = time_biases[pid]
        lb = p['follow_up_LAST'] - p['days_flwp']
        ub = p['follow_up_LAST']
        for i,e in enumerate(x[pid]['Episodes']):
            new_ep = e
            new_ep['JOURNALS'] = [j for j in e['JOURNALS'] if j['journal_datetime'] < lb ]
            x[pid]['Episodes'][i] = new_ep
        x[pid]['Medications'] = [m for m in x[pid]['Medications'] if m['medication_datetime'] >= lb ]


    logger(f"before time bias filter n pats= {len(x)}")
    x = { k:v for k,v in x.items() if get_flwp_dts(v) != []}
    logger(f"after time bias filter n pats= {len(x)}")

for counter, p_id in enumerate(x.keys()):
    p_flwp_start = x[p_id]['follow_up_LAST'] - FOLLOW_UP_PERIOD_DAYS
    p_flwp_end = x[p_id]['follow_up_LAST'] - FOLLOW_UP_HFPOS_CENS_WINDOW
    p_cens_window = (p_flwp_end , p_flwp_end+FOLLOW_UP_HFPOS_CENS_WINDOW)

    # loggin
    if counter < 10:
        logger(f"{p_id} followup period = {conv_date_days_to_str(p_flwp_start)}:{conv_date_days_to_str(p_flwp_end)}")
        logger(f"{p_id} censorship window = {conv_date_days_to_str(p_cens_window[0])}:{conv_date_days_to_str(p_cens_window[1])}")

    if counter % log_every_n == 0:
        logger(f"{(counter*100)/n_tot:0.1f}% completed quantization...")

    c_patient_txt_meds = {k:{} for k in txt_vars_meds } # DC - decided not to use
    c_patient_txt_msrs = {k:{} for k in txt_vars_msrs } # DC_notimplemented
    c_patient_txt_mods = {'Medications' : c_patient_txt_meds, 'Measurements': c_patient_txt_msrs}
    c_patient_txt_eps = {k:{} for k in txt_vars_ep } # DC2 
    c_patient_txt_js = {k:{} for k in txt_vars_j } # DC3

    c_patient_txt_jeps = {k:{} for k in txt_vars_ep+txt_vars_j } # DC2 + DC3 = new DC1
    #c_patient_txt = {k:{} for k in txt_vars_all } 
    for modality in modalities:
        # logger(f'{modality}-level')
        c_patient_txt_mod = c_patient_txt_mods[modality]
        id_key = {"Medications": 'ptnt_prc_med_id', 'Measurements': 'ptnt_prc_msrm_id'}[modality]
        date_key = {"Medications": 'medication_datetime', 'Measurements': 'measurement_datetime'}[modality]
        mod_cols = {"Medications": MEDICATION_COLS, 'Measurements': MEASUREMENT_COLS}[modality]
        txt_cols = {"Medications": txt_vars_meds, 'Measurements': txt_vars_msrs}[modality]
        for txt_col in txt_cols:
            for mod_idx, mod_item in enumerate(x[p_id][modality]):
                c_time = int(mod_item[date_key] - p_flwp_start) # c_time = n days since patient follow_up start
                c_text = mod_item[txt_col]
                if not type(c_text) == str: # no time data
                    continue
                c_id = f"{mod_idx}_{c_time}"
                c_patient_txt_mod[txt_col][c_id] = c_text # impossible to override here, so no need to check/append
    # construct dict(s) with current patient texts
    for ep_idx, _ in enumerate(x[p_id]['Episodes'] if 'Episodes' in x[p_id] else []):
        c_time = int(x[p_id]['Episodes'][ep_idx]['episode_start_date'] - p_flwp_start)
        c_ep_id = f"{ep_idx}_{c_time}" 
        c_ep_txt = ""
        for text_var in txt_vars_ep:
            c_text = x[p_id]['Episodes'][ep_idx][text_var]
            if not type(c_text) == str:
                continue
            if len(c_text.strip()) == 0:
                continue
            c_patient_txt_jeps[text_var][c_ep_id] = c_text
            c_ep_txt += f"{c_text} ."

        # construct txt from journal level
        for j_idx, _ in enumerate(x[p_id]['Episodes'][ep_idx]['JOURNALS'] if 'JOURNALS' in x[p_id]['Episodes'][ep_idx] else []):
            c_time = int(x[p_id]['Episodes'][ep_idx]['JOURNALS'][j_idx]['journal_datetime'] - p_flwp_start)
            for text_var in txt_vars_j:
                c_text = x[p_id]['Episodes'][ep_idx]['JOURNALS'][j_idx][text_var]
                if not type(c_text) == str:
                    continue
                if len(c_text.strip()) == 0:
                    continue
                c_j_id = f"{j_idx}_{c_time}" # avoid issues if multiple journals on same day
                c_patient_txt_jeps[text_var][c_j_id] = { "txt": c_text, "parent_txt": c_ep_txt }  # could be beter..

    # perform time quantization
    if ns.include_medications_text:
        c_patient_txt_meds = quantize_txt_patient(c_patient_txt_meds, txt_vars_meds, 'med')
        x[p_id].update(c_patient_txt_meds)
    c_patient_txt_jeps = quantize_txt_patient(c_patient_txt_jeps, txt_vars_ep, 'epj', txt_vars_j)
    x[p_id].update(c_patient_txt_jeps)
    if counter == 0:
        if ns.include_medications_text:
            text_cols += [list(c_patient_txt_meds.keys())[0]]
            time_cols += [list(c_patient_txt_meds.keys())[1]]
        text_cols += [list(c_patient_txt_jeps.keys())[0]]
        time_cols += [list(c_patient_txt_jeps.keys())[1]]        
    
    

save_pickle(out_file, {"text_cols": text_cols, "time_cols": time_cols, "x" : x })
logger("DONE", start_time)
