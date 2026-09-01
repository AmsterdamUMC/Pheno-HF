# -*- coding: utf-8 -*-

print(
    '''
    # WHAT THIS SCRIPT DOES:
    # 1.1. Filters patients based on cohort start/end days
    # 1.2. Adds patient-level tagging (from target_hf logic)
    # 1.3. Removes feaures not needed (from all levels)
    # 1.4. Removes patients who had HF diag before cohort start
    # 1.5. Applies delay window censorship on episodes before outcome occurrence
    # NOT handled here: censoring patient-level tags based on cohort start/end
    #   suggestion -> keep until very last end, there set tag = tag if t_tag < (last_followup_time - del_window) else pd.NaT
    # 2.Reads medications modality items data and adds it to patient level dataset [toggle on/off via ns.include_measurements]
    # 3.Reads measurements modality items data and adds it to patient level dataset [toggle on/off via ns.include_medications]
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
check_if_debugging(IS_DEBUG) # attach if debug
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
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
from sys import exit
# Boilerplate end

batch_size = STEP1D1_BATCH_SIZE
db_suffixes = ["AHA", "ANH"]

in_file_patients_d = {
        "AHA" : f"pats_dict_AHA_{subsampled_str}.pkl",
        "ANH" : f"pats_dict_ANH_{subsampled_str}.pkl"
}

in_file_medications_d = {
    "AHA": csv_dir/"AHA_TargetHF_Medicatie_outfinal.csv",
    "ANH": csv_dir/"ANH_TargetHF_Medicatie_outfinal.csv"
}

in_file_measurements_d = {
    "AHA": csv_dir/"AHA_TargetHF_Bepaling_outfinal2.csv",
    "ANH": csv_dir/"ANH_TargetHF_Bepaling_final.csv"
}
if not ns.include_medications and not ns.include_measurements:
    logger("WARN: Skipping merge_medications_measurements_on_patient_level as neither modality is enabled for use (see namespaces.py)")
    exit(0)

out_file_d = {
        "AHA" : f"pats_dict_AHA_1d1{subsampled_str}.pkl",
        "ANH" : f"pats_dict_ANH_1d1{subsampled_str}.pkl"
        }


def merge_medications_measurements_on_patient_level(promise_patients, promise_medications, promise_measurements):
    # :: Step 4 - merge records into a single object with patient as the first-class citizen
    t0 = logger(f'starting pats dict building (batch_size={batch_size})....')
    #  *************** mod_items with measurements  ************
    modalities_dict = {
        "Medications" : promise_medications,
        "Measurements": promise_measurements
    }
    mods_out_dict = {
        "Medications" : {},
        "Measurements": {}
    }

    p_id_vals, patients = promise_patients()
    
    patprac_id_extract = lambda ppmid: '|'.join(ppmid.split('|')[0:2]) # pat-prac-mod-id patid|pracid|modid
    for modality in modalities_dict.keys():
        # call promise to read modality data
        modalities_dict[modality] = modalities_dict[modality](p_id_vals)
        if modalities_dict[modality] is None:
            logger(f"SKIPING {modality}")
            continue

        logger(f"Modality = {modality}")

        logger(f"N = {len(modalities_dict[modality])} records")
        c_df = modalities_dict[modality]  # p_id_vals
        c_df = c_df[c_df['ptnt_prc_id'].isin(p_id_vals)]
        modalities_dict[modality] = c_df
        p_ids_with_modality = list(set(c_df['ptnt_prc_id'].values))
        logger(f"after filter on pat_ids left with {len(modalities_dict[modality])} records")

        id_key = {"Medications": 'ptnt_prc_med_id', 'Measurements': 'ptnt_prc_msrm_id'}[modality]
        date_key = {"Medications": 'medication_datetime', 'Measurements': 'measurement_datetime'}[modality]
        mod_cols = {"Medications": MEDICATION_COLS, 'Measurements': MEASUREMENT_COLS}[modality]
        mod_items = modalities_dict[modality]
        mod_id_vals = vals(mod_items[id_key])
        n_batches = round_up(len(mod_id_vals) / batch_size )
        t0 = None
        # Add modality item lists to mods_out_dict
        for c_batch in range(n_batches):
            batch_start = c_batch*batch_size
            batch_end = batch_start+batch_size
            mod_ids = mod_id_vals[batch_start:batch_end]
            t0 = logger(f"processing {modality} {batch_start} to {batch_end}... ({c_batch+1}/{n_batches})", t0)
            
            c_mod_items = mod_items[mod_items[id_key].isin(mod_ids)]
            p_ids = uniq(vals(c_mod_items['ptnt_prc_id']))
            if nuniq(c_mod_items[id_key]) != len(mod_ids):
                raise Exception(f"nuniq(c_mod_items.{id_key}) != len(mod_ids):")
            mod_items.drop(c_mod_items.index, inplace=True)
            for c_i in range(len(p_ids)):
                p_id = p_ids[c_i]
                c_c_mod_items = c_mod_items[c_mod_items['ptnt_prc_id'].isin([p_id])] # get mod_items for p_id
                c_c_mod_item_props = c_c_mod_items[mod_cols].to_dict('records') # make  list of med dicts for p_id

                if p_id in mods_out_dict[modality]:
                    prev_mod_item_props = mods_out_dict[modality][p_id]
                    mods_out_dict[modality][p_id] = prev_mod_item_props + c_c_mod_item_props
                else:
                    mods_out_dict[modality][p_id] = c_c_mod_item_props 

        for pid in p_ids_with_modality:
            mods_out_dict[modality][pid] = sorted(mods_out_dict[modality][pid], key= lambda item: item[date_key], reverse=T)

        # Add full modality item lists to each patient dict
        #p_ids_with_modality = list(mods_out_dict[modality].keys())
        
        n_batches = round_up(len(p_id_vals) / batch_size )
        t0 = None
        logger(f"Begin processing patients {modality} + mod_items/measurements (n_pat_ids =  {len(p_id_vals)})")
        #  *************** patients + mod_items  ************
        for c_batch in range(n_batches):
            batch_start = c_batch*batch_size
            batch_end = min(batch_start+batch_size, len(p_id_vals))
            p_ids = p_id_vals[batch_start:batch_end]
            t0 = logger(f"processing patient {batch_start} to {batch_end}... ({c_batch+1}/{n_batches})", t0)
            for p_id in p_ids:
                if p_id not in p_ids_with_modality:
                    patients[p_id][modality] = []
                else:
                    patients[p_id][modality] = mods_out_dict[modality][p_id]

    return patients

t0 = logger('Starting to read input files....')
nrows_medications = 1000000 if SUBSAMPLE_DATA  else None 
nrows_measures = nrows_medications

def set_pat_attr(p_dict, attr_nm, attr_vals):
    for i,k in enumerate(p_dict):
        p_dict[k].update([(attr_nm, attr_vals[i])] ) 
    return p_dict

for db_suffix in db_suffixes[::-1]:
    out_file = out_file_d[db_suffix]
    in_file_patients = in_file_patients_d[db_suffix]
    in_file_medications = in_file_medications_d[db_suffix]
    in_file_measurements = in_file_measurements_d[db_suffix]

    def promise_patients():
        patients = read_pickle(in_file_patients)
        logger(f"Read {len(patients)} patients")
        # sort episodes per patient, journals per episode (by time)
        for k in patients.keys():
            patients[k]['Episodes'] = sorted(patients[k]['Episodes'], key = lambda v: v['episode_start_date'], reverse=T)
            for i,e in enumerate(patients[k]['Episodes']):
                patients[k]['Episodes'][i]['JOURNALS'] = sorted(e['JOURNALS'], key = lambda v: v['journal_datetime'], reverse=T)

        # sort journals per episode            

        pid_vals = list(patients.keys())
        
        # util funcs
        reduce_replace = lambda x,repl : x if x != [] else repl
        apply_pat_lvl = lambda p_dict, p_fn : [p_fn(p) for p in p_dict.values()]
        apply_pat_lvl_items = lambda p_dict, p_fn : [p_fn(p) for p in p_dict.items()]

        # Need this df to get episode-level tagging of false-positives (so we can accurately tag on patient level after) 
        lukas_adj_df = read_pickle(infile_Lukas_adjudicated)
        pat_ids, ep_start_dates = get_HF_adj_pos_pat_ids(lukas_adj_df)
        lukas_adj_df['HF'] = lukas_adj_df['NO_HF'] != 'x'
        lukas_adj_df = add_composite_key(lukas_adj_df, 'patient', keep_components=False)
        lukas_adj_df = lukas_adj_df[['ptnt_prc_id', 'id', 'HF', 'episode_start_date', 'icpc_episode']]

        pat_adjs = apply_pat_lvl(patients, lambda x : x['adj_HF_diag'])
        pat_adjs = zip(pat_adjs, apply_pat_lvl_items(patients, lambda x : x[0]))
        pat_adjs = list(pat_adjs)
        pat_adjs = pd.DataFrame(pat_adjs)
        pat_adjs.columns = ['df_HF', 'ptnt_prc_id']

        adj_adjs = lukas_adj_df[['HF', 'ptnt_prc_id', 'episode_start_date']]
        #patients = pd.merge(patients, outcome_df, on = p_key, how='left').reset_index(drop=T)
        mrg_df = pd.merge(pat_adjs, adj_adjs, on = 'ptnt_prc_id', how = 'left').reset_index(drop=T)
        mrg_df['adj_HF'] = mrg_df['HF'] == T
        mrg_df = mrg_df.drop('HF', axis=1)
        false_positives = mrg_df[mrg_df['adj_HF'] != mrg_df['df_HF']]
        assert all(false_positives['df_HF'] == T) if nrow(false_positives) != 0 else T==T
        false_positives = false_positives[['ptnt_prc_id', 'episode_start_date']]
        gfps = false_positives.groupby('ptnt_prc_id')

    
        
        ep_dates = lambda x: [ e['episode_start_date'] for e in x['Episodes'] ]
        last_ep = lambda x: reduce_replace(ep_dates(x), [pd.NaT])[0]

        consult_dates = lambda x: sorted([j['journal_datetime'] for j in lol_to_l([e['JOURNALS'] for e in x['Episodes']])], reverse=T)
        last_consult  = lambda x: reduce_replace(consult_dates(x), [pd.NaT])[0] 
        
        is_ep_fp = lambda p_id, epdt: p_id in gfps.groups and epdt in gfps.get_group(p_id)['episode_start_date'].values
        # hf_ep will return episodes that are adjudicated as true from TARGET-HF
        hf_ep = lambda x : [ e['episode_start_date'] for e in x[1]['Episodes'] if (e['icpc_HF'] or e['text_HF']) and not is_ep_fp(x[0], e['episode_start_date']) ]
        t_hf = lambda x : reduce_replace(hf_ep(x), [pd.NaT])[0] 
        
        hf_icpc_ep = lambda x : [ e['episode_start_date'] for e in x[1]['Episodes'] if e['icpc_HF'] and not is_ep_fp(x[0], e['episode_start_date']) ]
        t_icpc_hf = lambda x : reduce_replace(hf_icpc_ep(x), [pd.NaT])[0] 

        hf_ep_text = lambda x : [ e['episode_start_date'] for e in x[1]['Episodes'] if e['text_HF'] and not is_ep_fp(x[0], e['episode_start_date']) ]
        t_text_hf = lambda x : reduce_replace(hf_ep_text(x), [pd.NaT])[0] 


        logger(f"[1]: cohort start/end time filtering")
        # apply funcs
        a_count = len(patients)
        patients = { k:v for k,v in patients.items() if len(consult_dates(v)) >= MIN_N_CONSULTS }
        a_diff = len(patients) - a_count
        if a_diff > 0:
            logger(f"Removing {a_diff} patients because no episode/consult data n={len(patients)}")

        pats_last_ep = apply_pat_lvl(patients, last_ep)
        pats_last_consult = apply_pat_lvl(patients, last_consult)

        pats_le_be4_cohort = [i for i,le in enumerate(pats_last_consult) if le < COHORT_TIME_START_DAYS ]
        if len(pats_le_be4_cohort) > 0:
            logger(f"Removing {len(pats_le_be4_cohort)} patients because no consult after cohort time start {COHORT_TIME_START_YEAR}.01.01 n={len(patients)}")
            patients = {kv[0]:kv[1] for i,kv in enumerate(patients.items()) if i not in pats_le_be4_cohort}
       
        logger(f"[2]: tagging on patinet-level")
        # tags from TargetHF
        # [ "t_min", "t_max", "deceased", "t_dereg", "t_birth", "sex", "t_death", "t_cvd_in_family",
        #  "t_coronary_artery_disease", "t_atrial_fibrillation", "t_heart_murmur", 
        #  "t_valvular_heart_disease", "t_hypertension", "t_stroke", "t_copd", 
        #  "t_diabetes_mellitus", "t_chronic_kidney_disease", "t_alcohol_abuse",
        #   "t_tobacco_use", "t_obesity", "t_material_deprivation", 
        #   "t_icpc_HF", "t_AF", "t_VHD", "t_HF", "t_text_AF", "t_text_VHD", "t_text_HF" ]

        tag_nms = [ "t_min", "t_max", "deceased", "t_dereg", "t_birth", "t_death", "t_cvd_in_family",
         "t_coronary_artery_disease", "t_atrial_fibrillation", "t_heart_murmur", 
         "t_valvular_heart_disease", "t_hypertension", "t_stroke", "t_copd", 
         "t_diabetes_mellitus", "t_chronic_kidney_disease", "t_alcohol_abuse",
          "t_tobacco_use", "t_obesity", "t_material_deprivation", "t_AF", "t_VHD", "t_text_AF", "t_text_VHD" ]

        patient_tags = try_read_pd_df(pqt_dir/f"persons_ty_{db_suffix}.parquet")
        patient_tags = patient_tags[tag_nms]
        personid_dict = { int(k.split('|')[0]):v for k,v in patients.items() }
        for t_nm in tag_nms:
            if t_nm.startswith('t_'):
                non_empty_vals = [x for x in vals(patient_tags[t_nm]) if x is not None]
                if non_empty_vals != []:
                    if type(non_empty_vals[0]) == str:
                        patient_tags[t_nm] = (pd.to_datetime(patient_tags[t_nm]) - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days 
                    else:
                        patient_tags[t_nm] = (patient_tags[t_nm] - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days 

        apply_pat_tag = lambda personid, person_tags: personid_dict[personid].update(person_tags) if personid in personid_dict else None
        for pid,row in patient_tags.iterrows():
            apply_pat_tag(pid, row.to_dict())

        del patient_tags
        patients = dict(zip(patients.keys(), personid_dict.values()))
        del personid_dict

        # End TargetHF tags

        # t_HF: time (in days since DATE_ARBITRARY_OFFSET_DAYS) until first HF diagnosis if diagnosed, pandas.NaT otherwise
        pats_t_hf = apply_pat_lvl_items(patients, t_hf)
        patients = set_pat_attr(patients, "t_HF", pats_t_hf)

        pats_t_icpc_hf = apply_pat_lvl_items(patients, t_icpc_hf)
        patients = set_pat_attr(patients, "t_icpc_HF", pats_t_icpc_hf)

        pats_t_text_hf = apply_pat_lvl_items(patients, t_text_hf)
        patients = set_pat_attr(patients, "t_text_HF", pats_t_text_hf)

        # inspect tags
        def inspect_tag(tag_nm):
            cnf_m3x = try_table(apply_pat_lvl(patients, lambda x: not pd.isnull(x[tag_nm] ))).to_dict()
            n_tps = cnf_m3x[T] if T in cnf_m3x else 0
            logger(f'True Positives first HF diag {tag_nm} in cohort =  {n_tps} (n = {len(patients)}) ({100*n_tps/len(patients):0.2f}%)')

        tag_nms += ['t_HF', 't_icpc_HF', 't_text_HF']
        for t_n in tag_nms:
            inspect_tag(t_n)



        logger(f"[3]: removing features we dont want")
        # p_features = ['age_days', 'patient_type', 'sex', 'postal_code', 'reg_date', 'dereg_date', 'dereg_cause', 'anonymous', 'missing', 'adj_HF_diag', 'follow_up_LAST', 'Episodes']
        p_features = ['age_days', 'sex', 'adj_HF_diag', 'follow_up_LAST', 'Episodes'] + tag_nms
        filter_features = lambda d,fs: {dk:d[dk] for dk in fs}
        filter_features_list = lambda ds,fs: [filter_features(d,fs) for d in ds]
        patients = {k:filter_features(v,p_features) for k,v in patients.items()}

        # e_features = [ "episode_start_date", "episode_end_date", "icpc_episode", "episode_attention", "episode_problem",
        #  "episode_status", "episode_description", "cvd_in_family", "coronary_artery_disease",
        #   "atrial_fibrillation", "heart_murmur", "valvular_heart_disease", "hypertension",
        #    "stroke", "copd", "diabetes_mellitus", "chronic_kidney_disease",
        #     "alcohol_abuse", "tobacco_use", "obesity", "material_deprivation", 
        #     "text_HF", "icpc_HF", "JOURNALS" ]
        # filter out episode-level features 
        e_features = [ "episode_start_date", "episode_end_date", "icpc_episode", "episode_status", "episode_description", "JOURNALS" ]
        patients = { k:{**v, **{'Episodes' :filter_features_list(v['Episodes'], e_features)} } for k,v in patients.items() }


        # j_features = [ "journal_datetime", "contact_type", "icpc_episode", "icpc_journal",
        #  "icpc_s", "icpc_o", "icpc_e", "icpc_p", "icpc_x", "text_s", "text_o", "text_e", 
        #  "text_p", "text_x", "dyspnea", "edema", "chest_complaints", "palpitations", "dizziness", "syncope", "tiredness" ]
        # filter out journal-level features 
        j_features = [ "journal_datetime", "contact_type", "icpc_journal",
         "icpc_s", "icpc_o", "icpc_e", "icpc_p", "icpc_x", "text_s", "text_o", "text_e", 
         "text_p", "text_x" ]
        filter_j_feats_for_pat = lambda pat, j_features: [ {**ep, **{'JOURNALS' :filter_features_list(ep['JOURNALS'], j_features)} } for ep in pat['Episodes'] ]
        patients = { k:{**v, **{'Episodes' :filter_j_feats_for_pat(v, j_features)} } for k,v in patients.items() }

        past_hf_pids = [k for k,v in patients.items() if not pd.isnull(v['t_HF']) and v['t_HF'] < COHORT_TIME_START_DAYS]
        logger(f"[4] removing {len(past_hf_pids)} patients who had first HF diag before cohort start  (n={len(patients)})" )
        patients = {k:v for k,v in patients.items() if k not in past_hf_pids}
        logger(f"Left with (n={len(patients)}) patients" )

        logger(f"[5] apply 3 month censoring window (on episodes, tags) prior to outcome (if outcome = T)")
        # pats[0]['follow_up_LAST'] - hard-stop, 
        # pats[0][']
        hf_pids = [k for k,v in patients.items() if not pd.isnull(v['t_HF'])]
        is_dt_dur_del_window = lambda hf_dt, _dt : _dt + FOLLOW_UP_HFPOS_CENS_WINDOW  >= hf_dt 

        logger(f'Found {len(hf_pids)} patients for censoring window')
        effective_count = 0
        log_first_n = 10
        censor_journals_inner = lambda hf_dt, v: v if v != 'JOURNALS' else [j for j in v if not is_dt_dur_del_window(hf_dt, j['journal_datetime'])] 
        censor_journals = lambda hf_dt, e: {k : censor_journals_inner(hf_dt, v) for k,v in e.items()}
        for i, hf_pid in enumerate(hf_pids):
            n_be4 = len( patients[hf_pid]["Episodes"] )
            hf_dt = patients[hf_pid]['t_HF']
            patients[hf_pid]['Episodes'] = [censor_journals(hf_dt, e) for e in patients[hf_pid]['Episodes'] if not is_dt_dur_del_window(hf_dt, e['episode_start_date'])]
            n_aftr = len( patients[hf_pid]["Episodes"] )
            if n_be4 != n_aftr:
                effective_count += 1
                if effective_count < log_first_n:
                    logger(f'{hf_pid} n_eps before censor = {n_be4}')
                if effective_count < log_first_n:
                    logger(f'{hf_pid} n_eps after censor = {n_aftr}')

        pid_vals = list(patients.keys())
        return pid_vals, patients

    def promise_modality(in_file, pids, nrows_mod, read_mod, use_mod):
        mod_df = read_mod(in_file, nrows = nrows_mod, pids=pids) if use_mod else None
        return mod_df

    pr_meds = lambda pids: promise_modality(pqt_dir/in_file_medications, pids, nrows_medications, read_medications, ns.include_medications)
    pr_meas = lambda pids: promise_modality(pqt_dir/in_file_measurements, pids, nrows_measures, read_measurements, ns.include_measurements)

    pats_dict = merge_medications_measurements_on_patient_level(promise_patients, pr_meds, pr_meas)

    save_pickle(out_file, pats_dict)

logger('pats dict write to file....DONE ', t0)
logger("DONE", start_time)
print("DONE")
