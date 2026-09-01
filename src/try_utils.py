from constants import *
import pandas as pd
import numpy as np
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import time
import json
from datetime import datetime as dt
from datetime import timedelta, date
import os
import sys
import pickle
import re
from matplotlib import pyplot as plt
from multiprocessing import Pool as ProcessPool
import sys
import statsmodels.api as sm
from itertools import product 
from try_logger import set_logfilename, get_logfilename, set_logger, get_logger

import random
import traceback
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# debug and commandline args
def check_if_debugging(IS_DEBUG):
    if IS_DEBUG: 
        try_log('THIS IS DEBUG')
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        try_log("Waiting for client to attach...")
        debugpy.wait_for_client()
        try_log("Houston We Have Liftoff.")
    else:
        try_log("THIS IS NOT DEBUG")

def parse_commandline_args(verbose = T, default = "T", required_extra_args = []):
    if os.getcwd().endswith('/src'):
        os.chdir(f"{os.getcwd()}/..")
    argv = sys.argv
    is_extra_args = len(required_extra_args) > 0
    is_flask_run = argv[0].endswith('flask') and argv[1] == 'run'
    is_argless_script = argv[0] in ['test_imports.py', '']

    if is_flask_run:
        argv = ['flask', default, default] 
    elif is_argless_script:
        argv = [argv[0], default, default] 

    validate_commandline_args(argv, 3 + len(required_extra_args)) # e.g., python -u script.py(1) T(2)  T(3) custom_target=blueberry(4)
    args = {
            "IS_DEBUG" : eval(argv[1]),
            "SUBSAMPLE_DATA" : eval(argv[2])
            }

    if is_extra_args:
        provided_extra_args = sys.argv[3:]
        assert len(provided_extra_args) == len(required_extra_args)
        
        for arg_s in sys.argv[3:]:
            kv_pair = arg_s.split("=")
            assert len(kv_pair) == 2
            arg_n = kv_pair[0]
            assert arg_n in required_extra_args
            assert arg_n not in args
            arg_v = kv_pair[1]
            args[arg_n] = arg_v

    if verbose:
        for k,v in args.items():
            print(f"COMMANDLINE FLAG PARSED: {k} = {v}")
    return args 

def validate_commandline_args(argv, expected_n_args):
    # argv[1] = IS_DEBUG T/F
    # argv[2] = SUBSAMPLE_DATA T/F

    assert len(argv) >= expected_n_args
    assert argv[1] in ['T', 'F']
    assert argv[2] in ['T', 'F']

# Logging & printing


def try_print_col_counts_by_type(df, log=print):
    cols = cns(df)
    get_col_types = lambda cols : list(set(["_".join(x.split('_')[:-1] if "text__t" not in x else x.split('_')[:-3]) for x in cols if "_" in x]))
    get_cols_per_type = lambda ctype, cols : [i for i in cols if i.startswith(f"{ctype}_") ]
    count_cols_per_type = lambda ctype, cols : len(get_cols_per_type(ctype, cols))
    log(f"column groups in new flat df {dim(df)}= ")
    ctypes = get_col_types(cols)
    cols_without_types = try_sd(cols, [ c for ctp in ctypes for c in get_cols_per_type(ctp, cols)])
    ctypes_counts = sorted([(ctp,count_cols_per_type(ctp, cols)) for ctp in ctypes ], key=lambda x:x[1], reverse=T)
    res = try_print_list([f"{ctp}; count={count}" for ctp,count in ctypes_counts ], log)
    res = f"{res}\n{try_print_list(cols_without_types, log)}"
    return res

def datetime_str():
    now = dt.now()
    return f"::|{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second}|::"


def cat(lst, sep = '\n', end='\n'):
    print(sep.join(map(str, lst)), end=end )


def __custom_print_fn(msg, file_out):
    with open(f"log/{file_out}", "a") as log_file:
        print(msg, file=log_file)

def _get_custom_print_fn(file_out):
    if file_out is None:
        return print
    return lambda msg: __custom_print_fn(msg, file_out)

def try_table(vs):
    return pd.Series(vs).value_counts(dropna=F)
    
def try_log(msg, start_time=None, track_time=True, flush=False, file_out=None):
    '''
    @Deprecated - use get_logger_fn(logfile)(...) or get_default_logger_fn(__file__)
    @flush - useful when running multiprocessing to force writing log to file on time
    '''
    end_time = None if start_time is None else round(time.time() * 10)
    start_time = start_time if start_time is not None else round(time.time() * 10)
    time_dur_secs = None if end_time is None else (end_time - start_time) / 10
    file_out = get_logfilename() if get_logfilename() else file_out
    custom_print_fn = _get_custom_print_fn(file_out)

    if not track_time:
        custom_print_fn(f"{datetime_str()}\t\t {msg}")
    else:
        if time_dur_secs is None:
            custom_print_fn(f"{datetime_str()}\t=== {msg}  :::")
        else:
            time_dur_min = round(time_dur_secs / 60) if time_dur_secs >= 60  else 0
            time_dur_secs = round(time_dur_secs % 60, 1)
            time_dur_hour = round(time_dur_min / 60) if time_dur_min >= 60 else 0
            time_dur_min = time_dur_min % 60
            time_dur_str = f"{time_dur_hour}h " if time_dur_hour > 0 else ""
            time_dur_str = f"{time_dur_str}{time_dur_min}m " if time_dur_min > 0 else time_dur_str
            time_dur_str = f"{time_dur_str}{time_dur_secs}s" if time_dur_secs > 0 else time_dur_str
            custom_print_fn(f"{datetime_str()}\t>>> {msg} ({time_dur_str}) *** \n")

    if time_dur_secs is None:
        return(start_time)
    if flush:
        sys.stdout.flush()
    return(end_time)

def get_logger_fn(file_out, override=True):
    set_logfilename(file_out, override=override)
    file_out = get_logfilename()
    try: # delete previous log file 
        os.remove(f"log/{file_out}")
    except OSError:
        pass
    logger = get_logger()
    if logger and not override:
        return logger

    def custom_logger(msg, start_time=None, track_time=True, flush=False):
        kwargs = {
            "start_time": start_time,
            "track_time": track_time,
            "flush": flush,
            "file_out": file_out
        }
        return try_log(msg, **kwargs)
    try_log(f"Starting log in {file_out}...")
    set_logger(custom_logger, override=True)
    return custom_logger

def get_default_logger_fn(module_filename, override=True, verbose=False):
    SUBSAMPLE_DATA = parse_commandline_args(verbose=verbose)["SUBSAMPLE_DATA"]
    subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
    outfile_infix = f"{subsampled_str}"
    logfile = f'{os.path.basename(module_filename)[:-3]}_{outfile_infix}.log'
    set_logfilename(logfile, override=override)
    logfile = get_logfilename()
    logger = get_logger_fn(logfile, override=override)
    print(f"Starting {module_filename}...")
    print(f"Script output written to {logfile}...")
    return logger


logger = get_default_logger_fn(__file__, override=False)


class NPEncoder(json.JSONEncoder):
    def default(self, obj):
        dtypes = (np.datetime64, np.complexfloating, pd.Timestamp)
        if isinstance(obj, dtypes):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd._libs.tslibs.nattype.NaTType):
            return None 
        
        return super(NPEncoder, self).default(obj)


def round_up(x):
    return((int)(np.ceil(x)))


def round_down(x):
    return((int)(np.floor(x)))

def fprint(o):
    if isinstance(o, list):            
        for oo in o:
            print(oo, flush=True)
    else:
        print(o, flush=True)

def dim(df):
    return(df.shape)

def nrow(df):
    return(df.shape[0])

def ncol(df):
    return(df.shape[1])

def cns(df):
    return(df.columns.to_list())

def pcns(df):
    print(cns(df))

def ccns(df):
    print(cat(cns(df)))

def sort_cols(df):
    return(df.reindex(sorted(df.columns), axis = 1))

def concate_cols(df, cols, new_col_nm, sep=''):
    if df.empty:
        df[new_col_nm] = pd.Series([], dtype=str)
    else:
        df[new_col_nm] = df[cols].astype(str).agg(sep.join, axis=1)
    return df

def try_extract_file_extension(filepath):
    parts = str(filepath).split('.')
    if len(parts) == 0:
        raise Exception(f"Can not extract file extension from filepath {filepath}")
    return parts[-1]

def try_read_pd_df(filepath, nrows=None, dtypes=None):
    logger(f"try_read_pd_df for {filepath} ... ")
    mapping_readers = {
        "parquet" : try_read_partquet,
        "csv": try_read_csv,
        "tsv": try_read_tsv
    }
    ftype = try_extract_file_extension(filepath)
    if ftype not in mapping_readers:
        raise NotImplementedError(f"try_read_pd_df not implemented for ftype={ftype}")
        return None
    return mapping_readers[ftype](filepath, nrows, dtypes)

def try_read_csv(f, nrows=None, dtypes=None, sep=",", subdir='csv', encoding='utf-8'):
    filepath = f"{subdir}/{f}" if not str(f)[0] == '/' else f
    df = None
    if nrows is None:
        df = pd.read_csv(filepath, dtype=dtypes, sep=sep, encoding=encoding)
    else:
        df = next(pd.read_csv(filepath, dtype=dtypes, chunksize=nrows, sep=sep, encoding=encoding))
    logger(f"try_read_csv - read {nrow(df)} rows")

    return df

def try_read_tsv(f, nrows=None, dtypes=None, sep=",", encoding='utf-8'):
    return try_read_csv(f, nrows, dtypes, '\t', 'tsv', encoding)

def try_read_partquet(f, nrows=None, dtypes=None):
    if nrows is None:
        return pd.read_parquet(f)
    pf = ParquetFile(f)
    first_n_rows = next(pf.iter_batches(batch_size=nrows))
    df = pa.Table.from_batches([first_n_rows]).to_pandas()
    logger(f"try_read_partquet - read {nrow(df)} rows")
    return df

def try_cached_call(fn, io_r, io_c, io_w, override_cache=False, cache_fn=None, **kwargs):
    # Note: cache key is the caller function's name, which is assumed (not guaranteed) to be unique across callers
    cache_filename = f"___ch_{fn.__name__}.pkl" if cache_fn is None else f"___ch_{cache_fn}.pkl"
    if not override_cache and io_c(cache_filename):
        logger(f"Skip execution of {fn.__name__}, loading last output from cache")
        return io_r(cache_filename)
    else:
          f_out = fn(**kwargs)
          io_w(cache_filename, f_out)
          return f_out


def add_composite_key(df, unit, dedup=False, extra_sort_keys=[], keep_components=False):
    key_nm = MAPPING_COMPOSITE_KEYS[unit]['key_nm']
    key_comp = MAPPING_COMPOSITE_KEYS[unit]['key_comp']
    logger(f"@add_composite_key {key_nm} = {key_comp} (nrow = {nrow(df)})")
    df = concate_cols(df, key_comp, key_nm, sep="|")
    if dedup:
        logger(f"@add_composite_key before dedup {nrow(df)}")
        df = df.sort_values(by=[key_nm] + extra_sort_keys, ascending=False)
        df = df.groupby(key_nm).head(1)
        logger(f"@add_composite_key after dedup {nrow(df)}")
    if not keep_components:
        df = df.drop(key_comp, axis=1)
    return(df)

def read_patients(in_path, nrows = None, pat_ids = [], pat_tags = None):
    """
    set age based on 'arbitrary' num days after date here (jan 1 1970)
    remove mentions of cohort start time (legacy).
    age will be determined from first follow-up date of patient 
    """
    t0 = logger(f"Reading patients from {in_path}...")
    p_key = MAPPING_COMPOSITE_KEYS['patient']['key_nm'] # person_id + practice_id (person_id alone was later said to be unique too, but we keep the composite key)
    df = try_read_pd_df(in_path, nrows=nrows)
    df = add_composite_key(df, 'patient', 
                        dedup=True, 
                        extra_sort_keys=['reg_date', 'dereg_date'],
                        keep_components=True
                        )
    logger(f'Read {nrow(df)} records')
    if pat_ids:
        logger(f"Going to select only patients using {len(pat_ids)} ids.")
        df = df[df[p_key].isin(pat_ids)]
        logger(f'Left with {nrow(df)} records')
    if pat_tags is not None:
        logger(f"Going to add {ncol(pat_tags)} tags: {cns(pat_tags)}")
        pat_tags[p_key] = pat_tags['pat_prac_id']
        pat_tags = pat_tags.drop(columns=['pat_prac_id'])
        df = pd.merge(df, pat_tags, on = p_key, how='left')
        logger(f'Left with {ncol(df)} columns in patient dataframe')


    # we want age in days to be the number of elapsed days from the patients birth till the cohort start time (convention)
    # conv_ts_to_days counts days after the start_time, we want to cound days before, essentially flip the sign of the counted number

    df = df.rename(columns={'birth_date' : 'age_days'})
    no_age_rows = len([x for x in vals(df['age_days']) if pd.isnull(x)])
    if no_age_rows > 0:
        logger(f"Removing {no_age_rows} patients with no birth date information")
        df = df[~df['age_days'].isnull()]

    cnv_date_fn = lambda x: conv_ts_to_days(x, offset=DATE_ARBITRARY_OFFSET_DAYS)
    df['age_days'] = df['age_days'].apply(lambda x: -cnv_date_fn(x))

    df['reg_date'] = df['reg_date'].apply(cnv_date_fn)
    df['dereg_date'] = df['dereg_date'].apply(cnv_date_fn)

    logger(f"After dedup: df with {df.shape[0]} rows and {df.shape[1]} columns")
    logger(f"Reading patients from {in_path}...DONE", t0)
    tag_names = cns(pat_tags) if pat_tags is not None else []
    return vals(df[p_key]), tag_names, df

def print_age_group_counts(df, log=print):
    ages = df[VAR_FOLLOW_UP_DATE] + df['age_days']
    ages = (ages.values/(365*5)).round()
    ages = pd.Series(ages)
    ages_counts_str = sorted(list(ages.value_counts().to_dict().items()), key=lambda x:x[0])
    return try_print_list([ f"[{int(k-1)*5}-{int(k)*5}]=#{v}" for k,v in ages_counts_str], log)

def apply_min_age_selection(df, min_age_days = THRESHOLD_MIN_AGE_DAYS):
    """
    """
    print_age_group_counts(df)
    
    nrows_removed = nrow(df[df[VAR_FOLLOW_UP_DATE] + df['age_days']  <= min_age_days])
    logger(f"Removing {nrows_removed} patients for having age less than {min_age_days/365} years")
    df = df[df[VAR_FOLLOW_UP_DATE] + df['age_days'] > min_age_days]
    print_age_group_counts(df)
    logger(f'Left with {nrow(df)} records')
    
    return df

def read_measurements(in_path, nrows=None, pids =[]):
    # Notes:
    # Many times journal_id / episode_id are missing
    # we always have datum, pat_prac_id, measurement_id 
    # columns Episode_icpc - seems empty?
    # vraagtype -categorical, with missing values 
    # NHGnummer - measurementType (many types..)
    # Uitslag_type, Soort, Afwijkende_uitslag - categorical
    # seems Bijzonder === Afwijkende_uitslag  ??
    t0 = logger(f"Reading measurements from {in_path}...")
    df = try_read_csv(in_path, nrows=nrows, dtypes=CSV_DTYPES['measurements'], sep="|", encoding="latin_1")
    df.columns = ['person_id', 'practice_id', 'import_id', 'measurement_id', 'contact_id', 'journal_id', 'episode_id', 'measurement_datetime', 'episode_icpc', 'nhgnummer', 'vraagtype', 'uitslag_type', 'soort', 'omschrijving', 'eenheid', 'referentie_min', 'referentie_max', 'afwijkende_uitslag', 'memo_mat_bijz', 'bijzonder', 'materiaal', 'memo', 'opmerking2', 'toelichting2', 'uitslag2', 'uitslag_tekst2']
    logger(f"Read {nrow(df)} measurements from {str(in_path)} OK")
    for col in cns(df):
        perc_missing = round((df[col].isna().sum() / nrow(df) ) * 100, 2)
        logger(f"Column: {col}: missing {perc_missing}% of time, {len(df[col].value_counts(dropna=F))} uniq values")

    dateless_measures = [i  for i,x in enumerate(vals(df["measurement_datetime"])) if type(x) != str]
    if dateless_measures:
        logger(f"Removing {len(dateless_measures)} rows which dont have a date")
        df = df.drop(dateless_measures, axis=0)
        df = df.reset_index(drop=T)
    logger(f'_______ Top 10 EARLIEST measurement_datetime = \n{sorted(df["measurement_datetime"])[:10]}')
    logger(f'_______ Top 10 LATEST measurement_datetime = \n{sorted(df["measurement_datetime"], reverse=T)[:10]}')
    df['measurement_datetime'] = (pd.to_datetime(df['measurement_datetime']) - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days # much faster
    logger(f'Filtering based on cohrot start/end time ({COHORT_TIME_START_YEAR}/{COHORT_TIME_END_YEAR})')
    df = df[df['measurement_datetime'] >= COHORT_TIME_START_DAYS] 
    df = df[df['measurement_datetime'] <= COHORT_TIME_END_DAYS]
    logger(f'Left with {nrow(df)} measurement entries')

    df = add_composite_key(df, 'measurement', dedup=True, keep_components=True)
    df = add_composite_key(df, 'patient', extra_sort_keys=['measurement_datetime'], keep_components=True)
    df = df.drop(columns=["import_id", "practice_id", "person_id", "episode_id", "measurement_id", "contact_id", "journal_id"], axis = 1)

    if pids != []:
        df = df[df['ptnt_prc_id'].isin(pids)]
        logger(f"after filter on pat_ids left with {len(df)} records")

    txt_cols = ['omschrijving', 'memo', 'referentie_min', 'referentie_max', 'opmerking2', 'toelichting2', 'uitslag2', 'uitslag_tekst2']
    if nrow(df) > 0:
        df['measurement_txt'] = df[txt_cols].astype(str).agg(' '.join, axis=1)

    cols_to_keep = ['measurement_datetime', 'nhgnummer', 'measurement_txt',
                     MAPPING_COMPOSITE_KEYS['patient']['key_nm'], MAPPING_COMPOSITE_KEYS['measurement']['key_nm']]
    cols_to_remove = [c for c in cns(df) if c not in cols_to_keep]
    assert all([ c in cns(df) for c in cols_to_keep])
    df = df.drop(columns=cols_to_remove, axis=1)

    logger(f"After dedup: df with {df.shape[0]} rows and {df.shape[1]} columns")
    logger(f"Reading measurements from {in_path}...DONE", t0)
    return df 

def read_medications(in_path, nrows=None, pids=[]):
    t0 = logger(f"Reading medications from {in_path}...")
    df = try_read_csv(in_path, nrows=nrows, dtypes=CSV_DTYPES['medications'], sep="|", encoding="latin_1")
    df.columns = [ 'person_id', 'practice_id', 'import_id', 'medication_id', 'contact_id', 'journal_id', 'episode_id', 'episode_icpc',
            'specialisme', 'actueel', 'medication_datetime', 'afleverdatum', 'einddatum',
            'atc_code', 'atc_omschrijving', 'omschrijving', 'hoeveelheid',
            'aflever_eenheid', 'product_sterkte', 'chronisch', 'volgnummer_herhaling', 
            'zindex_nummer', 'prk', 'gpk', 'hpk', 'toedieningsweg', 'gebruiksvoorschrift2', 'vrije_tekst2']
            
    df = df[[ 'person_id', 'practice_id', 'medication_id', 'medication_datetime',
            'atc_code', 'atc_omschrijving', 'omschrijving', 'gebruiksvoorschrift2', 'vrije_tekst2']]
    logger(f"Read {nrow(df)} medications from {str(in_path)} OK")
    cols_missings = sorted([(col, round((df[col].isna().sum() / nrow(df) ) * 100, 2)) for col in cns(df)], key=lambda x: x[1], reverse=T)
    for col, perc_missing in cols_missings:
        logger(f"Column: {col}: missing {perc_missing}% of time, {len(df[col].value_counts(dropna=F))} uniq values")

    dateless_measures = [i  for i,x in enumerate(vals(df["medication_datetime"])) if type(x) != str]
    if dateless_measures:
        logger(f"Removing {len(dateless_measures)} rows which dont have a date")
        df = df.drop(dateless_measures, axis=0)
        df = df.reset_index(drop=T)
    logger(f'_______ Top 10 EARLIEST medication_datetime = \n{sorted(df["medication_datetime"])[:10]}')
    logger(f'_______ Top 10 LATEST medication_datetime = \n{sorted(df["medication_datetime"], reverse=T)[:10]}')

    df = add_composite_key(df, 'patient', extra_sort_keys=['medication_datetime'], keep_components=True)
    if pids != []:
        df = df[df['ptnt_prc_id'].isin(pids)]
        logger(f"after filter on pat_ids left with {len(df)} records")
        
    df = add_composite_key(df, 'medication', dedup=True, keep_components=True)
    df = df.drop(columns=["practice_id", "person_id", "medication_id"], axis = 1)


    df['medication_datetime'] = (pd.to_datetime(df['medication_datetime']) - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days # much faster
    logger(f'Filtering based on cohrot start/end time ({COHORT_TIME_START_YEAR}/{COHORT_TIME_END_YEAR})')
    df = df[df['medication_datetime'] >= COHORT_TIME_START_DAYS] 
    df = df[df['medication_datetime'] <= COHORT_TIME_END_DAYS]
    logger(f'Left with {nrow(df)} medication entries')

    if nrow(df) > 0:
        df['medication_txt'] = df[['omschrijving', 'atc_omschrijving', 'gebruiksvoorschrift2', 'vrije_tekst2']].astype(str).agg(' '.join, axis=1)

    cols_to_keep =  MEDICATION_COLS + [MAPPING_COMPOSITE_KEYS['patient']['key_nm'], MAPPING_COMPOSITE_KEYS['medication']['key_nm']]
    cols_to_remove = [c for c in cns(df) if c not in cols_to_keep]
    assert all([ c in cns(df) for c in cols_to_keep])
    df = df.drop(columns=cols_to_remove, axis=1)

    logger(f"After dedup: df with {df.shape[0]} rows and {df.shape[1]} columns")
    logger(f"Reading medications from {in_path}...DONE", t0)
    return df 

def conv_ts_to_days(x, offset = COHORT_TIME_START_DAYS):
    if offset == COHORT_TIME_START_DAYS:
        logger("WARN: using legacy value for offset in conv_ts_to_days! switch to arbitrary offset")

    if pd.notna(x):
        return round(x.timestamp()/(60*60*24)) - offset
    return None

def conv_str_date_to_days(x, format_str="%m/%d/%Y", offset = COHORT_TIME_START_DAYS):
    if offset == COHORT_TIME_START_DAYS:
        logger("WARN: using legacy value for offset in conv_ts_to_days! switch to arbitrary offset")
    if pd.notna(x):
        return round(pd.to_datetime(x, format=format_str).timestamp()/(60*60*24)) - offset
    return None

def conv_date_days_to_str(days):
    delta = timedelta(days)
    target_date = date(DATE_ARBITRARY_OFFSET_YEAR, 1, 1) + delta
    return target_date.strftime("%Y-%m-%d")

def get_current_time_str(fstring='%Y_%m_%d_%H%M'):
    return dt.today().strftime(fstring)


def fetch_diag_TARGETHF():
    lukas_adj_df = try_read_pickle(infile_Lukas_adjudicated)
    pat_ids, ep_start_dts, hf_diag = get_HF_adj_pos_pat_ids(lukas_adj_df, return_all=T)
    outcome_df = pd.DataFrame({'id': pat_ids, 'episode_start_date': ep_start_dts, VAR_OUTCOME: hf_diag})
    return outcome_df

def calc_TARGETHF_scores(X):
    from TargetHF_ty.targethf.definitions import icpc_def
    from lifelines import KaplanMeierFitter, CoxPHFitter

    # ## Loading
    data_dir = old_data_dir
    # Define paths
    pqt_dir = data_dir/"parquet"
    cht_dir = data_dir/"cohort_A-ICPC"

    cohort = pd.read_parquet(cht_dir/"cohort_ty_merged.parquet") # load some dummy data to fit a dummy model


    cohort["journal_datetime"] = cohort["journal_datetime"].fillna(0)
    cohort["avg_use"] = (cohort["journal_datetime"]/cohort["years_history"]).fillna(0)

    risk_factors = list(icpc_def.risk_factors.keys())
    predictors = ["decades_age", "male"] + risk_factors

    cohort = cohort[predictors+["time_to_event", "event"]]

    targetHF_params = { 'decades_age' : 1.96,
                            'male' : 1.28,
                            'alcohol_abuse' : 1.55,
                            'obesity' : 1.10,
                            'material_deprivation' : 0.75,
                            'hypertension' : 1.09,
                            'diabetes_mellitus' : 1.44 ,
                            'coronary_artery_disease' : 1.52,
                            'atrial_fibrillation' : 2.10,
                            'heart_murmur' : 1.42,
                            'valvular_heart_disease' : 1.75,
                            'stroke' : 1.14,
                            'copd' : 1.47,
                            'chronic_kidney_disease' : 1.21,
                            'cvd_in_family' : 1.00,
                            'tobacco_use' : 1.00 }


    # T.Y. 
    def l_one_cox(df, lambda_, override_params = None):
            cph = CoxPHFitter(l1_ratio = 1.0 , penalizer=lambda_)
            #df['fake_event'] = random.choices([0,1], k =nrow(df)) # prevent converange error
            df['atrial_fibrillation'] = random.choices([F,T], k =nrow(df)) # prevent converange error
            cph.fit(df, duration_col="time_to_event", event_col="event")
            if override_params:
                cph.params_ = pd.Series(override_params)
            return cph



    cph_l_one = l_one_cox(cohort, 0.0005, targetHF_params)
    cph_l_one.params_
    hrs = cph_l_one.predict_partial_hazard(cohort)
    hrs = cph_l_one.predict_partial_hazard(X)
    return hrs


def calc_percentage(n, ntot):
    return 100*n/max(ntot, 1e-6)

def construct_full_outcome_df(pat_ids, episodes):
    """
    """
    p_key = MAPPING_COMPOSITE_KEYS['patient']['key_nm']

    outcome_df = fetch_diag_TARGETHF() # now contains TPs and FPs. Still need to enrich with TNs/FNs (FNs are assumed to be zero).
    # also contains patients from both ANH and AHA, filter those out here first
    outcome_df = outcome_df[outcome_df['id'].isin(pat_ids)]
    missing_ids = try_sd(pat_ids, outcome_df['id'])
    n_TNs_FNs = len(missing_ids)
    n_pats = len(pat_ids)
    tbl = {T: 0, F:0}
    tbl.update(try_table(outcome_df[VAR_OUTCOME]).to_dict())
    logger(f"{VAR_OUTCOME} STATS: (N pats={n_pats})")
    logger(f"TPs = {tbl[T]}/{n_pats} ({calc_percentage(tbl[T], n_pats):.1f}%)")
    logger(f"FPs = {tbl[F]}/{n_pats} ({calc_percentage(tbl[F], n_pats):.1f}%)")
    logger(f"TNs/FNs = {n_TNs_FNs}/{n_pats} ({calc_percentage(n_TNs_FNs, n_pats):.1f}%)")

    negs = episodes[episodes[p_key].isin(missing_ids)][[p_key, 'episode_start_date']]
    negs = negs.reset_index(drop=T)
    idx = negs.groupby(p_key)['episode_start_date'].idxmax()
    negs = negs.loc[idx].reset_index(drop=T) # keep only last recorded episode
    negs = negs.sort_values('episode_start_date', ascending=F)
    negs = negs.reset_index(drop=T)

    
    hf_diag_neg = [F] * n_TNs_FNs
    pat_ids_TNs_FNs_zero_eps = try_sd(missing_ids, vals(negs[p_key]))
    ids_of_interest_neg = vals(negs[p_key]) + pat_ids_TNs_FNs_zero_eps # keep order
    ep_start_dts_neg = vals(negs["episode_start_date"]) + [np.nan]*len(pat_ids_TNs_FNs_zero_eps) # pats without episodes have nan episdoe start date

    extra_outcome_df = pd.DataFrame({
        'id': ids_of_interest_neg,
        'episode_start_date': ep_start_dts_neg,
        VAR_OUTCOME : hf_diag_neg
    })

    outcome_df_full = pd.concat([extra_outcome_df, outcome_df], axis=0)
    outcome_df_full.columns = [p_key, VAR_FOLLOW_UP_DATE, VAR_OUTCOME] # NOTE : VAR_FOLLOW_UP_DATE here is based on last episode start date... will be adjusted later..
    outcome_df_full = outcome_df_full.reset_index(drop=T)
    return outcome_df_full

def filter_follow_up_period(
                df, 
                outcome_df,
                id_col,
                dt_col,
                dt_last_flwp_col,
                flpw_period=FOLLOW_UP_PERIOD_DAYS,
                append_outcome=T
                            ):
    df = pd.merge(df, outcome_df, on = id_col, how='left') # add adj_HF_diag 
    df = df.reset_index(drop=T) # filter where date was before last_flwp date was within flwp duration
    # e.g., for episode -  keep only rows where : 
    # 1) episode_start_date is before last flwp, AND 
    # 2) episode_strt_date is at most flpw_period [2 years in the experiment] away from the last flwp
    df = df.loc[(df[dt_col] <= df[dt_last_flwp_col]) & (df[dt_last_flwp_col] - df[dt_col] <= flpw_period) ]
    df = df.drop(columns=[dt_last_flwp_col] if append_outcome else [VAR_OUTCOME, dt_last_flwp_col])
    df = df.reset_index(drop=T)
    return df

# returns a dataframe of read episdoes, ids of patients who already had the outcome outside of the cohort time interval 
def read_episodes(in_path, nrows = None, pat_ids = [], skip_follow_up_time_filtering = True, is_debug = False):
    t0 = logger(f"Reading episodes from {in_path}...")
    df = try_read_pd_df(in_path, nrows=nrows, dtypes=CSV_DTYPES['episodes'])
    logger(f"Read {nrow(df)} episodes from {str(in_path)}")
    # plt.hist(df["episode_start_date"], bins = 100)
    # plt.savefig('plots/hist_ep_start_date.png')

    p_key = MAPPING_COMPOSITE_KEYS['patient']['key_nm']
    df = add_composite_key(df, 'patient', keep_components=True)
    if is_debug:
        logger(f"DEBUG: going to set all episodes read to patients available (useful when reading subset of input files)")
        df[p_key] = random.choices(pat_ids, k=nrow(df[p_key]))
    if pat_ids:
        logger(f"Going to select only episodes using {len(pat_ids)} pat ids.")
        df = df[df[p_key].isin(pat_ids)]
        logger(f'Left with {nrow(df)} records')

    logger(f'_______ Top 10 EARLIEST episode_start_dates = \n{sorted(df["episode_start_date"])[:10]}')
    logger(f'_______ Top 10 LATEST episode_start_dates = \n{sorted(df["episode_start_date"], reverse=T)[:10]}')
    logger(f'_______ Top 10 EARLIEST episode_end_dates = \n{sorted([x for x in df["episode_end_date"] if type(x) == str])[:10]}')
    logger(f'_______ Top 10 LATEST episode_end_dates = \n{sorted([x for x in df["episode_end_date"] if type(x) == str], reverse=T)[:10]}')
    df['episode_start_date'] = (pd.to_datetime(df['episode_start_date']) - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days # much faster
    df['episode_end_date'] = (pd.to_datetime(df['episode_end_date']) - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days # much faster
    # remove episodes with impossible dates (after the last possible cohort date) 
    ep_future_dates = df[df['episode_start_date'] > DATE_LAST_POSSIBLE_FOLLOWUP_DAYS]
    if nrow(ep_future_dates) > 0:
        logger(f"Removing {nrow(ep_future_dates)} episodes whose start date was after the last possible cohort date ({(100*nrow(ep_future_dates))/nrow(df)}%)")
        df = df[df['episode_start_date'] <= DATE_LAST_POSSIBLE_FOLLOWUP_DAYS ]
        logger(f"Left with {nrow(df)} episodes")

    # remove episodes that ended before they started (invalid value, cant infer which date is real...)
    ep_ended_be4_started = df[ df['episode_start_date'] > df['episode_end_date'].fillna(float('inf'))]
    if nrow(ep_ended_be4_started) > 0:
        logger(f"Removing {nrow(ep_ended_be4_started)} episodes whose start date was after their end date ({(100*nrow(ep_ended_be4_started))/nrow(df)}%)")
        df = df[df['episode_start_date'] <= df['episode_end_date'].fillna(float('inf'))] 
        logger(f"Left with {nrow(df)} episodes")
    outcome_df_full = construct_full_outcome_df(pat_ids, df)
    if not skip_follow_up_time_filtering:
        logger(f'Filtering based on outcome first diag date and {VAR_FOLLOW_UP_DATE}')
        df = filter_follow_up_period(df, outcome_df_full, p_key, 'episode_start_date', VAR_FOLLOW_UP_DATE)
        logger(f'Left with {nrow(df)} episodes')
    else:
        logger(f'SKIPPPING Filtering based on outcome first diag date')

    
    df = add_composite_key(df, 'journals_episodes', keep_components=True)
    df = add_composite_key(df, 'episode', dedup=True, keep_components=True) # debug, could remove?
    df = df.drop(columns=["Unnamed: 0", "import_id", "episode_id", "practice_id", "person_id"], axis = 1)
    logger(f"After dedup: episodes df with {df.shape[0]} rows and {df.shape[1]} columns")
    logger(f"Reading episodes from {in_path}...DONE", t0)
    pat_ids_nonempty_eps = list(set(vals(df[p_key])))
    return pat_ids_nonempty_eps, df, outcome_df_full

def read_journals(in_path, nrows=None, pat_ids = [], outcome_df=None, skip_follow_up_time_filtering=False, is_debug=False):
    t0 = logger(f"Reading journals from {in_path}...")
    p_key = MAPPING_COMPOSITE_KEYS['patient']['key_nm']
    df = try_read_pd_df(in_path, nrows=nrows, dtypes=CSV_DTYPES['journals'])
    logger(f"Read {nrow(df)} journals from {str(in_path)} OK")
    df = add_composite_key(df, 'patient', keep_components=True)
    if is_debug:
        logger(f"DEBUG: going to set all journals read to patients available (useful when reading subset of input files)")
        df[p_key] = random.choices(pat_ids, k=nrow(df[p_key]))
    if pat_ids:
        logger(f"Going to select only journals using {len(pat_ids)} pat ids.")
        df = df[df[p_key].isin(pat_ids)]
        logger(f'Left with {nrow(df)} records')

    logger(f'_______ Top 10 EARLIEST journal_datetime = \n{sorted(df["journal_datetime"])[:10]}')
    logger(f'_______ Top 10 LATEST journal_datetime = \n{sorted(df["journal_datetime"], reverse=T)[:10]}')
    logger(f'_______ Top 10 EARLIEST episode_start_dates (from journals) = \n{sorted([x for x in df["episode_start_date"] if type(x) == str])[:10]}')
    logger(f'_______ Top 10 LATEST episode_start_dates (from journals) = \n{sorted([x for x in df["episode_start_date"] if type(x) == str], reverse=T)[:10]}')
    

    df['journal_datetime'] = (pd.to_datetime(df['journal_datetime']) - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days # much faster

    # each journal belongs to 1 patient
    # each patient has a VAR_FOLLOW_UP_DATE 
    #   => this tells us to take only journals whose date is 
    #       :: before (<=) the VAR_FOLLOW_UP_DATE of the patient
    #           AND
    #       :: after (>) the start of the follow-up period (VAR_FOLLOW_UP_DATE - FOLLOW_UP_PERIOD_DAYS)
    if not skip_follow_up_time_filtering:
        df = filter_follow_up_period(df, outcome_df, p_key, 'journal_datetime', VAR_FOLLOW_UP_DATE)
    else:
        logger(f'WARN: SKIP Filtering based on follow-up time!')

    
    logger(f'Left with {nrow(df)} journal entries')

    df.episode_id = df.episode_id.astype('Int32')
    df = add_composite_key(df, 'journals_episodes', keep_components=True)
    df = add_composite_key(df, 'journal', dedup=True, extra_sort_keys=['journal_datetime'], keep_components=True) # debug
    df = df.drop(columns=["import_id", "practice_id", "person_id", "episode_id", "episode_start_date"], axis = 1)
    logger(f"After dedup: df with {df.shape[0]} rows and {df.shape[1]} columns")
    logger(f"Reading journals from {in_path}...DONE", t0)
    pat_ids_nonempty_js = list(set(vals(df[p_key])))
    return pat_ids_nonempty_js, df

def uniq(x, include_None = False, include_nan = False):
    c_res = set(x)
    if not include_None:
        c_res = c_res - set([None])
    if not include_nan:
        c_res = [c for c in c_res if not (isinstance(c,float) and np.isnan(c)) ]
    else:
        nan_present = len([c for c in c_res if not (isinstance(c,float) and np.isnan(c)) ]) > 0
        c_res = [c for c in c_res if not (isinstance(c,float) and np.isnan(c)) ]
        if nan_present:
            c_res.append(np.nan)
    return list(c_res)

def nuniq(x):
    return len(uniq(x))

def vals(pd_series_x, n=None):
    if n is None:
        return pd_series_x.values.tolist()
    return pd_series_x.values.tolist()[0:n]

def dvals(dict):
    return list(dict.values())

def pd_get_time_days(ts):
    if ts is None or isinstance(ts, pd._libs.tslibs.nattype.NaTType) or (isinstance(ts, float) and np.isnan(ts)):
        return None
    return ts.timestamp()

def names(dict):
    return list(dict.keys())    

def calc_age_at_date(age_days, date_days):
    return date_days + age_days

def identity(x):
  return(x)


def try_ohe(keys, vals, col_nm = 'cvar', lookup_triplet = None, special_vals_fn=lambda vs: list(zip(vs, [1]*len(vs)))):
    vals = special_vals_fn(vals)
    #uvals = uniq([x[0] for x in vals], include_nan = T)
    n_cats, lookup_d, rev_lookup_d = lookup_triplet 
    is_binary_values = n_cats <= 3 and (vals == [] or all([x == 0 or x == 1 for x in list(list(zip(*vals))[1])]))

    is_empty = len(lookup_d) == 0
    if is_empty:
        n_cats = 1
    columns= ['id'] + [f'{col_nm}_{i}' for i in range(n_cats)]
    res = []
    for i in range(len(vals)):
        v = vals[i][0]
        idx = 0
        placeholder_val = vals[i][1]
        c_v = v if not(isinstance(v, float) and np.isnan(v)) else np.nan
        if not (is_empty or not c_v in lookup_d):
            idx = lookup_d[c_v]
        else:
            placeholder_val = 0 # (not v in lookup_d) value occurred less than OHE_MIN_N_OCCURRENCES and thus is not present in the lookup-d;
        c_row = [keys[i]] + [0]*idx + [placeholder_val] + [0]*(n_cats-idx-1)
        res.append(c_row)
    res = pd.DataFrame(res, columns = columns)
    cols_to_keep = columns
    if is_binary_values and len(cols_to_keep) > 2 and vals != []:
        ref_cat = cols_to_keep[1]
        cols_to_keep = [cols_to_keep[0]] + cols_to_keep[2:]
        logger(f"Dropping redundant reference category ({ref_cat} => {col_nm} == {list(lookup_d.keys())[0]})")
        res = res[cols_to_keep]
    return res

def try_print_list(vs, print_fn=print):
    l_str =  "\n\t=[ \n\t\t'" +  "',\n\t\t '".join(vs) + "' \n\t]"
    print_fn(l_str)
    return l_str

def try_get_canonical_path(f, subsampled=F):
    return f"pkl/subsampled/{f}" if subsampled else f"pkl/{f}"        

def try_delete_pickle(f, subsampled=F):
    os.remove(try_get_canonical_path(f, subsampled))

def try_pickle_exists(f, subsampled=F):
    return os.path.exists(try_get_canonical_path(f, subsampled))   

def try_read_pickle(f, subsampled=F):
    t0 = logger(f"try_read_pickle for {f}...")
    with open(try_get_canonical_path(f, subsampled), 'rb') as f:
        x = pickle.load(f)
    _ = logger(f"try_read_pickle for {f}... done", t0)
    return x

def try_save_pickle(f, o, test_read = F, subsampled=F):
    t0 = logger(f"try_save_pickle for {f}...") 
    pkl_dir_rp = "pkl" if not subsampled else "pkl/subsampled"
    fp = try_get_canonical_path(f, subsampled)
    try:
        os.remove(fp)
    except OSError:
        pass
    is_file_in_weird_state = len([x for x in os.listdir(pkl_dir_rp) if x == f]) != 0 
    while is_file_in_weird_state:
        f = f"{f}0"
        logger(f"WARNING!: is_file_in_weird_state = True, changing filename to {f}")
        is_file_in_weird_state = len([x for x in os.listdir(pkl_dir_rp) if x == f]) != 0 
        fp = try_get_canonical_path(f, subsampled)
        try:
            os.remove(fp)
        except OSError:
            pass

    with open(fp,'wb') as fn:
        pickle.dump(o, fn)
        t0 = logger(f"pickle.dump for {f}...done")

    if test_read: # because sometimes pickle saving works but fails to load with "invalid key error \x00"
        logger(f"Save done. Test if can load {f}... ")
        try_read_pickle(f, subsampled=subsampled)
        logger(f"Save done. Test if can load {f}... OK ")

    _ = logger(f"try_save_pickle for {f}... done", t0)

def try_read_json(f):
 with open(f"json/{f}", 'r') as json_file:
    data = json.load(json_file)
    return data

def try_round(xs, d=0):
    if type(xs) == list:
        return [round(x, d) for x in xs]
    return round(xs,d)

def try_save_df(f, df):
    t0 = logger(f"try_save_df for {f}...")
    df.to_csv(f"tsv/{f}", sep='\t', encoding='utf-8')
    _ = logger(f"try_save_df for {f}... done", t0)

def try_save_parquet(f, df):
    t0 = logger(f"try_save_parquet for {f}...")
    df.to_parquet(f)
    _ = logger(f"try_save_parquet for {f}... done", t0)


def try_regex(p, s):
    return len(re.findall(p, s)) > 0

def try_match_strings(strings, pattern):
    return [c for c in strings if pattern in c]

def try_regex_multi(strings, pattern):
    return try_match_strings(strings, pattern)

def try_get_function_params(fn):
    return fn.__code__.co_varnames

def split_pat_prac_id(id_str):
    pat_prac_id = [int(x) for x in id_str.split('|')]
    return {"pat_id": pat_prac_id[0], "prac_id": pat_prac_id[1]}

def convert_pat_id_str2float(id, n_padding = 9):
    pat_prac_id = split_pat_prac_id(id)
    # prac_id_len = round_up(np.log10(pat_prac_id[1]))
    assert len(str(pat_prac_id["prac_id"])) <= n_padding
    return pat_prac_id["pat_id"] + pat_prac_id["prac_id"]/(10**n_padding)

def convert_pat_id_float2str(id, n_padding = 9):
    pat_id = int(id)
    prac_id = round((id - pat_id) * (10**n_padding) )
    return f'{pat_id}|{prac_id}'

def remove_csr_columns(cols_to_remove, data, data_cns):
    cols_to_drop_idxs = [i for i,_ in cols_to_remove]
    n_cols = data.shape[1]
    data = data[:, [x for x in range(n_cols) if x not in cols_to_drop_idxs]] 
    removed_col_nms = [v for _,v in cols_to_remove]
    data_cns = [x for x in data_cns if x not in removed_col_nms]
    # data_cns = [(i,v) for i,v in enumerate(data_cns)]
    return data, data_cns

# Perform reverse-lookup of icpc ohe number to code_lookup_dicts
# Returns actual icpc code
def one_hot_decode_varname(col_str, code_lookup_dicts):
    # n_underscores = len(col_str.split("_"))
    is_patient_level = not try_regex("^\d+_\d+_", col_str)
    is_one_hot_encoded = try_regex("^.*_\d+$", col_str) #and not "postal_code" in col_str

    # for non-patient-level vars:   remove time period (0/1), and aggr func code (2), and ohe code (-1)
    col_str_base_var =  "_".join(col_str.split("_")[3:]) if not is_patient_level  else col_str

    if is_one_hot_encoded:
        col_str_base_var = "_".join(col_str_base_var.split("_")[:-1])

    
    col_str_ohe_code =  "_".join(col_str.split("_")[-1]) if is_one_hot_encoded else None

    translate_mapper = {
        'ep_dur' : 'episode_duration',
        'icpc_ep' : 'icpc_episode',
        'e_att' : 'episode_attention',
        'e_pr' : 'episode_problem',
        'ep_sts' : 'episode_status',
        'ctyp' : 'contact_type',
        'icpc_s' : 'icpc_s',
        'icpc_o' : 'icpc_o',
        'icpc_e' : 'icpc_e',
        'icpc_p' : 'icpc_p',
        'icpc_x' : 'icpc_x',
        'icpc_j' : 'icpc_journal',       
    }
    col_str_base_var = col_str_base_var if col_str_base_var not in translate_mapper else translate_mapper[col_str_base_var]
    if is_one_hot_encoded and col_str_base_var in code_lookup_dicts['rev-lookup']:
        return str(code_lookup_dicts['rev-lookup'][col_str_base_var][int(col_str_ohe_code)]).strip()
    return col_str_base_var.strip()


def try_summarize_vals(xs):
    n = len(xs)
    min_v = min(xs)
    max_v = max(xs)
    q1 = np.quantile(xs, 0.25)
    q2 = np.quantile(xs, 0.5)
    q3 = np.quantile(xs, 0.75)
    std = np.std(xs)
    mean_v = np.mean(xs)
    return n, min_v, q1, q2, q3, max_v, std, mean_v

def try_log_vals_summary(xs, logger):
    n, min_v, q1, q2, q3, max_v, std, mean_v = try_summarize_vals(xs)
    logger(f"n = {n}")
    logger(f"range = {min_v:.3e} : {max_v:.3e}")
    logger(f"median (IQR)= {q2:.3e} ({q1:.3e} : {q3:.3e})")
    logger(f"mean (SD) = {mean_v:.3e} ({std:.3e})")


def is_cluster_interesting(incidence, mass, npos, base_incidence):
    # 'high-risk cluster' criteria per manuscript: size >=0.5% of cohort and
    # HF prevalence >=3x cohort baseline. (Manuscript also allows <=3x lower
    # prevalence as a symmetric 'low-risk' criterion, not implemented here;
    # no such cluster occurred in the reported results.)
    return mass >= 0.005 and incidence >= base_incidence*3


def try_multiindex(xs, v):
    return [ i for i,a in enumerate(xs) if a == v]

def get_clusters_inc_supp(Y, Y_pred, verbose=F):
    '''
    Y = binary outcome of interest
    Y_pred = cluster to which a record belongs
    '''
    if type(Y) == pd.DataFrame:
        Y = Y.iloc[:, 0].values
    n = len(Y)
    total_npos = np.sum(Y)
    total_nneg = n - total_npos
    cluster_sizes = pd.Series(Y_pred).value_counts().sort_index().to_numpy()

    ct = pd.crosstab(Y,  Y_pred)
    n_clusters = dim(ct)[1]

    cluster_npos = ct.iloc[1,:].to_numpy()
    cluster_nneg = ct.iloc[0,:].to_numpy()

    cluster_masses = cluster_sizes / n
    cluster_1masses = cluster_npos / total_npos
    cluster_0masses = (cluster_nneg / total_nneg) + 1e-16

    inci = cluster_npos/cluster_sizes
    supp = cluster_1masses / cluster_0masses # odds of being positive in this cluster compared to general population 
    
    return inci.tolist(), supp.tolist(), cluster_npos.tolist(), cluster_masses.tolist()

def oset(vs):
    return sorted(set(vs), key=vs.index)

def try_select_best_metric(xs):
    return (try_multiindex(xs, max(xs)), max(xs))

def calc_mass_per_cluster(Y_pred, verbose=F, d_round = 4):
    return [round(x/len(Y_pred), d_round) for x in vals(pd.Series(Y_pred).value_counts().sort_index())]

def calc_support_per_cluster(Y_pred, y):
    return 0

# Severities = lo/med/hi (based on quantiles of icpc code scores)
# Deprecated - switch to using categories instead of severity codes
def get_icpc_score_and_code_severities(in_file_code_scores):
    icpc_scores = try_read_json(in_file_code_scores)
    icpc_letters = sorted(uniq([x[0] for x in icpc_scores.keys()]))
    code_sev_mapping = {}
    for i_l in icpc_letters:
        c_codes = [(k,v) for k,v in icpc_scores.items() if k[0] == i_l]
        c_scores = [v for _,v in c_codes]
        lower_mid_thresh = np.quantile(c_scores, 0.33)
        mid_up_thresh = np.quantile(c_scores, 0.66)
        code_sev_mapping[f"{i_l}_lo"] = [k for k,v in c_codes if v <= lower_mid_thresh]
        code_sev_mapping[f"{i_l}_med"] = [k for k,v in c_codes if v > lower_mid_thresh and v <= mid_up_thresh]
        code_sev_mapping[f"{i_l}_hi"] = [k for k,v in c_codes if v > mid_up_thresh]

    return icpc_scores, code_sev_mapping

def get_icpc_cats(in_files_code_cats):
    icpc_cats  = {}
    code_cat_mapping = {} # k = cat , v = [icpc codes ]
    for i,in_file_code_cats in enumerate(in_files_code_cats):
        c_icpc_cats = try_read_json(in_file_code_cats) #  k = icpc code, v =  cat 
        c_icpc_cats = {k : f"cats{i}_{v}" for k,v in  c_icpc_cats.items()} # ensure cats have unique names
        icpc_cats.update(c_icpc_cats)

    for icpc_code, cat in icpc_cats.items():
        if cat not in code_cat_mapping:
            code_cat_mapping[cat] = [icpc_code] 
        else:
            code_cat_mapping[cat] += [icpc_code] 
    return icpc_cats, code_cat_mapping

# def spma3x_to_sptensor(X):
#     coo = X.tocoo()
#     indx = np.column_stack((coo.row, coo.col))
#     return tf.SparseTensor(indx, coo.data, coo.shape)


def filter_strings_regex(strs, regex, only_idxs = F, only_strs = F):
    """
    Args: 
        strs: list of strings
        regex: regex to match against
        only_idxs: return only indexes
        only_strs: return only matched strings
    Returns:
        list of index/column_name pairs, where index is the position of column_name in the inputted strs
    """
    if only_strs and only_idxs:
        logger("called filter_strings_regex with invalid paramters")
        return None
    if only_idxs:
        return [i for i,v in enumerate(strs) if try_regex(regex,v)]
    if only_strs:
        return [v for i,v in enumerate(strs) if try_regex(regex,v)]
    return [(i,v) for i,v in enumerate(strs) if try_regex(regex,v)]


def get_embedding_model_path(embedding_model):
    """
    doc2vec,sbert and custom models are not stored locally, hence they should not have an embedding model path
    """
    emb_m_path = None
    model_metadata = EMBEDDING_MODEL_METADATA[embedding_model]
    if embedding_model != 'doc2vec' and not model_metadata['is_custom'] and not model_metadata['is_sbert']:
        emb_m_path = f"/app/pretrained/{embedding_model}"

    return emb_m_path

def determine_n_records_per_split(n_records, n_splits):
    recs_per_split = round_down(n_records / n_splits)
    recs_per_split = [recs_per_split for _ in range(n_splits)]
    remainder_rows = n_records % n_splits
    recs_per_split = [x+1 if i<remainder_rows else x for i,x in enumerate(recs_per_split)]
    # assert sum(recs_per_split) == n_records
    return recs_per_split

def try_run_multiprocess(items, worker_fn, n_processes = 2):
    if n_processes > 1:
        pool = ProcessPool(n_processes)
        res = None
        try:
            res = pool.map(worker_fn, items)
        except Exception:
            print(traceback.format_exc())
            raise("oh geez!")
        return res
    if n_processes == 1:
        return [worker_fn(items[0])]

def module_to_dict(m):
    c = {}
    for s in dir(m):
        c[s] = getattr(m,s)
    return c

transform_cosine_dist = lambda x: (x+1)/2

def expand_hp_params(hp_params):
    expanded_params = [{k:v for k,v in zip(hp_params.keys(),p)} for p in product(*hp_params.values())]
    return expanded_params

def get_time_quantized_period_names(time_bins):
    bin_start, bin_end = None, None
    period_prefixes = []
    for bin_start, bin_end in time_bins: 
        period_nm = f'{int(bin_start/30)}_{int(bin_end/30)}'
        period_prefixes += [period_nm]
    return period_prefixes


def forwards_var_selection(x, y, metric='aic', n_inits=1):
    get_aic = lambda vrs: sm.OLS(y, x[vrs]).fit().aic
    get_bic = lambda vrs: sm.OLS(y, x[vrs]).fit().bic
    metrics_d = { 'aic': get_aic, 'bic': get_bic}
    get_metric = metrics_d[metric]
    all_vars = cns(x)
    c_vars = cns(x)[:1]
    c_metric = get_metric(c_vars)
    best_metric = c_metric
    best_vars = c_vars
    best_delta = 0
    while 1 == 1:
        next_var_to_add = None
        logger(f'forwards {metric}, with {len(best_vars)} vars')
        c_vars = best_vars
        best_delta = 0
        for cvar in try_sd(all_vars, c_vars): # cvar= c_vars[0]
            vars = c_vars + [cvar]
            c_metric = get_metric(vars)
            metric_delta = best_metric - c_metric

            if metric_delta > best_delta:
                best_delta = metric_delta
                best_metric = c_metric
                next_var_to_add = cvar
                best_vars = vars

        if next_var_to_add is None:
            logger(f'complete forwards {metric}')
            logger(f'selected {len(best_vars)} vars:')
            logger(f'{best_vars}')
            break

        logger(f"adding var {next_var_to_add} (best {metric} {best_metric:0.2f}, {metric} delta {best_delta:0.2f})")
    return best_vars


def forwards_AIC(x, y):
    return forwards_var_selection(x,y,'aic')

def forwards_BIC(x, y):
    return forwards_var_selection(x,y,'bic')

def backwards_var_selection(x, y, metric='aic'):
    get_aic = lambda vrs: sm.OLS(y, x[vrs]).fit().aic
    get_bic = lambda vrs: sm.OLS(y, x[vrs]).fit().bic
    metrics_d = { 'aic': get_aic, 'bic': get_bic}
    get_metric = metrics_d[metric]

    c_vars = cns(x)
    c_metric = get_metric(c_vars)
    best_metric = c_metric
    best_vars = c_vars
    best_delta = 0
    while 1 == 1:
        next_var_to_remove = None
        logger(f'backwards {metric}, left with {len(best_vars)} vars')
        c_vars = best_vars
        best_delta = 0
        for cvar in c_vars: # cvar= c_vars[0]
            vars = [cv for cv in c_vars if cv != cvar]
            c_metric = get_metric(vars)
            metric_delta = best_metric - c_metric

            if metric_delta > best_delta:
                best_delta = metric_delta
                best_metric = c_metric
                next_var_to_remove = cvar
                best_vars = vars

        if next_var_to_remove is None:
            logger(f'complete backwards {metric}')
            logger(f'selected {len(best_vars)} vars:')
            logger(f'{best_vars}')
            break

        logger(f"removing var {next_var_to_remove} (best {metric} {best_metric:0.2f}, aic delta {best_delta:0.2f})")
    return best_vars


def backwards_AIC(x, y):
    return backwards_var_selection(x,y,'aic')

def backwards_BIC(x, y):
    return backwards_var_selection(x,y,'bic')

def eval_model_auc(model, X, y):
    from sklearn.metrics import roc_auc_score
    pred_probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, pred_probs)
    return auc



def fit_pred_model_test(X, Y, vars_to_use, model_obj, cv_nfolds=4, eval_model_fn = eval_model_auc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (make_scorer, roc_auc_score, accuracy_score, confusion_matrix, classification_report, rand_score, silhouette_score)
    from sklearn import set_config
    from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score)

    set_config(enable_metadata_routing=T)
    fit_cns = cns(X)
    stop_after_n_folds = 10
    if cv_nfolds == 1:
        stop_after_n_folds = 1
        cv_nfolds = 4
    cv = StratifiedKFold(n_splits=cv_nfolds, shuffle=T)
    scorer = make_scorer(roc_auc_score, needs_proba=T).set_score_request(sample_weight=F)

    c_X = X[ vars_to_use ].to_numpy()
    if len(vars_to_use) == 1:
        c_X = c_X.reshape(-1,1)
    c_y = np.array(Y)
    train_aucs = []
    val_aucs = []
    fold_i = 1
    def assign_weights(ys):
        ys = [int(y) for y in ys]
        npos = sum(ys)
        n = len(ys)
        pos_w = (n-npos)/n
        neg_w = npos/n
        weights = [neg_w, pos_w]
        return [weights[y] for y in ys]

    
    logger(f'Using {len(vars_to_use)} features, namely:: \n{vars_to_use}')
    for train_idx, val_idx in cv.split(c_X, c_y):
        if fold_i > stop_after_n_folds:
            break
        X_train, X_val = c_X[train_idx], c_X[val_idx]
        y_train, y_val = c_y[train_idx], c_y[val_idx]
        weights_train = assign_weights(y_train)
        model = model_obj['init']()
        with_sample_weight = model_obj['sample_weight']

        best_params = None
        is_gs_needed = any([len(x) > 1 for x in model_obj['grid_params'].values()])
        if is_gs_needed:
            gs = GridSearchCV(estimator = model, param_grid = model_obj['grid_params'], scoring=scorer, cv = cv_nfolds)
            if with_sample_weight:
                gs.fit(X_train, y_train,  sample_weight=weights_train)
            else:
                gs.fit(X_train, y_train)
            best_params = gs.best_estimator_.get_params()
        else:
            best_params = {k:v[0] for k,v in model_obj['grid_params'].items() }
        
        logger(f"best params = {[ (k,best_params[k]) for k in model_obj['grid_params'].keys() ]}")
        if 'verbose' in best_params:
            best_params['verbose'] = T
        model = model_obj['init'](best_params)
        if with_sample_weight:
            model.fit(X_train, y_train,  sample_weight=weights_train)
        else:
            model.fit(X_train, y_train)
        train_auc = eval_model_fn(model, X_train, y_train)
        val_auc = eval_model_fn(model, X_val, y_val)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        fold_i += 1
    logger(f"mean train auc: {np.around(np.mean(train_aucs), 3)} ")
    logger(f"mean val auc: {np.around(np.mean(val_aucs), 3)} ")
    logger("Done fitting")
    return val_aucs 
        
def run_PM_test(
    X, 
    Y, 
    vars=[],
    model_type='logreg',
    cv_nfolds = 1,
    hidden_layer_sizes = [(100,50)], tol=[1e-2], alpha=[0.1], max_iter=[200]
    ):
    fit_cns = vars
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (make_scorer, roc_auc_score, accuracy_score, confusion_matrix, classification_report, rand_score, silhouette_score)
    from sklearn import set_config
    from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score)

    set_config(enable_metadata_routing=T)

    models = { "logreg": { 
                    "init" : lambda kwargs={}: LogisticRegression(**kwargs),
                    "grid_params": { 'solver' : ['saga'], 'penalty' : ['l1'], 'C': [1], 'max_iter' : [100]},
                    "sample_weight"  : F,
                },
                "decision_tree": {
                    "init": lambda kwargs={}: DecisionTreeClassifier(**kwargs),
                    "grid_params": { 'max_depth' : [5], 'min_samples_split' : [1000],  'random_state': [3940832094] } ,
                    "sample_weight"  : F,
                },
                "mlp": {
                    "init": lambda kwargs={}: MLPClassifier(**kwargs),
                    "grid_params": {
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'max_iter': max_iter, 
                        'alpha': alpha,
                        'tol' : tol,
                        'batch_size': [1000],
                        'learning_rate_init': [0.0001],
                        'learning_rate' : ['adaptive'],
                        'verbose': [T]
                        }, 
                        "sample_weight"  : F,
                }
    }

    model_obj = models[model_type]
    vars_to_use = [c for c in fit_cns if c != 'id'] 
    logger(f"Fitting {model_type} PM with {len(vars_to_use)} vars")
    return fit_pred_model_test(X, Y, vars_to_use, model_obj, cv_nfolds=cv_nfolds)




def get_HF_adj_pos_pat_ids(lukas_adj_df, return_all=F):
    """
    
    ep_start_dts - start date of the first HF diag episode, if no HF and return_all=T then start date of last episode in system
    """
    #lukas_adj_df = lukas_adj_df.loc[lukas_adj_df['id'].isin(p_ids)].reset_index(drop=T)
    lukas_adj_df['episode_start_date'] = (pd.to_datetime(lukas_adj_df['episode_start_date'], format='%m/%d/%Y') - DATE_ARBITRARY_OFFSET_TIMESTAMP).dt.days # much faster
    ep_start_dts = []
    hf_diag = []
    # note  some patients ids are present multiple times in lukas_adj_df (ones that had multiple HF episodes)
    df_outcome_positive =  lukas_adj_df[lukas_adj_df['NO_HF'].isna()]
    df_outcome_positive = df_outcome_positive.reset_index(drop=T)
    
    idx = df_outcome_positive.groupby('id')['episode_start_date'].idxmin()
    df_outcome_positive = df_outcome_positive.loc[idx].reset_index(drop=T) # keep only first recorded episode of outcome    
    df_outcome_positive = df_outcome_positive.sort_values('episode_start_date', ascending=F)
    df_outcome_positive = df_outcome_positive.reset_index(drop=T)
    ids_of_interest_pos = vals(df_outcome_positive["id"])
    ep_start_dts_pos = vals(df_outcome_positive["episode_start_date"])
    hf_diag_pos = [T] * len(ids_of_interest_pos)


    ids_of_interest = []
    ids_of_interest_neg = []
    ep_start_dts_neg = []
    hf_diag_neg = []


    if return_all:
        df_outcome_negative =  lukas_adj_df[~lukas_adj_df['id'].isin(ids_of_interest_pos)]
        df_outcome_negative = df_outcome_negative.reset_index(drop=T)
        idx = df_outcome_negative.groupby('id')['episode_start_date'].idxmax()
        df_outcome_negative = df_outcome_negative.loc[idx].reset_index(drop=T) # keep only last recorded episode
        df_outcome_negative = df_outcome_negative.sort_values('episode_start_date', ascending=F)
        df_outcome_negative = df_outcome_negative.reset_index(drop=T)
        ids_of_interest_neg = vals(df_outcome_negative["id"])
        ep_start_dts_neg = vals(df_outcome_negative["episode_start_date"])
        hf_diag_neg = [F] * len(ids_of_interest_neg)


    ids_of_interest = ids_of_interest_pos + ids_of_interest_neg
    ep_start_dts = ep_start_dts_pos + ep_start_dts_neg
    hf_diag = hf_diag_pos + hf_diag_neg

    if return_all:
        return ids_of_interest, ep_start_dts, hf_diag

    return ids_of_interest, ep_start_dts

def lol_to_l(lol):
    return [ii for i in lol for ii in i]

last = lambda xs: xs[::-1][0]

def __init_pats_dict_flwp_filtered(x):
    # i.e., filtered = rigth-censored
    import analyse_results_util as ar_util
    from dim_reduce_utils import group_atcs
    meds_df = [ {'p_id' : pid, 'atc_code' : med['atc_code'], 'medication_datetime' : med['medication_datetime'] } for pid,p in x.items() for med in p['Medications'] ]
    logger(f"Parsed meds df, dedup starting .. N = {len(meds_df)}")
    meds_df = pd.DataFrame(meds_df)
    meds_df = group_atcs(meds_df, dedup=T)

    c_groups = meds_df.groupby('p_id')
    def flwp_filter_meds_df_to_list(p, df_rows):
        meds = [{'atc_code' :i[2], 'medication_datetime': i[3] } for i in list(df_rows.to_records())]
        meds = [m for m in meds if ar_util.is_med_dt_during_flwp(p, m)]
        return meds
    all_HFpos_ids = [pid for pid,p in x.items() if not pd.isnull(p['t_HF']) ]
    for k in all_HFpos_ids:
        x[k]['follow_up_LAST'] = ar_util.apply_flwp_cens_window(k, x[k]['follow_up_LAST'], all_HFpos_ids)

    cnt = 0
    for c_group, c_rows in c_groups: #c_group, c_rows = (list(c_groups.groups.keys())[0], c_groups.get_group(list(c_groups.groups.keys())[0]))
        cnt+=1
        if cnt % 10000 == 0:
            logger(f"Medications done {cnt}/{len(c_groups)}...")
        x[c_group]['Medications'] = flwp_filter_meds_df_to_list(x[c_group], c_rows)
    cnt = 0
    for p_id in x.keys():
        cnt+=1
        if cnt % 10000 == 0:
            logger(f"Episode/Journals done {cnt}/{len(x)}...")
        x[p_id]['Episodes'] = ar_util.flwp_filter_eps(x[p_id])
        
        c_eps = x[p_id]['Episodes']
        for i, ep in enumerate(c_eps):
            c_eps[i]['JOURNALS'] = ar_util.flwp_filter_js(x[p_id], ep['JOURNALS'])
        x[p_id]['Episodes'] = c_eps
    return x


def __init_pats_dict_time_unbiased(x, MIN_N_CONSULTS, MIN_FLWP_WIN_DAYS, MATCH_DATE_THRESHOLD, IS_DEBUG):
    # NOTE: this removes the follow-up start time bias between cases and controls, but introduces another type of bias by removing many records..
    # decision is to keep this as a sensititivy analysis but not the main analysis..
    # i.e., 
    # 1. require at least 3 consults per record
    # 2. require a minimum observation window (6 months)
    # 3. try to balance out the starting dates of case and controls

    # ******************  ******************  ******************  ******************  ******************  ******************
    # x['109867|30010']['adj_HF_diag'] = T # DEBUG!!!
    case_pids = [pid for pid,p in x.items() if p['adj_HF_diag']]
    xx = {pid : {} for pid in x.keys()} # new data 
    j_level_op = lambda op, agg = identity: lambda pat: agg(lol_to_l([op(e['JOURNALS']) for e in pat['Episodes']])) 
    lsum = lambda xs: [sum(xs)]
    lwrap = lambda fn: lambda xs : [fn(xs)]
    # for each pat -> calc:
        #   number of consults
    cnt_n_js = lambda pat : j_level_op(lwrap(len), sum)(pat)
        #   dates of each consult [including date of first consult after cohort start period]
    get_j_dts = lambda pat: sorted(j_level_op(lambda js: [j['journal_datetime'] for j in js])(pat))

        #   num of days between first and last consult
    ndays_ft_lt_j = lambda j_dts: j_dts[::-1][0] - j_dts[0] if len(j_dts) > 1 else -1

    filter_j_dts_ep = lambda js, dt_min: [j for j in js if j['journal_datetime'] >= dt_min]
    filter_j_dts = lambda pat, dt_min: [{k : v if k != 'JOURNALS' else filter_j_dts_ep(v, dt_min) for k,v in e.items()} for e in pat['Episodes']]
    filter_m_dts = lambda pat, dt_min: [m for m in pat['Medications'] if m['medication_datetime'] >= dt_min]
    filter_e_dts = lambda pat, dt_min: [e for e in pat['Episodes'] if e['episode_start_date'] >= dt_min]

    pids_to_remove = []
    strata = {}
    stratum_dates = {}

    # (3).
    #   for each HF+ pat -> do:
    #       stratify on sex and birth year band (35-45, 45-55, 55-65, 65-75, 75+)
    def get_stratum(pat):
        res = ""
        sex_mapping = {"M" : "M", "V" : "F", "N": "M", "O": "V"}[pat["sex"]]
        age_mapping = "<1945"
        if pat["age_days"] > round(DAYS_IN_YEAR*25):
            age_mapping = "1945-1955"
        if pat["age_days"] > round(DAYS_IN_YEAR*15):
            age_mapping = "1955-1965"
        if pat["age_days"] > round(DAYS_IN_YEAR*5):
            age_mapping = "1965-1975"
        if pat["age_days"] > round(DAYS_IN_YEAR*5):
            age_mapping = "1975-1985"            
        if pat["age_days"] > round(DAYS_IN_YEAR*15):
            age_mapping = ">1985"            

        res = f"{sex_mapping}_{age_mapping}"
        return res

    for pid, pat in x.items(): # 'JOURNALS'
        xx[pid]["n_consult"] = cnt_n_js(pat)
        # filter on (1).
        if xx[pid]["n_consult"] < MIN_N_CONSULTS:
            pids_to_remove+= [pid]
            continue

        xx[pid]["consult_dates"] = get_j_dts(pat) # todo: check if sorted, should be...
        xx[pid]["days_flwp"] = ndays_ft_lt_j(xx[pid]["consult_dates"])
        # filter on (2).
        if xx[pid]["days_flwp"] < MIN_FLWP_WIN_DAYS : # six months min
            pids_to_remove+= [pid]
            continue

        xx[pid]["is_case"] = pid in case_pids
        #       determine stratum age/sex
        stratum = get_stratum(x[pid])
        xx[pid]["stratum"] = stratum
        strata[stratum] = (strata[stratum] + 1) if stratum in strata else 1
        if xx[pid]["is_case"]:
            stratum_dates[stratum] = stratum_dates[stratum] + [xx[pid]["consult_dates"][0]] if stratum in stratum_dates else [xx[pid]["consult_dates"][0]]
    
    pids_to_remove = sorted(list(set(pids_to_remove)))
    logger(f"Applying filters for require at least {MIN_N_CONSULTS} consults per record")
    logger(f"Applying filters for require a minimum observation window ({MIN_FLWP_WIN_DAYS/30} month(s))")

    removed_cases = try_si(pids_to_remove, case_pids)

    logger(f"Removing {len(pids_to_remove)} pats, (#cases = {len(removed_cases)}).")
    xx = {pid: xx[pid] for pid in try_sd(xx.keys(), pids_to_remove)}
    logger(f"Left with {len(xx)} patients")


    case_pids = [pid for pid in xx.keys() if x[pid]['adj_HF_diag']]

    n_controls = len(xx) - len(case_pids)

    #   case_index_dates = x.filter(HF+).select(date of first consult)
    case_index_dates = [xx[pid]['consult_dates'][0] for pid in case_pids]

    #       per stratum -> calc:
    for stratum, s_count in strata.items():
        logger(f"stratum {stratum} (n = {s_count})")
        #           create sample set from start dates
    unmatched_pids = []
    #   for each HF- pat -> do:
    if IS_DEBUG:
        logger("DEBUG setting random stratum date (avoid missing stratum dates)")
        strat_vals = lol_to_l(list(stratum_dates.values()))
        for k in strata.keys():
            stratum_dates[k] = random.sample(strat_vals, 1)[0]
            
            

    for pid in try_sd(xx.keys(), case_pids):
        pat = xx[pid]
        stratum = pat["stratum"]
        cdf_dates = stratum_dates[stratum]
        #       sample date from stratum
        sampled_days = np.random.choice(cdf_dates, size = 50, replace=T)
        control_cons_dates = pat["consult_dates"]
        matched_index_dt = None
        #       check if compatible with pat, yes => set as new start date; no => sample date again (try 50 times)
        # find earliest compatible date (compatible ~ within 3 months delta)
        for s_d in sampled_days:
            candidate_dates = [d for d in control_cons_dates if abs(d - s_d) <= MATCH_DATE_THRESHOLD]
            if candidate_dates == []:
                continue
            candidate_date = candidate_dates[0]
            # check if pat has 
            # enough consults 
            cand_cons = [ cc  for cc in control_cons_dates if cc >= candidate_date ]
            if len(cand_cons) < MIN_N_CONSULTS:
                continue
            # enough flwp window (6 months)
            cand_flwp_dur = cand_cons[::-1][0] - cand_cons[0]
            if cand_flwp_dur < MIN_FLWP_WIN_DAYS:
                continue

            matched_index_dt = s_d
            break 


        #       if no match after 50 samples => exclude patient
        if matched_index_dt is None:
            unmatched_pids += [pid]
        else:
            # update consults according to new start date 
            x[pid]['Episodes'] = filter_j_dts(x[pid], matched_index_dt)
            # x[pid]['Medications'] = filter_m_dts(x[pid], matched_index_dt) # medication_datetime            
            xx[pid]["n_consult"] = cnt_n_js(x[pid])
            xx[pid]["consult_dates"] = get_j_dts(x[pid]) # todo: check if sorted, should be...
            xx[pid]["days_flwp"] = ndays_ft_lt_j(xx[pid]["consult_dates"])

            # check! if are there any episodes that start after the last consult date? Yes.... todo see if we need to remove later?
            #assert xx[pid]["consult_dates"] == [] or xx[pid]["consult_dates"][::-1][0] >= max([e['episode_start_date'] for e in x[pid]['Episodes']])


    # ******************  ******************  ******************  ******************  ******************  ******************
    
    res = { pid: x[pid] for pid in xx.keys() }
    for pid in xx.keys():
        res[pid]["n_consult"] = xx[pid]['n_consult']
        res[pid]["days_flwp"] = xx[pid]['days_flwp']

    return res

# Aliases

def tty(*args):
    return try_table(*args)

def llz(lols, idx=0):
    return list(list(zip(*lols))[idx])

def try_reduce(lols):
    return llz(lols)

def tet(vs, idxs = []):
    if idxs == []:
        idxs = range(len(vs))
    return list(zip(idxs,vs))

def tlr(v):
    return list(range(v))


def try_expand(l):
    return [[li] for li in l]


# set operations


def try_su(vs1, vs2):
    return list(set.union(set(vs1), set(vs2)))

def try_si(vs1, vs2):
    return list(set.intersection(set(vs1), set(vs2)))

def try_sd(vs1, vs2):
    return list(set.difference(set(vs1), set(vs2)))

def try_sdui(vs1, vs2): # diff between union and intersect, i.e., which vars are either only in vs1 or vs2, but not in both
    return try_sd(try_su(vs1,vs2), try_si(vs1,vs2))


# filenaming
def full_filename_ext(fname="temp", suffix="", ext="pkl"):
    return f"{fname}{suffix}.{ext}"
# plot utils

def try_plot_hist_vals(vals, title="no title", outpath = "plots/tmp/", outfile='test', subsampled=F):
    y = try_table(vals).tolist()
    x = list(try_table(vals).to_dict().keys())
    plt.figure(figsize=(10,6))
    plt.hist(vals, bins=100, color='skyblue', edgecolor='black', log=T)
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('frequency (log)')
    if subsampled:
        outpath = f"{outpath}subsampled/"
    plt.savefig(f'{outpath}{outfile}.png')