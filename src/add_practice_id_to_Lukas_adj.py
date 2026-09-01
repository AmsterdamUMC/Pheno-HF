print(''' WHAT THIS SCRIPT DOES:
        Reads inputs from verification_icpc_[AHA/ANH].csv (missing practice id by default)
        Reads patients_[AHA/ANH].parquet (has practice id)
        Adds practice_id to each person_id from the former.
        Saves new adjudicated file with AHA/ANH data merged into outfile.
''')
# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
logger = get_default_logger_fn(__file__)
IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
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
# Boilerplate end

outfile_infix = f"{subsampled_str}"

outfile = f"all_adjudicated_HF_Lukas{outfile_infix}.pkl"

db_suffixes = ["AHA", "ANH"]

in_file_patients_d = {
    "AHA" : pqt_dir/"patients_AHA.parquet",
    "ANH" : pqt_dir/"patients_ANH.parquet"
}
in_file_adjudicated_d = {
    "AHA" : data_dir/"cohort/verification_icpc_AHA.csv",
    "ANH" : data_dir/"cohort/verification_icpc_ANH.csv"
}
all_adj_df = None
# Note: adjudication was done on episode level.. so each patient could have multiple adjudications
for db_suffix in db_suffixes: # db_suffix = db_suffixes[0]
    logger(f"Running for db_suffix = {db_suffix}")
    adj_df = try_read_pd_df(in_file_adjudicated_d[db_suffix])
    # ep key = ep_start_date+icpc_episode+episode_desc ;( 
    cols_to_keep = ['person_id', 'episode_start_date', 'icpc_episode', 'episode_description', 'NO_HF', 'text_HF', 'icpc_HF']
    adj_df = adj_df.drop(columns=[c for c in cns(adj_df) if c not in cols_to_keep], axis=1)
    #adj_df = adj_df.drop_duplicates(ignore_index=T) , should have no effect if key is good
    p_df = try_read_pd_df(in_file_patients_d[db_suffix])
    cols_to_keep = ['person_id', 'practice_id']
    p_df = p_df.drop(columns=[c for c in cns(p_df) if c not in cols_to_keep], axis=1)
    p_df = p_df.drop_duplicates(ignore_index=T)
    prac_ids = vals(p_df['practice_id'])
    p_ids = vals(p_df['person_id'])
    adj_df['practice_id'] = -1
    adj_df['id'] = ''
    id_idx  =  -1
    practice_id_idx = -2
    n_pids = len(p_ids)
    for i in range(n_pids): # i, pp_id = (1, (p_ids[1], prac_ids[1]))        
        if not i % 10000:
            logger(f"{(i*100)//n_pids}% complete")
        person_id, practice_id = p_ids[i], prac_ids[i]
        pat_prac_id = f"{person_id}|{practice_id}"
        matched_adj_rows = adj_df[adj_df['person_id'] == person_id]
        if not nrow(matched_adj_rows):
            continue
        adj_ridxs = adj_df[adj_df['person_id'] == person_id].index
        adj_df.iloc[adj_ridxs,practice_id_idx] = practice_id
        adj_df.iloc[adj_ridxs,id_idx] = pat_prac_id
    assert not nrow(adj_df[adj_df['id'] == ''])
    all_adj_df = pd.concat([all_adj_df, adj_df]) # assert len(try_si(all_adj_df['id'], adj_df['id'])) == 0


# assume that recorsd not present in this df are counted as negative/negative
# i.e., no HF from ICPC code, no HF from ep desc
try_save_pickle(outfile, all_adj_df)

logger("DONE")
print("DONE")

    

    
            
