print(
    '''
# WHAT THIS SCRIPT DOES:
# 1. For a given topic from Top2Vec, finds the patient ids of the top N documents that are closest to said topic
'''
)
# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
check_if_debugging(IS_DEBUG)
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
from try_utils import *
from constants import *
import numpy as np
import random
RANDOM_SEED = 42
logger = get_default_logger_fn(__file__) # init logger
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# Boilerplate end
#'42_48_text_edsoepx_intrmtn_top2VecModels_d2v_yconc_modifs_dm0_non_quantized_ntr.pkl'
from MedRoBERTaNL_wrapper import RobertaForMaskedLM_embedding_model

# What we need to load
# the Top2Vec model(s)
# inputs used for said models 
embedding_model = "RobertaForMaskedLM_embedding_model"
emb_m_shortname = EMBEDDING_MODEL_METADATA[embedding_model]['short_name']
is_older_model = T 

concat_modifiers = embedding_model == 'doc2vec'

subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
concat_modifiers_str = "yconc_modifs" if concat_modifiers else "nconc_modifs"
# this param was not defined in older models (default was 0, d2v specific, but used in file naming convention)
dm_str = f"dm{d2v_dm}" if not is_older_model else  "" 
# this param was not defined in older models (default was 0, used in file naming convention)
infix_qt = "" if is_older_model else  'non_quantized' if not SHOULD_SOEP_SEPERATE_TEXTS else 'quantized' 

opt_sep = '_' if not is_older_model else ''
outfile_infix = f"{subsampled_str}_{emb_m_shortname}_{concat_modifiers_str}{MM_INFIX}{opt_sep}{dm_str}{opt_sep}{infix_qt}"

# pats_dict_merged_text_emb_d2v_yconc_modifs.pkl
# pats_dict_merged_text_emb_d2v_yconc_modifs_msrs_meds.pkl
# pats_dict_merged_text_emb_d2v_yconc_modifs_dm0_non_quantized.pkl

# pats_dict_merged_text_emb_medrobnl_nconc_modifs.pkl
# pats_dict_merged_text_emb_medrobnl_nconc_modifs_dm0_non_quantized.pkl

# pats_dict_merged_text_emb_sbertnl_nconc_modifs.pkl
# pats_dict_merged_text_emb_sbertnl_nconc_modifs_dm0_non_quantized.pkl

filename_t2vmodel = f'pats_dict_merged_text_emb{outfile_infix}.pkl'

tmp = try_read_pickle(filename_t2vmodel)
tv2_models = tmp['models']
list(tv2_models.keys()) # o_t0_dist 24_48
topic_models_of_interest = [ '24_48_text_o',
        '0_48_text_s',
        '0_48_text_o',
        '0_48_text_p',
        '24_48_episode_description']
topics_of_interest_varnames = [ '24_48_text_o_t0_dist',
        '0_48_text_s_t0_dist',
        '0_48_text_o_t0_dist',
        '0_48_text_p_t0_dist',
        '24_48_episode_description_t3_dist']

def embed_medrob_with_logger(logger):
        def partial_fn(args):
                agrs_with_logger = [logger] + args
                return RobertaForMaskedLM_embedding_model(*agrs_with_logger)     
        return  partial_fn

topic_dist_var_nm = f'{topic_model_of_interest}_t0_dist'
t2v_model = tv2_models[topic_model_of_interest]
t2v_model.embed = embed_medrob_with_logger(logger)
t2v_model._check_model_status()
is_reduced = t2v_model.hierarchy is not None
n_topics = t2v_model.get_num_topics(is_reduced)

t2v_inputs = tmp['x'] # patient dict, k = pat_id, v = dict of all features (all in primitive, or dict struct, )
topic_var_nms = [x for x in list(t2v_inputs.values())[0].keys() if '_dist' in x]
top_n_patients = 10

    


#c_doc = list(t2v_inputs.values())[0]['Episodes'][0]['JOURNALS'][0]['text_o']
#list(t2v_inputs.values())[0]['Episodes'][0]['JOURNALS'][0]['journal_datetime']
del tmp

# search for topic distances for a doc
#t_words, w_scores, t_scores, t_nums = t2v_model.query_topics(c_doc, n_topics, is_reduced)
#t_scores #!

# get some docs from a topic
#docs_in_topic, doc_scores, doc_ids = t2v_model.search_documents_by_topic(0, 1, return_documents =T, reduced = is_reduced)
#docs_in_topic[0]
#doc_scores[0]

# check - do we get the same score if we query for this doc? Yes!
#t_words, w_scores, t_scores, t_nums = t2v_model.query_topics(docs_in_topic[0], n_topics, is_reduced)

# realization! t2v_inputs holds the distances of each patient to said topics 
# so we can easily get the top N patients closest to a topic
# we can also easily get the top N docs colsest to a topic
# what is a bit more tricky is then finding the patients from where those documents came from....
# leave that for last! get patients first
# toi = topic of interest


for topic_model_of_interest in topic_models_of_interest:
    c_t_varnames = [ x for x in topic_var_nms if topic_model_of_interest in x]
    c_toi_varnames = [ x for x in topics_of_interest_varnames if topic_model_of_interest in x]
    logger(f'in {topic_model_of_interest} there are {len(c_t_varnames)} topics, {len(c_toi_varnames)} of which is of interest')
    for c_t_varname in c_toi_varnames:
        patient_dists_to_toi = [(p_id, t2v_inputs[p_id][c_t_varname]) for p_id in t2v_inputs.keys() ]
        patient_dists_to_toi = sorted(patient_dists_to_toi, key= lambda item: item[1], reverse=T)


        p_ids = [x for x,_  in patient_dists_to_toi[0:top_n_patients] ]

        try_save_pickle(f = f'{c_t_varname}_pat_ids.pkl' ,  o = {p_id:t2v_inputs[p_id] for p_id in p_ids} )




patient_dists_to_toi = [(p_id, t2v_inputs[p_id][topic_dist_var_nm]) for p_id in t2v_inputs.keys() ]
patient_dists_to_toi = sorted(patient_dists_to_toi, key= lambda item: item[1], reverse=T)
#patient_dists_to_toi[0]

# interesting -  a lot of patients have -1 distance....
#dists = [x for _,x in patient_dists_to_toi]
#np.mean(dists)
#np.median(dists)
#np.quantile(dists, q=[0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] )

top_n_patients = 10


p_ids = [x for x,_  in patient_dists_to_toi[0:top_n_patients] ]
#t2v_inputs[p_ids[0]]



# What we need to specify
# topic variable name(s)
# number of closest documents to find