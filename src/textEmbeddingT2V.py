# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES:
# 2. Uses Top2Vec to create embeddings of each text field 
'''
)
# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn

IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
#check_if_debugging(IS_DEBUG)
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
cached_call = lambda fn, override_cache=F, **kwargs : try_cached_call(fn, io_r=read_pickle, io_c=pickle_exists, io_w=save_pickle, override_cache=override_cache, **kwargs)

# Boilerplate end
from collections import OrderedDict
from symspellpy.symspellpy import SymSpell, Verbosity

# from top2vec import Top2Vec
from Top2Vec import Top2Vec
from sys import stdout
from os import path
from multiprocessing import Pool as ProcessPool
from sklearn.decomposition import TruncatedSVD
#from MedRoBERTaNL_wrapper import RobertaForMaskedLM_embedding_model
import re
import gc 
import analyse_results_util as ar_util
from try_utils import __init_pats_dict_flwp_filtered, __init_pats_dict_time_unbiased
os.environ['OMP_NUM_THREADS'] = "8"
os.environ['OPENBLAS_NUM_THREADS'] = "8"
logger = print
# debug-specific
TS_WEIGHT = 0 # ignore size of topic when combining topic vectors during heirarch topic reduction
USE_PP_T2V_SCORES = False # skip re-computing distance metrics for each patient?
USE_INTERIM_FILES = False # skip fitting T2V model (if stored in ns.interim_out_file) + topic number selection?
RERUN_TOPIC_REDUCTION = True # run topic number selection?
USE_PP_DOC_FILES = False
OVERRIDE_RESULTS = True #False if not SUBSAMPLE_DATA else True
SKIP_RERUN_TOPIC_REDUCTION_COLUMNS = [] # ['0_24_text_s', '0_24_text_o', '0_24_text_e', '0_24_text_p'  ] # 

# hardcoded hyperparams
SILHOUETTE_TOPIC_PENALTY = 0.005 # i.e. per 100 extra topics, penalize score by 0.1
PRE_PROCESS_TEXT = True
MIN_DOCS_FLAT = 500 if not SUBSAMPLE_DATA else 50
MIN_DOCS_FLAT_NONEMPTY = 300 if not SUBSAMPLE_DATA else 30
# Top2Vec-specific hyperparams
t2v_learning_speed = "learn" if SUBSAMPLE_DATA else "learn" # when texts are non-quantized even with subsampling it runs slow..
n_workers = 4 if SUBSAMPLE_DATA else 8
min_count = 10 if SUBSAMPLE_DATA else 50
model_params = { "speed" : t2v_learning_speed, 
                "workers" : n_workers,
                "verbose" : T,
                'min_count': min_count }

k_doc_embeddings_file = f"doc_embeddings_file{subsampled_str}"
k_umap_embeddings_file = f"umap_embeddings_file{subsampled_str}"
k_knn_cache_file = f"umap_knn_cache_file{subsampled_str}"
k_hdbscan_labels_file = f"hdbscan_labels_file{subsampled_str}"


def print_text_stats(flat_docs, create_word_blacklist=False):
    logger("__\tTEXT STATS BEGIN\n\n*****")
    n_docs = len(flat_docs)
    from collections import Counter
    words_counts = Counter()
    for doc in flat_docs:
        words_counts.update(doc.split())
    vocab_size = len(words_counts)
    words_total = sum(words_counts.values())
    vocab, counts = list(zip(* sorted([ (k,v) for k,v in words_counts.items()], key = lambda v : v[1], reverse=T)))
    vocab = list(vocab)
    counts = list(counts)
    d_lens = [len(d.split()) for d in flat_docs]
    sd_doc_len = np.std(d_lens)
    iqr_doc_len = np.percentile(d_lens, [25,75])
    mean_doc_len = np.mean(d_lens)
    med_doc_len = np.median(d_lens)
    logger(f"_VOCAB_SIZE={vocab_size}\tMEAN_DOC_LEN={mean_doc_len:0.1f}(SD {sd_doc_len:0.2f})\tN_DOCS={n_docs}")
    logger(f"MEDIAN_DOC_LEN={med_doc_len:0.1f}(IQR {iqr_doc_len})")
    logger("\n\n***\n\n__\tTEXT STATS END\n\n*****")
    words_bl = []
    if create_word_blacklist:
        w_counts = [x for x in dict(words_counts).values()]
        suggested_min_count_thresh = int(np.quantile(w_counts, [0.025])[0]) 
        suggested_max_count_thresh = int(np.quantile(w_counts, [0.99999])[0]) 
        too_common_words = [w for w,c in zip(vocab, counts) if c>=suggested_max_count_thresh]
        too_rare_words = [w for w,c in zip(vocab, counts) if c<suggested_min_count_thresh]
        logger(f"Going to remove top {len(too_common_words)} most frequent words... = {too_common_words} count thesh = {suggested_max_count_thresh}")
        logger(f"Going to remove words {len(too_rare_words)} most infrequent words... = {too_rare_words} count thesh = {suggested_min_count_thresh}")
        words_bl = too_common_words + too_rare_words
    return words_bl

# Main function
def run_it(logger1, embedding_model, d2v_dm, include_medications, include_measurements, in_file, out_file, interim_out_file):
    global logger
    logger = logger1
    # init embedding model
    embedding_model_path = get_embedding_model_path(embedding_model)
    emb_m_shortname = EMBEDDING_MODEL_METADATA[embedding_model]['short_name']
    is_callable = EMBEDDING_MODEL_METADATA[embedding_model]['is_callable']
    embedding_model_for_top2vec = eval(embedding_model) if is_callable else embedding_model
    is_custom_sbert = EMBEDDING_MODEL_METADATA[embedding_model]['is_custom'] and EMBEDDING_MODEL_METADATA[embedding_model]['is_sbert']
    custom_sbert_embedding_model = embedding_model if is_custom_sbert else None
    # define some intermitted filenames
    doc_embeddings_file = EMBEDDING_MODEL_METADATA[embedding_model][k_doc_embeddings_file]
    umap_embeddings_file = EMBEDDING_MODEL_METADATA[embedding_model][k_umap_embeddings_file]
    umap_knn_cache_file = EMBEDDING_MODEL_METADATA[embedding_model][k_knn_cache_file]
    hdbscan_labels_file = EMBEDDING_MODEL_METADATA[embedding_model][k_hdbscan_labels_file]
    logger(f"doc_embeddings_file = {doc_embeddings_file}")
    logger(f"umap_embeddings_file = {umap_embeddings_file}")
    logger(f"hdbscan_labels_file = {hdbscan_labels_file}")
    logger(f"umap_knn_cache_file = {umap_knn_cache_file}")
    
    subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
    interim_out_file = f"{interim_out_file}.pkl" # f'intrmtn_top2VecModels.pkl'
    pp_doc_file = f'documents_preprocessed_.pkl'
    sil_score_out_file = f'sil_scores_$ID$.tsv'

    if not OVERRIDE_RESULTS and path.exists(out_file):
        logger(f"INFO: output file {out_file} already exists! Skipping... ")
        return out_file

    config_params = {
        "SILHOUETTE_TOPIC_PENALTY" : SILHOUETTE_TOPIC_PENALTY,
        "embedding_model" : embedding_model,
        "emb_m_shortname" : emb_m_shortname,
        "t2v_learning_speed": t2v_learning_speed,
        "in_file" : in_file,
        "interim_out_file" : interim_out_file,
        'pp_doc_file' : pp_doc_file,
        "sil_score_out_file" : sil_score_out_file,
        "out_file" : out_file,
        "embedding_model_path": embedding_model_path,
        "include_measurements" : include_measurements,
        "include_medications" : include_medications
    }

    logger("====> config_params:=")
    for k,v in config_params.items():
        logger(f"{k} = {str(v)}")

    tmp = read_pickle(in_file)
    x = tmp['x']
    if SENS_ANALYSIS_UNBIAS_FLWP_START:
        x = cached_call(__init_pats_dict_flwp_filtered, override_cache=T, x = x)

    # dedup x texts 
    transform_p1 = lambda p1, doc_time_text: {'p_id': p1[0], 'doc_time':doc_time_text[0], 'doc' : doc_time_text[1] }
    get_texts_dicts_pat = lambda p1 : [ transform_p1(p1,i) for i in zip(p1[1]['0_24_epj_time_'], p1[1]['0_24_epj_text_'])]

    all_pats_dicts_texts = [get_texts_dicts_pat((k,v)) for k,v in x.items()]
    all_pats_dicts_texts = [ i for o in all_pats_dicts_texts for i in o]
    # p_id, doc_time, doc # 
    pats_texts_df = pd.DataFrame.from_records(all_pats_dicts_texts)
    n1 = nrow(pats_texts_df)
    pats_texts_df['doc'] = pats_texts_df['doc'].apply(str.strip)
    pats_texts_df = pats_texts_df[pats_texts_df['doc'] != '']
    pats_texts_df = pats_texts_df.drop_duplicates(ignore_index=T)
    logger(f"Before texts deduplication N={n1}; After texts deduplication N={nrow(pats_texts_df)}")

    empty_txt_p_ids = try_sd(x.keys(), pats_texts_df['p_id'].values)
    for p_id in empty_txt_p_ids:        
        x[p_id]['0_24_epj_time_'] = []
        x[p_id]['0_24_epj_text_'] = []

    for p_id,p_df in  pats_texts_df.groupby('p_id'):
        x[p_id]['0_24_epj_time_'] = list(p_df['doc_time'].values)
        x[p_id]['0_24_epj_text_'] = list(p_df['doc'].values)

    age_id_df = [(k, v['age_days'], v[VAR_FOLLOW_UP_DATE]) for k,v in x.items()]
    age_id_df = pd.DataFrame.from_records(age_id_df)
    age_id_df.columns = ['id', 'age_days', VAR_FOLLOW_UP_DATE]
    age_id_df = apply_min_age_selection(age_id_df) # duplicated in 3_pre_process.py; candidate to do once, earlier (see OPEN_QUESTIONS.md)
    x = {k:x[k] for k in age_id_df.id.values.tolist()}
    del age_id_df
    def create_id_Y_id(x):
        id_Y_df = pd.DataFrame.from_records([(k, v['t_HF'], v['age_days'], v[VAR_FOLLOW_UP_DATE]) for k,v in x.items()])
        id_Y_df.columns = ['id', 'Y', 'age_days', VAR_FOLLOW_UP_DATE]
        id_Y_df['age_years'] = (id_Y_df[VAR_FOLLOW_UP_DATE] + id_Y_df['age_days'])/365
        return id_Y_df

    id_Y_df = create_id_Y_id(x)
    case_ids = id_Y_df[~pd.isnull(id_Y_df['Y'])]['id']
    if SUBSAMPLE_DATA and len(case_ids) < 10:
        logger(f"DEBUG SUBSAMPLE ADDING ARTIFICIALLY CASE labels (avoid 0 cases)")
        new_case_ids = random.sample(list(x.keys()), round(min(len(x), max(len(x)/20, 10))))
        for pid in new_case_ids:
            x[pid]['t_HF'] = x[pid]['follow_up_LAST']
        id_Y_df = create_id_Y_id(x)
        case_ids = id_Y_df[~pd.isnull(id_Y_df['Y'])]['id']
        
    assert len(id_Y_df[~pd.isnull(id_Y_df['Y'])]['age_years']) > 0
    del id_Y_df

    
    n_pats = len(x)
    text_cols = tmp['text_cols']
    time_cols = tmp['time_cols']

    if not include_measurements:
        text_cols = [tc for tc in text_cols if 'measurement_txt' not in tc]
    if not include_medications:
        text_cols = [tc for tc in text_cols if 'medication_txt' not in tc]
    logger(f"N patients read from input  = {len(x)}")
    del tmp
    top2vec_X = {}
    models = {}
    # text_cols = text_cols
    # time_cols = time_cols
    all_pids_docs_times = {}
    all_pids_to_idx ={}
    cached_all_pids_docs_times = f"all_pids_docs_times.pkl"

    if not pickle_exists(cached_all_pids_docs_times) or not USE_INTERIM_FILES:    
        for text_col, time_col in zip(text_cols, time_cols): 
            pids_docs = [ (p_id, x[p_id][text_col])  for p_id in x.keys() if text_col in x[p_id] and len(x[p_id][text_col]) > 0]
            all_pids_docs_times[text_col] = pids_docs
            pids_times = [ (p_id, x[p_id][time_col])  for p_id in x.keys() if time_col in x[p_id] and len(x[p_id][time_col]) > 0]
            all_pids_docs_times[time_col] = pids_times
        all_pids_to_idx = { k:i for i,k in enumerate(x.keys())}
    
        save_pickle(f"all_pids_docs_times.pkl", { 'all_pids_docs_times': all_pids_docs_times,
                                                                'all_pids_to_idx': all_pids_to_idx  })
        all_pids_docs_times = {}
        all_pids_to_idx = {}

    del x
    for text_col, time_col in zip(text_cols, time_cols): 
        gc.collect()
        logger(f"****************** START {text_col}  *************************")
        c_interim_out_file = f"{text_col}_{interim_out_file}"
        c_pp_doc_file = f"{text_col}_{pp_doc_file}"
        tfidf_dict = {}
        model = None
        already_reduced = F
        new_num_topics = None
        c_pp_doc_used = F
        if USE_PP_DOC_FILES and pickle_exists(c_pp_doc_file):
            tmp = read_pickle(c_pp_doc_file)
            flat_docs = tmp['flat_docs']
            flat_times = tmp['flat_times']
            docid_2_pid_idx = tmp['docid_2_pid_idx']
            c_pp_doc_used = T
            del tmp
        

        if USE_INTERIM_FILES and pickle_exists(c_interim_out_file):
            logger(f"{text_col}; Reading previous state from {c_interim_out_file}")
            o = read_pickle(c_interim_out_file)
            model = o['model']
            if is_callable:
                model.embedding_model = embedding_model_for_top2vec
            if not emb_m_shortname == 'd2v':
                model._check_model_status()
            tfidf_dict = o['tfidf_dict']
            already_reduced = o['reduced']
            if already_reduced:
                new_num_topics = o['new_num_topics']

        if RERUN_TOPIC_REDUCTION and already_reduced and text_col not in SKIP_RERUN_TOPIC_REDUCTION_COLUMNS:
            already_reduced = F
            logger(f"RERUN_TOPIC_REDUCTION = TRUE, going to re-run selection of optimum number of topics for {text_col}")
        if RERUN_TOPIC_REDUCTION and already_reduced and text_col in SKIP_RERUN_TOPIC_REDUCTION_COLUMNS:
            logger(f"SKIP_RERUN_TOPIC_REDUCTION_COLUMNS for {text_col}")
        if model and already_reduced:
            logger(f"{text_col}; Skipping model build, tfidf_dict build, select optimum topics; loading these from {c_interim_out_file}..")
            models[text_col] = c_interim_out_file
        if model and not already_reduced:
            logger(f"{text_col}; Skipping model build, tfidf_dict build; loaded from {c_interim_out_file}; select optimum topics now going to run ...")

        if not c_pp_doc_used:
            tmp = read_pickle(cached_all_pids_docs_times)
            pids_docs = tmp['all_pids_docs_times'][text_col]
            pids_times = tmp['all_pids_docs_times'][time_col]
            all_pids_to_idx = tmp['all_pids_to_idx']
            del tmp
            
            if(len(pids_docs) < 100):
                logger(f"Skipping text column {text_col} due to insufficient non-empty texts.")
                continue
            c_pids = list(list(zip(*pids_docs))[0])
            c_documents = list(list(zip(*pids_docs))[1])
            c_times = list(list(zip(*pids_times))[1])
            logger(f"Starting: Column {text_col}: has {len(c_documents)} documents and {sum([len(x) for x in c_documents])} characters")
            # Optional steps: 
            logger(f"{text_col} i) Remove as many \"informationless\" words as possible,  lowercase all")
            logger(f"{text_col} ii) concatenate negation words with next word")

            if PRE_PROCESS_TEXT:

            
                # translate doctors shorthands & misspellings
                shorthand_mappings_pos = { '+' : 'positief',
                                        '++' : 'heel-positief',
                                        '+++' : 'erg-positief'}

                shorthand_mappings_neg = { '-' : 'negatief', 
                                        '--' : 'heel-negatief',
                                        '---' : 'erg-negatief'}

                t0 = logger(f"Calculating terms_per_doc 1st time...")
                term_splitter_fn = lambda c_doc: [x[0] if x[0] != '' else x[1] for x in re.findall(r"(<<>>)|(\b\w[\w\-']*\b)", c_doc.lower())]
                terms_per_doc = []
                if SHOULD_SOEP_SEPERATE_TEXTS:
                    terms_per_doc = [term_splitter_fn(c_doc) for c_doc in c_documents]
                if not SHOULD_SOEP_SEPERATE_TEXTS:
                    for i_docs, c_docs in enumerate(c_documents):
                        terms_per_doc += [ [term_splitter_fn(c_doc) for c_doc in c_docs] ]
                t0 = logger(f"Calculating terms_per_doc 1st time... done", t0)

                for terms in terms_per_doc:
                    terms_corrected = []


                # remove: single-letter words,  single-digit words, persoon- words, 
                words_to_remove_regexes = ['^.$', '^\d+.*$', '^perso.+$', '^telefo.+$', '^datum.*$', '^\w*amstel.*$', '^patient.*$', '^.*nummer.*$',
                '^ziekenhuis.*$', '^.*afspraak.*$', '^.*formulier.*$', '^.*administratie.*$', '^locatie-\d+$']
                words_to_remove_exact = list(set(['dd', 'daarbij' ,'en' ,'bij' ,'met' ,'na' ,'van' ,'in' ,'toch' ,'voor' ,'de' ,'of' ,'op' ,'door' ,
                'te' ,'re' ,'nog' ,'het' ,'als' ,'wat' ,'een' , 
                'aan' ,'links' ,'wel' ,'is' ,'dat' ,'naar' ,'st' ,'al' ,'ex' ,'dan' ,'wil' ,'ik' ,'om' ,'kan' ,'sinds' ,'li' ,'mg' ,'via' ,'deze' ,
                'dus' ,'tot' ,'mijn', 'dr', 'ziekte', 'hr',
                'per', 'ze', 'toe', 'uur', 'alleen', 'laten', 'zo', 'mogelijk', 'week', 'wk', 'eigenlijk', 'heel', 'omdat', 
                'daar', 'veel', 'want', 'vooral', 'af', 'gv', 'vraag', 'mevrouw', 
                'mee', 'graag', 'hier', 'anders', 'die', 'maar', 'iets', 'alles', 'soms', 'zijn', 'zelf', 'ook', 'haar',  'ok', 'dr',
                'hun', 'kunnen', 'hebben', 'hem', 'dhr',
                'zoon', 'er', 'meer', 'willen', 'minder', 'mw', 'hele', 'hij', 'net', 'gekregen', 'maken', 'zij', 'doet', 
                'doen', 'lijkt', 'was', 'zich', 'dit', 'over',
                'nl', 'fax', 'code', 'flevohuis', 'koesoebjono', 'vanwege', 'soort', 'zh', 'heer', 'vandaag', 'gaat', 'gaan', 'gegaan' ]))
                prefix_modifier_words_pos = ['goede', 'goed'] 
                prefix_modifier_words_neg = ['geen', 'zonder', 'niet', 'niets', 'nooit', 'tegen', 'goed_NEG'] # niet could come after.. 
                
                suffix_modifier_words_pos = ['pos','goed', 'ja']  + list(shorthand_mappings_pos.values()) # niet could come after.. 
                suffix_modifier_words_neg = ['neg', 'goed_NEG']  + list(shorthand_mappings_neg.values()) # niet could come after.. 

                # negative words that come aftter: neg, negatief "$WORD - "
                # that come before/after : vaak
                # positive words that come before : goede 
                # positive words that come after : goed , + , ja 
                t0 = logger(f"{text_col} filter/concatenate-effectors in terms in terms_per_doc...")
                if SHOULD_SOEP_SEPERATE_TEXTS:
                    terms_per_doc = [terms_per_doc] # to simulate doc of docs structure used in non-quantized version
                    c_documents = [c_documents]

                for i_docs, _ in enumerate(terms_per_doc):
                    for i, c_terms in enumerate(terms_per_doc[i_docs]):
                        c_terms_filtered = [x for x in c_terms if x not in words_to_remove_exact and not any([try_regex(c_r, x) for c_r in words_to_remove_regexes]) ]
                        c_terms_filtered = [ f"{c_term}_NEG" if j>0 and c_terms_filtered[j-1] in prefix_modifier_words_neg else c_term
                                                for j, c_term in enumerate(c_terms_filtered) if c_term not in prefix_modifier_words_neg]
                        c_terms_filtered = [ f"{c_term}_POS" if j>0 and c_terms_filtered[j-1] in prefix_modifier_words_pos else c_term
                                                for j, c_term in enumerate(c_terms_filtered) if c_term not in prefix_modifier_words_pos]

                        c_terms = c_terms_filtered
                        c_terms = [ f"{c_term}_NEG" if j<(len(c_terms)-1) and c_terms[j+1] in suffix_modifier_words_neg else c_term
                                                for j, c_term in enumerate(c_terms) if c_term not in suffix_modifier_words_neg]

                        c_terms = [ f"{c_term}_POS" if j<(len(c_terms)-1) and c_terms[j+1] in suffix_modifier_words_pos else c_term
                                                for j, c_term in enumerate(c_terms) if c_term not in suffix_modifier_words_pos]
                        c_documents[i_docs][i] = " ".join(c_terms).strip()
                t0 = logger(f"{text_col} filter/concatenate-effectors in terms in terms_per_doc... done", t0)
                del terms_per_doc
            else:
                logger(f"Skip PRE_PROCESS_TEXT...")

            n_docs = sum([len(docs) for docs in c_documents])
            n_chars = sum([len(doc) for docs in c_documents for doc in docs])
            logger(f"Now: Column {text_col}: has {n_docs} documents and {n_chars} characters")            
            pid_2_fdoc_ids = {} # can delete
            fdoc_ids_2_pid = {} # can delete
            doc_counter = 0
            for pid, c_docs  in pids_docs: # create a lookup from pid to doc id
                if pid not in pid_2_fdoc_ids:
                    pid_2_fdoc_ids[pid] = []
                n_docs = len(c_docs)
                pid_2_fdoc_ids[pid] += list(range(doc_counter, doc_counter+n_docs))
                doc_counter+=n_docs
            for pid, fdoc_ids in pid_2_fdoc_ids.items(): # pid, fdoc_ids = list(pid_2_fdoc_ids.items())[0]
                fdoc_ids_2_pid.update({str(fdoc_id): pid for fdoc_id in fdoc_ids})

            docid_2_pid_idx = [] # keep
            for d_id in range(len(fdoc_ids_2_pid)):
                docid_2_pid_idx += [all_pids_to_idx[fdoc_ids_2_pid[str(d_id)]]]
            docid_2_pid_idx = np.array(docid_2_pid_idx)

            del fdoc_ids_2_pid
            del pid_2_fdoc_ids

            flat_docs = [c_doc for c_docs in c_documents for c_doc in c_docs]
            flat_times = [c_time for c_tms in c_times for c_time in c_tms]
            del c_documents
            del pids_times
            del pids_docs

            logger(f"Now: Column {text_col}: has {len(flat_docs)} flat docs")
            # words_bl = print_text_stats(flat_docs, create_word_blacklist =T)
            
            #logger(f"Spell corrections will be done for {len(spell_corrections)} words: {list(spell_corrections.items())[:10]}")
            # flat_docs = [  " ".join([w for w in d.split() if w not in words_bl]) for d in flat_docs ]
            # flat_docs = [  " ".join([spell_corrections[w] if w in spell_corrections else w for w in d.split() if w not in words_bl]) for d in flat_docs ]
            logger(f"After corrections... stats are::")
        print_text_stats(flat_docs)
        if len(flat_docs) < MIN_DOCS_FLAT:
            logger(f"skipping column {text_col}, as it has less than {MIN_DOCS_FLAT} flat docs...")
            continue
        if sum([1 if len(doc) > 1 else 0 for doc in flat_docs]) < MIN_DOCS_FLAT_NONEMPTY:
            logger(f"skipping column {text_col} as it has less than {MIN_DOCS_FLAT_NONEMPTY} non-empty flat docs...")
            continue

        if not c_pp_doc_used:
            save_pickle(f=c_pp_doc_file, o = {          'flat_docs': flat_docs,
                                                        'flat_times' :  flat_times,
                                                        'docid_2_pid_idx' : docid_2_pid_idx
                                                        })
        

        
        #del pids_docs
                                                                
        if not model:
            t2 = logger(f"Running Top2Vec for {text_col}")
            n_docs = len(flat_docs)

            # run specific
            
            # min_count - n times a word should appear at minimum 
            min_count = 1 # handled elsewhere in print_stats
            model_params['min_count'] = min_count
            model_params["sentencizer"] =  None # lambda doc: doc.split(TXT_ENTRY_SEP)  # t.y. 21.Oct.2024 , is_episode_desc else None
            model_params["embedding_model"] = embedding_model_for_top2vec
            model_params["embedding_model_path"] = embedding_model_path
            model_params["sbert_embedding_model"] = custom_sbert_embedding_model
            model_params["split_documents"] = F  # t.y. 21.Oct.2024  if is_episode_desc else None
            model_params["document_chunker"] = None  # t.y. 22.Oct.2024  if is_episode_desc else None
            model_params["d2v_dm"] = d2v_dm
            model_params['keep_documents'] = F
            model_params['topic_merge_delta'] = [0.1] #np.arange(0.1, 0.95, 0.005)
            model_params["umap_subsample"] = 0.5 # 

            from numba import get_num_threads
            n_cpus = get_num_threads()

            if not SUBSAMPLE_DATA and n_docs > 1e6:
                os.environ['OMP_NUM_THREADS'] = "8"
                os.environ['OPENBLAS_NUM_THREADS'] = "8"
            
            min_clust_size_hdbscan =  200 if not SUBSAMPLE_DATA else 20
            umap_n_neighbors = 200 if not SUBSAMPLE_DATA else 5

            if SENS_ANALYSIS_UNBIAS_FLWP_START and not SUBSAMPLE_DATA:
                min_clust_size_hdbscan = 100 # since we have fewer records here
                umap_n_neighbors = 50
            
            
            model_params["umap_args"] = {'n_neighbors': umap_n_neighbors, # modified by  T.Y. (2.Aug.2024) default: 15
                                        'n_components': 8, # default 2, T2V default is 5
                                        'min_dist': 0.1, # default 0.1
                                        'metric': 'cosine', # match doc2vec metric
                                        'n_jobs': n_cpus if SUBSAMPLE_DATA else min(10, n_cpus),
                                        'verbose': True,
                                        'knn_cache_file' : umap_knn_cache_file[text_col] if text_col in umap_knn_cache_file else None } 
            model_params['hdbscan_args']  = {'min_cluster_size': min_clust_size_hdbscan , # modified by  T.Y. (31.Jul.2024) default: 15 
                                            'metric': 'euclidean',
                                            'cluster_selection_method': 'eom',
                                            'min_samples': max(int(min_clust_size_hdbscan/20), 5),
                                             'core_dist_n_jobs': 8 }

            model_params['umap_preprocessor'] = None #TruncatedSVD(n_components=100)

            model_params['previous_embeddings_file'] = doc_embeddings_file[text_col] if text_col in doc_embeddings_file else None
            model_params['previous_umap_emb_file'] = umap_embeddings_file[text_col] if text_col in umap_embeddings_file else None
            
            model_params['previous_hdbscan_labs_file'] = hdbscan_labels_file[text_col] if text_col in hdbscan_labels_file else None
            model_params['in_logger'] = logger
            model_params['SUBSAMPLE_DATA'] = SUBSAMPLE_DATA

            # we run into performance bottlenecks when too many topics are found
            # above set of params works for edsoepx format for 6m, and 12m, but then gets too slow
            is_oom_likely = not SUBSAMPLE_DATA and ( not REDUCE_N_TOPICS or embedding_model_for_top2vec != 'doc2vec' ) 

            logger(f"Model params are:\n\t\t{model_params}")
            if model_params['previous_embeddings_file'] is not None:
                flat_docs = None
            model = Top2Vec(flat_docs, **model_params)
            logger(f"Running Top2Vec for {text_col} .. done ", t2)
            # logger(f"Top 3 words of top 5 biggest topics: {[i[:3] for i in model.get_topics()[0][:5]]}")
            # save before topic reducing 
            embed_fn = model.embed if embedding_model != 'doc2vec' else None
            if embedding_model != 'doc2vec':
                model.embed = None # otherwise cant be pickled
            save_pickle(f=c_interim_out_file, 
                o = {"model": model,
                    "text_col" : text_col,
                    "tfidf_dict" : tfidf_dict,
                    "reduced" : F,
                    "new_num_topics" : model.topic_vectors.shape[0]})
            if embedding_model != 'doc2vec':
                model.embed = embed_fn
        else:
            logger(f"Model Top2Vec for {text_col} ... already derived! Skipping run... ")
        del flat_docs
        n_topic_orig = model.get_num_topics(reduced = F)
        new_num_topics = n_topic_orig
        #remove_trailing_nonalpha = lambda x : re.sub( r"[^a-zA-Z_]+$", "", x)
        topic_words, _, topic_idxs = model.get_topics(reduced = F)
        #word_scores = [transform_cosine_dist(x) for x in word_scores]
        log_every_n_percent = 0.5 if SUBSAMPLE_DATA else 0.05
        if REDUCE_N_TOPICS:
            if not already_reduced:
                t0 = logger(f"{text_col} Select best number of topics..")
                new_num_topics = select_best_number_of_topics(text_col, model, verbose=T, penalized = T, sil_score_out_file = sil_score_out_file)
                t0 = logger(f"{text_col} Select best number of topics... done", t0)
            else:
                logger(f"best num of topic from pre-loaded state: {new_num_topics}")

            # new_topics = model.get_topics()
            if new_num_topics != n_topic_orig:
                new_topics = None
                if not already_reduced:
                    logger(f"Running try_hierarchical_topic_reduction on {n_topic_orig} topics, to make into {new_num_topics}")
                    model, new_topics = try_hierarchical_topic_reduction(model = model, num_topics = new_num_topics)
                else:
                    new_topics = model.get_topic_hierarchy()

                for i, v in enumerate(new_topics):
                    logger(f"NEW topic_{i} = old topics [{v}]")
                    if i > 5:
                        print("...")
                        break
                
        if embedding_model != 'doc2vec':
            model.embed = None # otherwise cant be pickled
        save_pickle(f=c_interim_out_file, 
            o = {"model": model,
                "text_col" : text_col,
                "tfidf_dict" : tfidf_dict,
                "reduced" : not model.hierarchy is None,
                "new_num_topics" : new_num_topics})

        # topic_words = n by d array where n = num of topics, d =50;
        # for each topic the top 50 words are returned in order of semantic similarity to topic 
        did_reduce = not model.hierarchy is None
        topic_words, _, topic_idxs = model.get_topics(reduced = did_reduce) # using reduced=T links it to the try_hierarchical_topic_reduction
        #word_scores = [transform_cosine_dist(x) for x in word_scores]
        logger(f"model for {text_col} found {len(topic_idxs)} topics")
        models[text_col] = c_interim_out_file

        t_log = max(1, round(new_num_topics*log_every_n_percent))
        logger("Start calculating topic distances for each doc...")

        pat_topic_metrics = {
            "MX" : { # what is the max similarity this patient has ever had to this topic
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_mx_dist",
                "scoring" : lambda c_vals, c_tms: [ max(c_scores) for c_scores in c_vals ] 
            },
            "PA" : { # 
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_pa_dist",
                "scoring" : lambda c_vals, c_tms: [ np.mean(c_scores) for c_scores in c_vals ] 
            },
            # uses doc order as a recency proxy (docs are stored most-recent-first) since
            # per-doc timestamps aren't threaded through from 2d1_pre_process_text.py yet (see OPEN_QUESTIONS.md)
            "twPA" : { # list(range(1,len(c_scores)+1))[::-1]
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_twPA_dist",
                "scoring" : lambda c_vals, c_tms : [ np.average(c_scores, weights=c_times) if sum(c_times)!= 0 else 0 for c_scores, c_times in zip(c_vals,c_tms) ] 
            }, 
            "DIT" : {
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_dit_dist",
                "scoring" : lambda c_vals, c_tms: [ sum([1 if c_sc > 0 else 0 for c_sc in c_scores]) for c_scores in c_vals ] 
            },
            "twDIT" : {
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_twdit_dist",
                "scoring" : lambda c_vals, c_tms : [ sum([ c_tm if c_sc > 0 else 0 for c_sc,c_tm in zip(c_scores, c_times) ]) for c_scores, c_times in zip(c_vals, c_tms) ] 
            },
            "PS" : {
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_ps_dist",
                "scoring" : lambda c_vals, c_tms: [ sum(c_scores) for c_scores in c_vals ] 
            },
            "twPS" : {
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_twps_dist",
                "scoring" : lambda c_vals, c_tms : [ np.sum(np.array(c_scores)*np.array(c_times)) if sum(c_times)!= 0 else 0 for c_scores, c_times in zip(c_vals,c_tms) ] 
            }
        }
        pat_topic_metrics["twDIT50"] = {
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_twdit50_dist",
                "scoring" : lambda c_vals, c_tms :  np.add(np.array(pat_topic_metrics["twDIT"]["scoring"](c_vals, c_tms))*0.5, np.array(pat_topic_metrics["DIT"]["scoring"](c_vals, c_tms))*0.5 )  
            }
        pat_topic_metrics["twPA50"] = {
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_twpa50_dist",
                "scoring" : lambda c_vals, c_tms :  np.add(np.array(pat_topic_metrics["twPA"]["scoring"](c_vals, c_tms))*0.5, np.array(pat_topic_metrics["PA"]["scoring"](c_vals,c_tms))*0.5 )  
            }
        pat_topic_metrics["twPS50"] = {
                "col_naming" : lambda text_col, t_n: f"{text_col}_t{t_n}_ps_dist",
                "scoring" : lambda c_vals, c_tms :  np.add(np.array(pat_topic_metrics["twPS"]["scoring"](c_vals, c_tms))*0.5, np.array(pat_topic_metrics["PS"]["scoring"](c_vals,c_tms))*0.5 )  
            }
        top10_logged = F
        
        n_docs = len(model.document_ids)
        for k_metric in pat_topic_metrics.keys():
            top2vec_X = {}
            c_metric_fn = f"t2v_scrs_{text_col}{k_metric}.pkl"
            if USE_PP_T2V_SCORES and pickle_exists(c_metric_fn):
                logger(f'Skip topic distance metrics for {k_metric} in column {text_col} (already available in {c_metric_fn})')
                top2vec_X = read_pickle(c_metric_fn)
                continue
            
            for t_n, t_s in enumerate(model.get_topic_sizes(reduced = did_reduce)[0]): # t_n, t_s = list(enumerate(model.get_topic_sizes(reduced =T)[0]))[0]
                k_col_nm = pat_topic_metrics[k_metric]['col_naming'](text_col, t_n)
                # store only topic distance scores where at least this many docs are in there
                should_store = t_s >= 500 if not SUBSAMPLE_DATA else t_s >= 20
                if not should_store:
                    logger(f'Stopping after topic {t_n} since not enough documents present...')
                    break

                if not top10_logged and t_n <= 10 :
                    logger(f'::> topic {t_n} with {t_s} elements and top 10 words:\n\t\t{[g for h,g in enumerate(topic_words[t_n]) if h < 10]}\n\n')
                    if t_n == 10:
                        print("...")  
                if t_n % t_log == 0:
                    logger(f'metric {k_metric} processing topic {t_n}... ouf of {new_num_topics} ({( (100*t_n)/t_log)*log_every_n_percent}% complete)')
                # document_scores showes the distance of each doc to the curren topic (only non -1 vals shown)
                document_scores, document_ids = model.search_documents_by_topic(topic_num = t_n, num_docs = t_s, reduced = did_reduce)
                x_is = docid_2_pid_idx.take(document_ids)
                
                document_scores = [transform_cosine_dist(x) for x in document_scores]
                document_times = [flat_times[d_id] for d_id in document_ids]
                c_col_nm = f"{text_col}_t{t_n}_dist" # current topic metric name (default is plain average PA) , old, decprecated...
                c_vals = [0.0] *(n_pats) if SHOULD_SOEP_SEPERATE_TEXTS else [[]] *(n_pats) # intialize PA distance metric per each patient
                c_time_weights = [0.0] *(n_pats) if SHOULD_SOEP_SEPERATE_TEXTS else [[]] *(n_pats)
                if SHOULD_SOEP_SEPERATE_TEXTS:
                    for x_i,v in zip(x_is, document_scores): # x_i,v = list(zip(x_is, document_scores))[0]
                        c_vals[x_i] = v
                else:
                    for x_i,v,t in zip(x_is, document_scores, document_times ): # x_i,v = list(zip(x_is, document_scores))[0]
                        c_vals[x_i] = c_vals[x_i] + [v]
                        c_time_weights[x_i] = c_time_weights[x_i] + [t]
                c_vals = [c if len(c)>0 else [0] for c in c_vals]
                c_time_weights = [c if len(c)>0 else [0] for c in c_time_weights]
                if SHOULD_SOEP_SEPERATE_TEXTS:
                    top2vec_X[c_col_nm] = c_vals
                if not SHOULD_SOEP_SEPERATE_TEXTS:
                    c_vals = [c_scores if c_scores else [0] for c_scores in c_vals]
                    k_scores = pat_topic_metrics[k_metric]['scoring'](c_vals, c_time_weights)
                    top2vec_X[k_col_nm] = k_scores

            logger(f"Save metric scores for {k_metric} of text col {text_col}.. n topics stored = {len(top2vec_X)} out of {new_num_topics}")
            save_pickle(f = c_metric_fn, o = {text_col : top2vec_X})
            del top2vec_X
            top10_logged = T


    logger("NOTE: patients without text will be given a 0 distances to the topic (0.0)")
    # update x dict with all topic scores..
    
    new_x = read_pickle(in_file)['x']
    new_x = cached_call(__init_pats_dict_flwp_filtered, override_cache=F, x = new_x)
    age_id_df = [(k, v['age_days'], v[VAR_FOLLOW_UP_DATE]) for k,v in new_x.items()]
    age_id_df = pd.DataFrame.from_records(age_id_df)
    age_id_df.columns = ['id', 'age_days', VAR_FOLLOW_UP_DATE]
    age_id_df = apply_min_age_selection(age_id_df)
    new_x = {k:new_x[k] for k in age_id_df.id.values.tolist()}
    del age_id_df

    for k in new_x.keys():
        for text_col in text_cols:
            if text_col in new_x[k]:
                del new_x[k][text_col] 

    varsize_divizor = 2 ** 30 if not SUBSAMPLE_DATA else 2 ** 10
    size_unit = 'Gb' if not SUBSAMPLE_DATA else 'Kb'
    for text_col in text_cols:
        logger(f"Size of new_x so far = {sys.getsizeof(new_x)/varsize_divizor:.5f}{size_unit}")
        for k_metric in pat_topic_metrics.keys():
            c_metric_fn = f"t2v_scrs_{text_col}{k_metric}.pkl"
            top2vec_X = read_pickle(c_metric_fn)
            for n_k in top2vec_X.keys(): # n_k = list(top2vec_X.keys())[0]
                for n_dst in top2vec_X[n_k].keys(): 
                    for i,k in enumerate(new_x.keys()): # i,k = list(enumerate(x.keys()))[0]
                        new_x[k][n_dst] = top2vec_X[n_k][n_dst][i]

    save_pickle(out_file,
        o = { 'models': models,
                'x' : new_x }) # xxx.hierarchy is None => did_reduce =NO; xxx.get_topics(reduced = xxx.hierarchy is not None)
    logger("DONE")
    return out_file


# Util functions
def compute_silhouette_score_from_topic_doc_scores(model, penalized=False, is_reduced=False,
        topic_penalty=0.0, subsample=False, verbose=False, topic_size_weighted=False):
    transform_cosine_dist = lambda x: (x+1)/2
    topic_words, _, topic_nums = model.get_topics(reduced = is_reduced)

    n_topics = len(topic_nums)
    if n_topics < 2:
        # logger(f"WARN: cannot compute silhouette score for model with {n_topics} topics. Returning 0.")
        return 0
    all_doc_scs = [] # store the sil scores of each topic
    doc_ids = model.document_ids
    # subsampled_doc_id_to_doc_id = []
    if subsample or len(doc_ids) > 10000:  # even if not specified, subsample for large doc corpora
        c_shuffle = list(range(len(doc_ids)))
        random.shuffle(c_shuffle)
        subsamplecount = min(round(len(c_shuffle)/10), 25000)
        doc_ids = sorted([ doc_ids[i] for i in c_shuffle[:subsamplecount]])
    
    # topic_ids_of_docs - more like ranking list of topics per doc, we care only about the first one for now
    topic_ids_of_docs, topic_scores, _, _ = model.get_documents_topics(doc_ids = doc_ids,  reduced = is_reduced, num_topics = 1 ) # get the topic of each document
    topic_scores = transform_cosine_dist(topic_scores)
    sil_scores = []
    # topic_scores = n docs by n of topics
    docids_of_topics = [ np.where(topic_ids_of_docs == ti)[0] for ti in model.get_topics(reduced=is_reduced)[2]]
    docids_of_topics = [ [d for d in dids if d in doc_ids] for dids in docids_of_topics ]
    
    for ti in range(len(docids_of_topics)):
        c_doc_ids = docids_of_topics[ti]
        c_doc_idxs = [i for i,v in enumerate(doc_ids) if v in c_doc_ids]
        if len(c_doc_idxs) == 0: # it seems that some topics end up not having documents...
            sil_scores += [0.5]
            continue
        c_doc_scores = topic_scores[c_doc_idxs] # how similar are documents of ti to topic centroid, closer to 1 better
        # clarification: higher doc/topic scores == shorter distance, score is transformed to range from 0 to 1, so dist:= 1-score

        c_a_i = 1 - np.mean(c_doc_scores) # dist between topic center and c_docs , closer to 0 better

        _, outer_topic_scores, _, _= model.get_documents_topics(doc_ids = c_doc_ids,  reduced = is_reduced, num_topics = n_topics ) # get the topic of each document

        c_doc_scores_outside_topics = transform_cosine_dist(outer_topic_scores[:, 1:])
        c_b_i = 1 - np.mean(c_doc_scores_outside_topics) # dist between c_docs and all other topic centroids, we want b_i > a_i this for good fits

        # by default this ranges from -1 to 1, transform_cosine_dist will make it range from 0 to 1 (easier to interpret)
        c_sc = transform_cosine_dist( (c_b_i - c_a_i)/max(c_b_i, c_a_i) ) # recall, if b_i > a_i, then c_sc will be positive, e.g., (0.8 - 0.4 )/ 0.8 = 0.5
        if np.isnan(c_sc):
            c_sc = 0.5
        sil_scores += [c_sc]

    if topic_size_weighted:
        new_sc_score = np.average(sil_scores, weights=model.get_topic_sizes(reduced=is_reduced)[0])    
    else:
        new_sc_score = np.mean(sil_scores) 
    sil_score = new_sc_score # will range from 0 to 1
    if penalized:  
        sil_score -= n_topics*topic_penalty 

    return (-1)*sil_score

def try_hierarchical_topic_reduction(model, num_topics, run_indexing=F):
    t0 = logger(f"Calling  try_hierarchical_topic_reduction...")
    if run_indexing:  # seems indexing does not speed-up the topic reduction... );
        if model.topic_index is None:
            model.index_topic_vectors()
        if model.document_index is None:
            model.index_document_vectors()
        if model.word_index is None:
            model.index_word_vectors()
    topics = model.hierarchical_topic_reduction(num_topics = num_topics, topic_size_weight = TS_WEIGHT)
    logger(f"Calling  try_hierarchical_topic_reduction...done", t0)

    return model, topics



def select_best_number_of_topics(text_col, model, penalized=T, verbose=F, sil_score_out_file=None, upper_bound=50):
    '''
    tf_idf = only used when using i2 metric
    '''


    n_topic_orig = model.get_num_topics(reduced=F)
    metric_vals = {} # k = num topics, v = metric val
    # 'small topic' threshold: 0.5% of the corpus size. Manuscript describes
    # this penalty as targeting topics under a fixed 1,000-document count;
    # this formula only equals that literal figure if len(document_ids) is
    # ~200k for the run in question (not independently confirmed here).
    small_topic_thresh = max(1, round(len(model.document_ids)  * 0.005 ))
    
    c_sc = compute_silhouette_score_from_topic_doc_scores(model, is_reduced=F, verbose=F)   
    n_sts = len([1 for x in model.get_topic_sizes(reduced=F)[0] if x <=small_topic_thresh]) 
    metric_vals[n_topic_orig] = min(c_sc + n_sts*SILHOUETTE_TOPIC_PENALTY, 0) # avoid useless values
    logger(f"STARTING select_best_number_of_topics, model has {n_topic_orig} topics originally (sil_score_penalized= {metric_vals[n_topic_orig]:0.3f} (non-pen = {c_sc:0.3f}))")
    logger(f"small_topic_thresh = {small_topic_thresh}, SILHOUETTE_TOPIC_PENALTY = {SILHOUETTE_TOPIC_PENALTY*penalized}")
    logger(f"n small topics current = {n_sts}")
    model.log_topic_words(reduced=F)
    # we can start from the smallest number of topics where the penalized score will be > 0
    start_max_n_small_topics = round(1/SILHOUETTE_TOPIC_PENALTY)
    start = min([n_topic_orig, start_max_n_small_topics, upper_bound])
    end = 2 # min number of topics / lower bound
    if start <= end:
        return start 
    best = n_topic_orig
    # this will be too slow when hierarchical clustering takes a long time
    bounds = [end, start]
    last_ub = start
    num_tops_used = set()
    mvals = sorted([(nt, sc) for nt,sc in metric_vals.items()], key = lambda x: x[1])
    while 1 == 1:
        last_ub = bounds[1]
        ub_div = 15
        if last_ub > 2000:
            ub_div = 3
        elif last_ub > 1000:
            ub_div = 4
        elif last_ub > 300:
            ub_div = 8

        step_size = max(1, round(last_ub/ub_div))

        num_tops = list(np.arange(bounds[0], last_ub, step_size))
        num_tops = [x for x in num_tops if x not in num_tops_used]
        if num_tops == []:
            break
        num_tops_used.update(num_tops)
        logger(f"Trying topic numbers {num_tops}, best so far = n_topics: {mvals[0][0]} score: {-mvals[0][1]:0.4f}")
        
        sil_scs = model.multiple_hierarchical_topic_reduction(
            nums_topics = list(num_tops),
            topic_penalty=SILHOUETTE_TOPIC_PENALTY,
            small_topic_thresh = small_topic_thresh,
            topic_size_weight = TS_WEIGHT )

        for n_topics, sil_sc, n_small_topics in sil_scs:
            metric_vals[n_topics] = sil_sc

        mvals = sorted([(nt, sc) for nt,sc in metric_vals.items()], key = lambda x: x[1])
        best = mvals[0][0]
        bounds = list(list(zip(*sorted(mvals[:3], key = lambda x: x[0])))[0])
        if len(bounds) != 3:
            break
        elif bounds[2] == last_ub:
            bounds.pop(2)
        else:
            bounds.pop(1)

    if verbose:
        logger(f"Based on penalized sil_sc, optimal number of topics = {best}")
        logger(f"with penalized sil_sc = {-metric_vals[best]:0.3f}")
        df_metric_vals = pd.DataFrame.from_records(list(metric_vals.items()))
        df_metric_vals.columns = ['n_topics', 'pen_sc_score' if penalized else 'sc_score']
        df_metric_vals['penalty_hp'] = SILHOUETTE_TOPIC_PENALTY
        df_metric_vals['variable'] = text_col
        if sil_score_out_file:
            try_save_df(sil_score_out_file.replace("$ID$", text_col), df_metric_vals)

    return best



# check_if_debugging(IS_DEBUG)