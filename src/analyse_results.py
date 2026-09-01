# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES
# 1. Analysses clusters found from best config from C_runner.py
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
full_filename_pkl = lambda fname : full_filename_ext(fname, subsampled_str, "pkl")
full_filename_tsv = lambda fname, batch_n: full_filename_ext(fname, f"{subsampled_str}b{batch_n}", "tsv") 
full_filename_xlsx = lambda fname: full_filename_ext(fname, subsampled_str, "xlsx")
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
# Boilerplate end


infile = full_filename_pkl(ns.infile)
GEN_WORDCLOUDS = ns.plot_wordclouds
GoF_metric = ns.goodness_of_fit_metric


import numpy as np
import seaborn as sns
import pandas as pd
from try_stepmix import StepMixBICScore
from stepmix.utils import get_mixed_descriptor
from tensorflow.keras.models import Model
from sklearn.metrics import rand_score
import matplotlib.pyplot as plt
from scipy.stats import sem
import os
from Top2Vec import Top2Vec
from analyse_results_util import get_nested_gmm_clsvars

# full_filename_pkl
GOODNESS_OF_FIT_METRIC = GoF_metric

pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)

idx = 0 if SUBSAMPLE_DATA else 1

DIM_REDUC_TECHNIQUE = {
                "autoencoder": "dim_red__autoenc",
                "expert_knowledge" : "dim_red__expert_knowledge" 
                }
DIM_REDUC_TECHNIQUE = DIM_REDUC_TECHNIQUE['expert_knowledge']
#emb_model_shortnames = [v['short_name'] for v in EMBEDDING_MODEL_METADATA.values()]
embedding_model_used =  'd2v' #[x for x in emb_model_shortnames if f"_{x}_" in infile][0]

assert pickle_exists(infile)
infiles_code_cats = [f'icpc_cats{i}.json' for i in [1,2]]

infile_scale_maxabs_lookup = full_filename_pkl(ns.infile_scale_maxabs_lookup) # cluster_fs1_crs_scaled_scale_multipliers_SUBSAMPLED 
infile_scale_robust_lookup = full_filename_pkl(ns.infile_scale_robust_lookup) #cluster_fs1_df_scaled_scale_multipliers_SUBSAMPLED
infile_code_lookups = full_filename_pkl(ns.infile_fs1_ohe_lookup) # cluster_fs1_lookup_dicts
infile_topic_model =  full_filename_pkl(ns.infile_t2v_model)  # pats_dict_merged_text_emb3

assert pickle_exists(infile_scale_maxabs_lookup)
assert pickle_exists(infile_code_lookups)
assert pickle_exists(infile_topic_model)

outfile = full_filename_xlsx(ns.outfile)
outfile_adjudication =  full_filename_pkl(f"{ns.outfile}_adj")
outfile_adjudication_random_sample = full_filename_pkl(f"adj_rnd_sample{ns.outfile}")

# define constatns
icpc_letter_desc = {
    "A" : "General/unspecified",
    "B" : "Blood/blood-forming organs/immune system",
    "D" : "Digestive",
    "F" : "Eye",
    "H" : "Ear",
    "K" : "Cardiovascular",
    "L" : "Musculoskeletal",
    "N" : "Neurological",
    "P" : "Psychological",
    "R" : "Respiratory",
    "S" : "Skin",
    "T" : "Endocrine/metabolic/nutritional",
    "U" : "Urological",
    "W" : "Pregnancy/childbirth/family planning",
    "X" : "Female genital system and breast",
    "Y" : "Male genital system",
    "Z" : "Social problems",
    "-" : "NEC"
    }
icpc_cat_desc = { 
        "cats0" : "ICPC x Components 1-7 (per letter)",
        "cats1" : "ICPC x CV disease or sympt" 
        }
icpc_subcat_desc = {
    "cats0": lambda subcat: "(" + icpc_letter_desc[subcat[0]] + ") " + 
    {"1" : "Complaints",
    "2" : "diagnostic, screening and preventive procedures",
    "3" : "medication, treatment and procedures",
    "4" : "test results",
    "5" : "administrative",
    "6" : "referrals and other reasons for encounter",
    "7" : "diseases"}[subcat[1:]],
    "cats1": lambda subcat: 
    {"1" :	"CV disease diagnoses",
    "2" :	"high CV risk",
    "3" :	"Diabetes Marchal",
    "4" :	"Other relevant CV morbidity CVRM",
    "5" :	"Family history CVRM",
    "6" :	"Presenting symptoms and risk factors included in the TARGET-HF paper ",
    "7" :	"Other symptoms/disease associated with HF, AF or VHD"}[subcat[1:]]
}

# local functions
def _prepare_records_to_adjudicate(X_labelled, Y_pred, target_cluster):
    # get actual records of patients in cluster for adjudication
    c_idxs = [i for i,c in enumerate(Y_pred) if c == target_cluster]
    p_ids = [x for i,x in enumerate(vals(X_labelled['id'])) if i in c_idxs]
    p_ids_str = [convert_pat_id_float2str(x) for x in p_ids]
    full_data_dict = read_pickle(f'pats_dict_merged{subsampled_str}.pkl')
    to_adj_data_dict = { k:v for k,v in full_data_dict.items() if k in p_ids_str}
    len(to_adj_data_dict)
    save_pickle(outfile_adjudication, to_adj_data_dict)

def _prepare_records_to_adjudicate_random_sample(repeat = F, sample_size = 1000, stratify = F):
    # get actual records of patients in cluster for adjudication
    full_data_dict = read_pickle(f'pats_dict_merged{subsampled_str}.pkl')
    ks = list(full_data_dict.keys())
    to_adj_data_dict = {}
    selected_ks = []
    for i in range(sample_size):
        available_ks = ks 
        if not repeat:
            available_ks = list(set(available_ks) - set(selected_ks))

        c_k = available_ks[random.randint(0, len(available_ks))]
        selected_ks += [c_k]
    
    to_adj_data_dict = { k:v for k,v in full_data_dict.items() if k in selected_ks}
    
    len(to_adj_data_dict)
    save_pickle(outfile_adjudication_random_sample, to_adj_data_dict)

def _describe_clusters(X_labelled, translations, prev_per_clust):
    protected_cols = ['id', VAR_FOLLOW_UP_DATE, 'cluster_id', 'deceased_1']
    unprotected_cols = try_sd(cns(X_labelled), protected_cols)
    #assert try_sdui(scale_robust_scaler.feature_names_in_, unprotected_cols ) == []
    unprotected_cols = scale_robust_scaler.feature_names_in_ # order matters for the scaler

    X = X_labelled[try_sd(cns(X_labelled), protected_cols)]
    missing_cols = try_sdui(scale_robust_scaler.feature_names_in_, cns(X) )
    for missing_col in missing_cols:
        X[missing_col] = 0
    unprotected_cols = scale_robust_scaler.feature_names_in_
    X = X[unprotected_cols]
        

    # for each variable used to define the clusters. make a nice row describing the variable, write to excel after 
    X = scale_robust_scaler.inverse_transform(X)
    X_labelled[unprotected_cols] = pd.DataFrame(X)
    X_labelled = X_labelled.drop(missing_cols, axis = 1)
    unprotected_cols = try_sd(unprotected_cols, missing_cols)
    del X
    for c_var in unprotected_cols:
        if c_var in scale_maxabs_lookup_d:  
            c_scale = scale_maxabs_lookup_d[c_var]
            X_labelled[c_var] = X_labelled[c_var]*c_scale

    translations_d = {k:v for k,v in translations}
    n = nrow(X_labelled)
    uniq_clusters = uniq(X_labelled['cluster_id'])
    vars = [x for x in  cns(X_labelled) if x not in ['id', 'cluster_id']]
    res = None # build-up a dataframe with significance score of each variable per cluster
    for c_var in vars: # c_var = vars[42]
        # logger(f"describe_clusters for {c_var}")
        c_res = [c_var] 
        global_mean = np.mean(vals(X_labelled[c_var]))
        global_std = np.std(vals(X_labelled[c_var]))
        global_sem = sem(vals(X_labelled[c_var]))
        # global_ci95_ub = global_mean + 1.96*global_sem
        # global_ci95_lb = global_mean - 1.96*global_sem
        c_res.append(global_mean)
        c_res.append(global_std)
        c_res.append(global_sem)
        for c_cluster in uniq_clusters: # c_cluster = uniq_clusters[0]
            # logger(f"describe_clusters for {c_var} , cluster {c_cluster}/{len(uniq_clusters)}")
            c_rows = X_labelled[X_labelled['cluster_id'] == c_cluster]
            c_mean = np.mean(vals(c_rows[c_var]))
            c_std = np.std(vals(c_rows[c_var]))
            c_sem = sem(vals(c_rows[c_var]))
            c_sigma = abs(global_mean - c_mean) / global_sem
            c_res.append(c_mean)
            c_res.append(c_std)
            c_res.append(c_sigma)

        c_res = pd.DataFrame(c_res).T
        c_res.columns = ['var', 'global_mean', 'global_std', 'global_sem'] + [f'c_{x}_{sfx}' for x in uniq_clusters for sfx in ['_mean', '_std', '_sigma']]
        res = pd.concat([res, c_res])    
    min_sigma = 0
    res_pretty = None
    
    txt_vars_and_t_nums = extract_txt_vars_and_t_nums_from_column_names(cns(X_labelled), compact=T) # WIP: get top N words as description for topic vars
    txt_vars = [x for x in cns(X_labelled) if try_regex(TOPIC_DISTANCE_COLUMN_REGEX,x)]
    topic_words_d = {}
    for txt_var, t_nums in txt_vars_and_t_nums:  # txt_var, t_nums = txt_vars_and_t_nums[0]
        model = None
        for t_num in t_nums: # t_num = t_nums[0]
            logger(txt_var)
            if model is None: #only load once per text var
                model = get_t2v_model_for_txt_var(txt_var)  # takes several minutes (~1-5min)


            topic_words = model.topic_words_reduced if model.hierarchy is not None else model.topic_words
            relevant_vars = [txt_vars[i] for i in range(len(txt_vars)) if txt_var in txt_vars[i] and f'_t{t_num}_' in txt_vars[i] ]
            assert len(relevant_vars) > 0
            for rv in relevant_vars:
                topic_words_d[rv] = topic_words[int(t_num)]

    for j, c_clust in enumerate(uniq_clusters): # c_clust = uniq_clusters[1]
        print(f"cluster {c_clust}")
        res = res.sort_values([f'c_{c_clust}__sigma'], ascending = [F])
        c_res = res[res[f'c_{c_clust}__sigma'] >= min_sigma]
        c_res = c_res[c_res[f'c_{c_clust}__sigma'].notna()]
        c_X = X_labelled[X_labelled['cluster_id'] == c_clust] #vals(X_labelled[c_var])
        for i in range(nrow(c_res)): # i = 0
            c_row = c_res.iloc[i]
            c_var = c_row['var']
            c_mean = c_row[f'c_{c_clust}__mean']
            g_mean = c_row['global_mean']
            g_sem = c_row['global_sem']
            c_sigma = c_row[f'c_{c_clust}__sigma']
            if c_var == 'age_days':
                c_mean /= 365
                g_mean /= 365

            is_time_var = try_regex('^\d+_\d+_\w+_\w+', c_var) 
            is_cat_var = try_regex('_\d+$', c_var) or try_regex('_(lo|med|hi)$', c_var) 
            is_icpc = 'icpc' in c_var
            c_var_time_beg_end = c_var.split("_")[0:2] if is_time_var else [None,None]
            c_cat = "_".join(c_var.split("_")[-2:-1]) if 'icpc' in c_var else None
            c_subcat = c_var.split("_")[-1] if 'icpc' in c_var else None
            icpc_codes = mapping_cat_to_icpc["_".join(c_var.split("_")[-2:])] if 'icpc' in c_var else None
            simple_name_start_idx = 3 if is_time_var else 0
            simple_name_end_idx = -1 if is_cat_var else None
            var_simple_name = "_".join(c_var.split("_")[simple_name_start_idx:simple_name_end_idx])

            var_desc = one_hot_decode_varname(c_var, lookup_d)
            
            if is_cat_var:
                var_desc += c_var.split("_")[-1] + " "
            elif is_icpc:
                var_desc = f"{icpc_subcat_desc[c_cat](c_subcat)} (from {icpc_cat_desc[c_cat]})" 
            
            if c_var in topic_words_d:
                var_desc = ", ".join(topic_words_d[c_var])

            new_row = { "cluster" : c_clust,
                        "var" : var_simple_name,
                        "time_period": f"{c_var_time_beg_end[0]}-{c_var_time_beg_end[1]}",
                        "var_desc" : var_desc,
                        "cluster_mean": c_mean,
                        "global_mean": g_mean,
                        "clust_preval": prev_per_clust[j],
                        "clust_mass": (100*nrow(c_X)) / nrow(X_labelled),
                        "global_sem": g_sem,
                        "sigma" : c_sigma
                        }
            c_df = pd.DataFrame(new_row.values()).T
            c_df.columns = new_row.keys()
            res_pretty = pd.concat([res_pretty, c_df])
    
    res_pretty.to_excel(f"excel/{outfile}")

def extract_txt_vars_and_t_nums_from_column_names(vars_selected, compact=F):
    txt_vars_selected = [x for x in vars_selected if try_regex(TOPIC_DISTANCE_COLUMN_REGEX,x)]
    txt_vars = [x[:re.search('_t\\d+_.*_dist$', x).start()] for x in txt_vars_selected ]
    t_nums = [x[re.search('_t\\d+_.*_dist$', x).start() + 2: re.search('_t\\d+_', x).end()-1 ] for x in txt_vars_selected ]

    txt_vars_and_t_nums = []
    if compact: # all topic nums under same var 
        for tx_v in oset(txt_vars): # oset to maintain order
            idxs = try_multiindex(txt_vars, tx_v)
            c_t_ns = [t_nums[i] for i in idxs]
            txt_vars_and_t_nums += [(tx_v, c_t_ns)]
    else: # topic_var/topic_num  pairs
        txt_vars_and_t_nums = [ (txt_vars[i], int(t_nums[i])) for i in range(len(t_nums))]
        txt_vars_and_t_nums = list(set(txt_vars_and_t_nums))

        
    return txt_vars_and_t_nums

def get_t2v_model_for_txt_var(txt_var):
    model = read_pickle(topic_model_filenames[txt_var])['model']
    if embedding_model_used != "d2v":
        if model.embed is None:
            emb_model_fullname = [k for k,v in EMBEDDING_MODEL_METADATA.items() if v['short_name'] == embedding_model_used][0]
            is_callable = EMBEDDING_MODEL_METADATA[emb_model_fullname]['is_callable']
            embedding_model_for_top2vec = eval(emb_model_fullname) if is_callable else emb_model_fullname
            model.embedding_model = embedding_model_for_top2vec
        model._check_model_status()
    return model

# read input GMM cluster model
tmp = read_pickle(infile)
tmp.keys() # model, X, Y, params, cluster_metrics
model = tmp['model']

is_nested =  '_12_nested_gmm_output' in infile


Y = tmp['Y']
X = tmp['X']
comp_vars = []
if is_nested:
    X, comp_vars = get_nested_gmm_clsvars(X, model, comp_gmms = ['_7_other_nested_gmm_output.pkl', '_8_icpc_nested_gmm_output.pkl',
                                '_10_atc_nested_gmm_output.pkl', '_12_txt_nested_gmm_output.pkl'] 
                                )

gmm_cluster_metrics = tmp['cluster_metrics']
gmm_params = tmp['params']
del tmp

base_inci = 100*sum(Y)/len(Y)
logger(f"Outcome nPos = {sum(Y)} incidence = {base_inci:0.2f}%")
protected_cols = ['id', VAR_FOLLOW_UP_DATE, 'deceased_1']
unprotected_cols = try_sd(cns(X), protected_cols)
X_ids = X[protected_cols]
vars_used = list(model.feature_names_in)
X_in = X[vars_used]
topic_model_filenames = read_pickle(infile_topic_model)['models'] 
if GEN_WORDCLOUDS:
    vars_selected = vars_used
    for plot_only_topics_vars_used in [F, T]:
        txt_vars_and_t_nums = extract_txt_vars_and_t_nums_from_column_names(vars_selected, compact=T)
        for ty_version in [F]: 
            for txt_var, t_nums in txt_vars_and_t_nums:
                t2vModel = None
                subsampledpath = "/subsampled" if SUBSAMPLE_DATA else ""
                logger(txt_var)
                if t2vModel is None: #only load once per text var
                    t2vModel = get_t2v_model_for_txt_var(txt_var) 
                did_reduce = t2vModel.hierarchy is not None
                if not plot_only_topics_vars_used:
                    t_nums = list(range(t2vModel.get_num_topics(reduced=did_reduce)))
                t2vModel.generate_topic_wordcloud = Top2Vec.generate_topic_wordcloud
                pp_doc_file = f"{txt_var}_documents_preprocessed_.pkl"
                assert pickle_exists(pp_doc_file)
                flat_docs = read_pickle(pp_doc_file)['flat_docs'] if ty_version else None
                fig_nrows = round_up(len(t_nums)/2)
                fig, axes = plt.subplots(nrows = fig_nrows, ncols = 2 , figsize = (16, 6*fig_nrows)  )
                axes = axes.ravel()
                for i, t_num in enumerate(t_nums):
                    
                    #fig_filepath = f'plots/analyse_results{subsampledpath}/grid_wordcloud_{txt_var}_t{t_num}_ty{ty_version}.png'
                    t2vModel.generate_topic_wordcloud(t2vModel, topic_num=int(t_num) , reduced=did_reduce, ty_version =ty_version, flat_docs = flat_docs,
                    ax = axes[i])
                    #logger(f"Calling save for {fig_filepath}...")
                    #plt.savefig(fig_filepath)
                    #plt.close()
                fig_filepath = f'plots/analyse_results{subsampledpath}/grid_wordcloud_{txt_var}_selected_{plot_only_topics_vars_used}_ty{ty_version}.svg'
                plt.tight_layout()
                plt.savefig(fig_filepath, dpi = 300, bbox_inches = "tight")
                plt.close(fig)

                





lookup_d = read_pickle(infile_code_lookups)
scale_maxabs_lookup_d = read_pickle(infile_scale_maxabs_lookup)
scale_robust_scaler = read_pickle(infile_scale_robust_lookup)
mapping_icpc_to_cat, mapping_cat_to_icpc = get_icpc_cats(infiles_code_cats)


# assert try_sdui(vars_used, unprotected_cols) == []

X_in= X_in[vars_used]
Y_pred = model.predict_class(X_in)
if len(set(Y_pred)) != model.n_components:
    logger('warning: not all clusters from GMM have members, usually not a good sign...')
#assert len(set(Y_pred)) == model.n_components
vars_used += comp_vars
X = X[vars_used]
incis, supps, npos, mass = get_clusters_inc_supp(Y, Y_pred, verbose=T)

model_params = model.get_mm_df().round(2).reset_index()

# initialize translations dict for mapping from fs to is
translations = []
vars_to_desc = comp_vars if is_nested else set(model_params['variable'])
for param in vars_to_desc: # param = 'postal_code_10' 
    l_v = one_hot_decode_varname(param, lookup_d)    
    is_icpc = 'icpc' in l_v
    if is_icpc and not is_nested:
        cur_cat = "_".join(l_v.split("_")[-2:])
        translations.append((param, mapping_cat_to_icpc[cur_cat]))
    else:
        translations.append((param,l_v))
    

mm_df = model.get_mm_df()
params_df = model.get_parameters_df()
cw_df = model.get_cw_df()
# go over mm_df / per row / each row is a variable  /  each column is a cluster_id 
# say we are at var_1 , say it has coeff 0.2, 0.25, 0.5, 0.02 in clusters [1-4]
# 
# we say then that higher values of var_1 were more strongly associated with members of cluster 3
# but what is a good way to describe the clusters....

# take a more empirical approach
# take all the patients and label them per cluster
# then look at each variable (its mean/std) within a cluster, and also globally 
Y_pred_df = pd.DataFrame(Y_pred)
Y_pred_df.columns = ['cluster_id']
X_labelled = pd.concat([X_ids, X, Y_pred_df], axis = 1)

cluster_ids = list(set(Y_pred))

incis, supps, npos, mass = get_clusters_inc_supp(Y, Y_pred, verbose=T) # is_cluster_interesting(incidence, mass, npos, base_incidence):
interesting_clusters = [cluster_ids[i] for i,inci in enumerate(incis) if is_cluster_interesting(inci, mass[i], npos[i], base_inci) or (SUBSAMPLE_DATA and i < 2) ] 
#target_cluster, cluster_inci = [(cluster_ids[i],inci) for i,inci in enumerate(incis) if is_cluster_interesting(inci, mass[i], npos[i], base_inci)][0]
logger(f"Found {len(interesting_clusters)} clusters of interest ({interesting_clusters})")



_describe_clusters(X_labelled, translations, incis)
#_prepare_records_to_adjudicate(X_labelled, Y_pred, target_cluster)
logger('DONE!')
print('DONE!')



import sys
sys.exit(0)
# try to distil cluster further
tmp_data = read_pickle(infile_XY)

Y_pred = model.predict_class(X.drop('id', axis=1))
distil_file = 'distill_medrobnl.pkl'
distil_o = tmp_data

distil_idxs = [i  for i,v in enumerate(Y_pred) if v == target_cluster]

distil_X = X_labelled[X_labelled['cluster_id'] == target_cluster]
distil_Y = [y for i,y in enumerate(Y) if i in distil_idxs]
distil_Y = np.array(distil_Y).reshape(-1,1)

distil_o['X'] = distil_X.drop(columns=['cluster_id'], axis=1)
distil_o['Y'] = distil_Y
distil_o['fit_cns'] = cns(distil_o['X'])
distil_o['Y_prev'] = np.squeeze(np.asarray( distil_o['Y_prev']) )
distil_o['Y_prev'] = [y for i,y in enumerate(distil_o['Y_prev']) if i in distil_idxs]
distil_o['Y_prev'] = np.array(distil_o['Y_prev']).reshape(-1,1)

distil_o['Y_earlier'] = np.squeeze(np.asarray( distil_o['Y_earlier']) )
distil_o['Y_earlier'] = [y for i,y in enumerate(distil_o['Y_earlier']) if i in distil_idxs]
distil_o['Y_earlier'] = np.vstack(distil_o['Y_earlier'])

distil_o['outcome_of_interest']
save_pickle(distil_file, distil_o)

from GaussMMStepMix import run_it as run_LCA 

hp_params = {"n_clusters_to_try" : [ [5, 10, 20, 40, 60] if not SUBSAMPLE_DATA else list(range(5,10))] , #list(range(3,21))
             "use_outcome" : [T, F] , "use_only_text_icpc" : [F]}
 

run_LCA()

logger("DONE")
# next idea: see how many of the patients matching the vars from time period 1, end up evolving into patients that match period 2