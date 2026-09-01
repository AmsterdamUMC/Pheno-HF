# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES 
# Performs variable selection using dataset produced by previous step
# Saves dataset with reduced feature space from variables selected
# Todo: consider bias reduction (cross-validation/boostrap)
'''
)

# Boilerplate start
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]

subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
full_filename_pkl = lambda fname : f"{fname}{subsampled_str}.pkl"
full_filename_tsv = lambda fname, batch_n: f"{fname}{subsampled_str}b{batch_n}.tsv"
plot_hist_vals = lambda vals, **kwargs: try_plot_hist_vals(vals, outpath="plots/VariableSelectionBoruta/", subsampled=SUBSAMPLE_DATA, **kwargs)
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
delete_pickle = lambda f: try_delete_pickle(f, subsampled=SUBSAMPLE_DATA)
# Boilerplate end

import dask.dataframe as dd
from os import environ
environ['OMP_NUM_THREADS'] = "4"
environ['OPENBLAS_NUM_THREADS'] = "4"

MIN_N_VARS = 20 if not SUBSAMPLE_DATA else 3 # ideally we should not have to use this... all selected vars should be already present in rank 1

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import xgboost as xgb
from kneed import KneeLocator
# from boruta import BorutaPy

logger = get_default_logger_fn(__file__, override=False) # if called from a runner, use the runners logfile
start_time = logger("Start running...")

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

def find_knee(importances, conv_ws=5, varnames=[], plot = F):
    importances_raw = importances
    # importances = importances_raw
    importances = np.convolve(importances_raw, np.ones(conv_ws)/conv_ws, mode='valid')
    pesimistic_kneedle = KneeLocator(range(len(importances)), importances, curve="convex", direction="decreasing", online=F)
    optimistic_kneedle = KneeLocator(range(len(importances)), importances, curve="convex", direction="decreasing", online=T)
    middle_ground = int((pesimistic_kneedle.knee + optimistic_kneedle.knee)/2)
    if plot:
        plt.clf() # plt.close()
        plt.figure(6)
        top_n_feature_imps = 100
        plt.scatter(range(len(importances_raw[:top_n_feature_imps])), np.log(importances_raw[:top_n_feature_imps]) )
        if varnames != []:
            for i, txt in enumerate(varnames[:middle_ground]):
                plt.text(i, importances_raw[i]-0.0015, txt, fontsize=5, verticalalignment='center')

        plt.savefig(f'plots/VariableSelectionBoruta/rf_logimportances_top0-{top_n_feature_imps}-conv-ws=0.png', format='png', dpi=300)
        plt.clf() # plt.close()
        plt.figure(7)
        plt.scatter(range(len(importances_raw[:50])), importances_raw[:50])
        if varnames != []:
            for i, txt in enumerate(varnames[:middle_ground]):
                plt.text(i, importances_raw[i]-0.0015, txt, fontsize=5, verticalalignment='center')
        plt.savefig('plots/VariableSelectionBoruta/rf_importances_top0-50-conv-ws=0.png', format='png', dpi=300)

    return middle_ground
    # plt.clf() # plt.close()
    # plt.figure(2)
    # top_n_feature_imps = 75
    # plt.scatter(range(len(importances[:top_n_feature_imps])), importances[:top_n_feature_imps])
    # plt.savefig(f'plots/VariableSelectionBoruta/rf_importances_top0-{top_n_feature_imps}-conv-ws={conv_ws}.png', format='png', dpi=300)
    # plt.clf() # plt.close()
    # plt.figure(2)
    # plt.scatter(range(len(importances_raw[:50])), importances_raw[:50])
    # plt.savefig('plots/VariableSelectionBoruta/rf_importances_top0-50-conv-ws=0.png', format='png', dpi=300)
    # return optimistic_kneedle.knee
    # pesimistic_kneedle = KneeLocator(range(len(importances)), importances, curve="convex", direction="decreasing", online=F)
    # return pesimistic_kneedle.knee

def log_topn_vars(vars, importances, top_n=10):
    nvars = len(vars)
    top_n = min(nvars, top_n)

    logger(f'top {top_n} features = ')
    try_print_list(vars[:top_n], logger)
    logger(f'with importances {top_n} importances  = {try_round(importances[:top_n], 5)}')

    logger(f'bottom {top_n} features = ')
    try_print_list(vars[-top_n:], logger)
    logger(f'bottom {top_n} importances ={try_round(importances[-top_n:], 5)}')
    
def eval_model_var_selection(model, X, y, cns_X, cv_nfolds=1):
    # evaluate by using AUC of model fitted using vars selected from current var selection strategy
    if type(X) != pd.DataFrame:
        X = pd.DataFrame(X, columns=cns_X)
    selected_vars = []
    if model is not None:
        importances = model.feature_importances_
        idxs = importances.argsort()[::-1]
        importances = sorted(importances, reverse=T)

        smallest_nvars = find_knee(importances)
        thresh_idx = smallest_nvars
        var_idxs = idxs[:thresh_idx]
        selected_vars = [X.columns[i] for i in var_idxs]
    else:
        selected_vars =   cns_X  
    # aucs = run_PM_test(X, y, cns(X), model_type="decision_tree")
    aucs = run_PM_test(X, y, selected_vars, model_type="decision_tree", cv_nfolds=cv_nfolds)
    return aucs


def run_rf(
    X,
    Y,
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    max_leaf_nodes=10,
    cv=F,
    min_n_other_vars = 10,
    min_n_atc_vars = 10,
    min_n_txt_vars = 10,
    min_n_icpc_vars = 10
    ):
    t0 = logger("Starting RandomForestClassifier ...")
    logger(f"USING n_estimators = {n_estimators}")
    logger(f"USING max_depth = {max_depth}")
    logger(f"USING min_samples_split = {min_samples_split}")
    logger(f"USING max_leaf_nodes = {max_leaf_nodes}")
    random_state = 873458311
    rf = RandomForestClassifier(n_estimators= n_estimators,
                                random_state=random_state,
                                n_jobs=-1,
                                class_weight='balanced_subsample', 
                                max_depth=max_depth,
                                min_samples_split = min_samples_split,
                                max_leaf_nodes = max_leaf_nodes,
                                )

    Y = Y.values.ravel() 
    if cv: # is this running as part of a cross-validation?
        model_obj = {
                    
                        "init": lambda kwargs={}: RandomForestClassifier(**kwargs),
                        "grid_params": { 
                            "n_estimators" : [n_estimators],
                            "random_state": [random_state],
                            "max_depth": [max_depth],
                            "min_samples_split" : [min_samples_split],
                            "max_leaf_nodes": [max_leaf_nodes],
                            "n_jobs" : [-1],
                            "class_weight": ["balanced"] # hardcoded !
                        } ,
                        "sample_weight"  : F,
                    }
        
        eval_fn = lambda *args : eval_model_var_selection(*args, cns_X = cns(X))    
        tmp = fit_pred_model_test(X, Y, cns(X), model_obj, eval_model_fn=eval_fn, cv_nfolds=4)
        mean_val_auc = np.mean(tmp)
        std_val_auc = np.std(tmp)
        return mean_val_auc,std_val_auc

    rf.fit(X, Y)
    importances = rf.feature_importances_
    idxs = importances.argsort()[::-1]
    importances = sorted(importances, reverse=T)
    plot_hist_vals(importances, outfile="hist_rf_importances")
    varnames = [X.columns[i] for i in idxs] # try_print_list(varnames, logger) # logger("Features ranked by importance")
    # varnames_orig = varnames # todo: hack
    smallest_nvars = find_knee(importances, varnames = varnames, plot=T)
    rank_atc_vars = [i for i,v in enumerate(varnames) if try_regex('_atc_', v)]
    rank_txt_vars = [i for i,v in enumerate(varnames) if try_regex('_text_', v)]
    rank_icpc_vars = [i for i,v in enumerate(varnames) if try_regex('_icpc_', v)]
    all_non_other_vars = rank_atc_vars + rank_txt_vars + rank_icpc_vars
    all_other_vars = [varnames[i] for i in try_sd(list(range(len(varnames))), all_non_other_vars)]
    if min_n_atc_vars+ min_n_txt_vars + min_n_icpc_vars == 0:
        all_other_vars = varnames[:min_n_other_vars]
    rank_other_vars = [i for i,v in enumerate(varnames) if v in all_other_vars] 
    rank_vars_keep = rank_atc_vars[:min_n_atc_vars] + rank_txt_vars[:min_n_txt_vars] 
    rank_vars_keep = rank_vars_keep + rank_icpc_vars[:min_n_icpc_vars] + rank_other_vars[:min_n_other_vars] 
    rank_vars_keep = sorted(rank_vars_keep)
    included_vars = [varnames[i] for i in rank_vars_keep]
    non_included_vars = [varnames[i] for i in range(len(varnames)) if i not in rank_vars_keep]
    varnames = included_vars + non_included_vars
    # varnames  = varnames_orig # todo: hack
    thresh_idx = max(smallest_nvars, len(included_vars) )
    logger(f"Knee determined top {smallest_nvars} vars, but we will use {thresh_idx} vars")
    logger(f"Variables ordered by feature importance (from most important to least):")
    try_print_list(varnames, logger)
    
    rf_vars = varnames[:thresh_idx]
    importances = importances[:thresh_idx] # todo: not correct
    log_topn_vars(rf_vars, importances, int(thresh_idx/2)+1)

    auc = np.mean(eval_model_var_selection(None, X, Y, rf_vars, cv_nfolds=4))
    return auc, rf_vars



def run_xgb(
    X,
    Y,
    n_estimators=100,
    max_depth=5,
    min_child_weight=1,
    colsample_bytree=1,
    cv=F,
    ):


    t0 = logger("Starting XGBClassifier ...")
    logger(f"USING n_estimators = {n_estimators}")
    logger(f"USING max_depth = {max_depth}")
    logger(f"USING min_child_weight = {min_child_weight}")
    logger(f"USING colsample_bytree = {colsample_bytree}")
    random_state = 873458311

    xgbm = xgb.XGBClassifier(n_estimators= n_estimators,
                                random_state=random_state,
                                n_jobs=-1,
                                scale_pos_weight=1, 
                                subsample=0.9,
                                max_depth=max_depth,
                                min_child_weight = min_child_weight,
                                colsample_bytree = colsample_bytree
                                )
    
    Y = Y.values.ravel() 
    if cv:
        model_obj = {
                    
                        "init": lambda kwargs={}: xgb.XGBClassifier(**kwargs),
                        "grid_params": { 
                            "n_estimators" : [n_estimators],
                            "random_state": [random_state],
                            "max_depth": [max_depth],
                            "min_child_weight" : [min_child_weight],
                            "colsample_bytree": [colsample_bytree],
                            "n_jobs" : [-1],
                            "scale_pos_weight": [1] ,# hardcoded !
                            "subsample": [0.9],
                        } ,
                        "sample_weight"  : F,
                    }
        eval_fn = lambda *args : eval_model_var_selection(*args, cns_X = cns(X))    
        tmp = fit_pred_model_test(X, Y, cns(X), model_obj, eval_model_fn=eval_fn)
        mean_val_auc = np.mean(tmp)
        std_val_auc = np.std(tmp)
        return mean_val_auc,std_val_auc

    xgbm = xgbm.fit(X, Y)
    importances = xgbm.feature_importances_
    idxs = importances.argsort()[::-1]
    importances = sorted(importances, reverse=T)

    smallest_nvars = find_knee(importances)
    thresh_idx = smallest_nvars
    xgb_var_idxs = idxs[:thresh_idx]
    xgb_vars = [X.columns[i] for i in xgb_var_idxs]
    importances = importances[:thresh_idx]
    logger(f'Using xgbm , selected {len(xgb_vars)} features')
    log_topn_vars(xgb_vars, importances, 10)
    auc = np.mean(eval_model_var_selection(xgbm, X, Y, xgb_vars, cv_nfolds=4))
    return auc, xgb_vars


def run_it(infile, hp_params, outfile, override_res_file = T, use_text_vars=T):
    hp_params_to_try = expand_hp_params(hp_params)
    
    run_function = run_rf #run_xgb # run_rf
    # not really useful to select variables like these...
    protected_cols = [
        'id',
        VAR_FOLLOW_UP_DATE, 
        'deceased_1'] 
    # or these..
    junk_cols = [
        't_death', 
        '0_24_tw_atc_code_',
        '0_24_tw_icpc_ep_',
        '0_24_tw_icpc_s_',
        '0_24_tw_icpc_o_',
        '0_24_tw_icpc_e_',
        '0_24_tw_icpc_p_'
    ]
    out_files = []
    params_scores = []
    t0 = logger("Starting run_it ...")
    i = -1
    log_once = lambda s: logger(s) if i == 0 else None
    best_score = {}
    model_inputs = read_pickle(infile)
    X = model_inputs['X']
    X = X[try_sd(cns(X), junk_cols)]
    logger("remove duplicate columns")
    X = X[try_sd(cns(X), ['t_VHD', 't_AF'])]

    logger('adjust txt vars lower bound')

    txt_vars = try_regex_multi(cns(X), '_text_')
    if use_text_vars:
        for c in txt_vars:
            cvals = vals(X[c])
            is_needing_adjustment = min([v for v in cvals if v != 0]) > 0.5 and max(cvals) <= 1
            if is_needing_adjustment:
                logger(f"adjusting {c}...")
                cvals = [ max(2*v-1, 0) for v in cvals ]
                X[c] = cvals
    elif not use_text_vars:
        X = X[try_sd(cns(X), txt_vars)]
        

    logger(f"Compute correlation matrix to remove highly collinear variables")
    ddf = dd.from_pandas(X[try_sd(cns(X), protected_cols)], npartitions=14)
    corr_matrix = ddf.corr().compute()
    abs_corrs = corr_matrix.abs()

    upper = abs_corrs.where(np.triu(np.ones(abs_corrs.shape), k=1).astype(bool))
    corr_upper_thresh = 0.95
    corrs_d_max = {}
    corrs_d = {}
    rownames = vals(upper.index)
    for c in cns(upper):
        c_corrs = [x if not np.isnan(x) else -1 for x in vals(upper[c]) ]
        if c_corrs == []:
            c_corrs = [-1]
        corrs_d_max[c] = max(c_corrs)
        c_corrs_filtered = { k:v for k,v in zip(rownames,c_corrs) if v >= corr_upper_thresh }
        if c_corrs_filtered != {}:
            corrs_d[c] = c_corrs_filtered
            logger(f'Dropping column {c}. Reason: more than {corr_upper_thresh} correlation with: { ";".join(c_corrs_filtered.keys()) }')

    to_drop = [c for c in upper.columns if corrs_d_max[c] >= corr_upper_thresh]
    find_shadows = lambda c: list(corrs_d[c].keys())
    special_cols = [] #special_cols = ['text__t14_', 'text__t19_', 'text__t20_', 'text__t28_', 'text__t34_']
    to_drop_special = [ [to_drop[i]] if to_drop[i] not in special_cols else find_shadows(to_drop[i]) for i in range(len(to_drop))]
    to_drop_special = list(set([i for o in to_drop_special for i in o]))

    if to_drop != []:
        logger(f"Going to drop {len(to_drop)} columns because of high multicollinearity")
        # X = X.drop(to_drop_special, axis=1)
        X = X.drop(to_drop, axis=1)

    fit_cns = try_sd(cns(X), protected_cols)

    in_X = X[fit_cns]
    Y = model_inputs['Y']
    Y = pd.DataFrame(Y)
    outcome_of_interest = model_inputs['outcome_of_interest'][1]
    logger(f"{len(fit_cns)} Input columns:=")
    try_print_list(sorted(fit_cns), logger) 
    if len(hp_params_to_try) > 1:
        logger("Try all param configurations using CV-auc to determine best set of params")
        for c_params_to_try in hp_params_to_try:
            i+=1
            hp_params_str = "_".join([ f"{k[:3]}{k[-4:]}{c_params_to_try[k]}" for k in hp_params.keys()])
            c_outfile = full_filename_pkl(f"{outfile}{hp_params_str}")
            if not override_res_file and pickle_exists(c_outfile):
                logger(f"SKIP run for {c_outfile}... file already exists!")
                continue
            log_once(f"Using following {len(fit_cns)} columns for variable selection :")
            
            del model_inputs
            txt_cols = filter_strings_regex(fit_cns, '_dist$')
            log_once(f"Read {len(protected_cols+fit_cns)} columns, {len(txt_cols)} of which text columns, outcome of interest = {outcome_of_interest}")
            try_print_col_counts_by_type(X, log_once)
            bin_cols = [x for x in fit_cns if try_regex('(^t_|_\d+$)', x)] # e.g.,  t_VHD, sex_1
            cont_cols = [x for x in fit_cns if x not in bin_cols and x != 'id']
            inci = Y.sum().values[0] / nrow(Y)
            Y.columns = [outcome_of_interest] 
            logger(f"{outcome_of_interest} Incidence = {inci*100:0.2f}%")
            rf_auc, rf_auc_sd = run_function(in_X, Y, cv=T, **c_params_to_try)
            logger(f"auc(sd)= {rf_auc:0.2f}({rf_auc_sd:0.3f})")
            c_score_with_params = [{"auc": rf_auc, "auc_sd": rf_auc_sd, "params": c_params_to_try, "outfile": c_outfile}]
            params_scores += c_score_with_params


        params_scores = sorted(params_scores, key=lambda x:x['auc'], reverse=T)
        best_score = params_scores[0]
    elif len(hp_params_to_try) == 1: # ::|2025/5/15 10:53:9|::	=== best params = [('n_jobs', -1), ('class_weight', 'balanced')]  :::
        hp_params_str = "_".join([ f"{k[:3]}{k[-4:]}{hp_params_to_try[0][k]}" for k in hp_params.keys()])
        c_outfile = full_filename_pkl(f"{outfile}{hp_params_str}")
        best_score['outfile'] = c_outfile
        best_score['params'] = hp_params_to_try[0]
        best_score['auc'] = '-1'
    else:
        logger("Error: provide at least one set of hyperparam values (hp_params_to_try == [])")

    logger(f"Found best hyper-params, selecting variables using said params...")
    rf_auc, vars_selected = run_function(in_X, Y, cv=F, **best_score['params'])
    vars_to_save = protected_cols + vars_selected
    out_obj = {
                "X" : X[vars_to_save],
                "outcome_of_interest": outcome_of_interest,
                "Y" : Y,
                'selection_auc_score' : rf_auc
                }

    save_pickle(best_score['outfile'], out_obj)


    logger(f"Best score: {best_score['auc']}")
    logger(f"Params from best score: {best_score['params']}")
    logger(f"({len(vars_selected)})Vars from best score: {vars_selected}")
    logger(f"File best score: {best_score['outfile']}")

    logger("DONE", t0)
    return out_files


