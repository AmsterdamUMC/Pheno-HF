# -*- coding: utf-8 -*-
print(
    '''
# WHAT THIS SCRIPT DOES
# Performs clustering analysis using dataset produced by previous step
# Saves fitted models/  predictions/ metrics/ metadata
'''
)

from constants import T, F
from try_utils import *
import pandas as pd
from stepmix.stepmix import StepMix, StepMixClassifier
from stepmix.utils import get_mixed_descriptor
from try_stepmix import StepMixBICScore
from itertools import product
from try_utils import parse_commandline_args, check_if_debugging
IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
logger = get_default_logger_fn(__file__)

pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)

import random
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

protected_cols = ['id', VAR_FOLLOW_UP_DATE, 'deceased_1']

USE_GS = T # GridSearchCV for hyperparams/ otherwise use last-cached version of params
FIT_PM_TEST_AUC = F # run with very small number of hyperparams / iterations

def run_it(
    infile,
    hp_params,
    outfile,
    use_only_vars=[],
    n_components = list(range(3,15))
    ):
    subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
    use_outcome = hp_params['use_outcome']

    use_only_text_icpc = hp_params['use_only_text_icpc']
    idx = 0 if SUBSAMPLE_DATA else 1
    
    DIM_REDUC_TECHNIQUE = {
                    "autoencoder": "dim_red__autoenc",
                    "expert_knowledge" : "dim_red__expert_knowledge" 
                    }
    DIM_REDUC_TECHNIQUE = DIM_REDUC_TECHNIQUE['expert_knowledge']

    model_inputs = read_pickle(infile)
    X = model_inputs['X']
    Y = model_inputs['Y']
    Y = Y.iloc[:,0].values
    n = nrow(X)
    fit_cns = try_sd(cns(X), protected_cols)


    if use_only_vars != []:
        fit_cns = try_sd(use_only_vars, protected_cols)
        X = X[try_su(protected_cols, use_only_vars)]
        logger(f"Using vars: {use_only_vars}")

    try_print_list(sorted(fit_cns), logger)
    assert len(fit_cns) < 600
    outcome_of_interest = model_inputs['outcome_of_interest']
    npos = np.sum(Y)
    inci = (npos*100)/n
    # logging 
    txt_cols = [c for c in fit_cns if '_dist' in c]
    logger(f"Going to run StepMix for data with {dim(X)[0]} rows and {dim(X)[1]} columns, {len(txt_cols)} of which text-based columns")
    logger(f"Outcome of interest = {outcome_of_interest}; N_positives = {npos}; (inci={inci:0.2f}%)")

    results_expert_knowledge = []
    t1 = logger("Start fitting Gaussian Mixture models... (i.e., LCA/LFA)")

    logger(f"Going to try with {n_components} clusters (count = {len(n_components)})")
    
    if FIT_PM_TEST_AUC:
        logger("Fitting PM with age only")    
        run_PM_test(X, Y, ['age_days'])

        logger("Fitting PM with all vars")
        run_PM_test(X, Y)

    def create_model(
                    n_components=5,
                    random_state=42,
                    use_outcome=T,
                    auto_weight_class=0,
                    measurement = 'binary',
                    progress_bar=0,
                    n_init=1,
                    abs_tol=1,
                    max_iter=50):
        return StepMixBICScore(
                                auto_weight_class=auto_weight_class,
                                use_outcome=use_outcome, 
                                n_components = n_components,
                                measurement = measurement,
                                structural='binary',
                                random_state=random_state, 
                                init_params='random',
                                n_init=n_init, 
                                verbose=0, 
                                n_steps=1, # consider 2, 3 
                                abs_tol=abs_tol, 
                                progress_bar=progress_bar, 
                                max_iter=max_iter)
    if USE_GS:
        param_grid = { 
                        'random_state' : [222], #list(range(5)),
                        'n_components':  n_components,
                        'use_outcome' : [F],
                        'auto_weight_class' : [0], #1
                    }  
        
        def expand_grid(g):
            return pd.DataFrame([r for r in product(*g.values())], columns=g.keys())
        best_model = create_model()
        best_score = 0
        best_params = {}
        exp_params_grid = expand_grid(param_grid)
        params_used = []
        for i, c_hp in exp_params_grid.iterrows():
            c_params = c_hp.to_dict()
            if c_params not in params_used:
                params_used.append(c_params)
            else:
                continue
        logger(f"Going to try {len(params_used)} different param configurations...")
        val_scores = []
        train_scores = []
        should_early_stop = F
        c_param_wo_rs_counter = {}
        rs_counter = 0
        non_zero_score_counter = 0
        best_vars_cutoff = [-1,-1] 
        for i,c_params in enumerate(params_used):
            c_param_wo_rs = dict(c_params)
            del c_param_wo_rs['random_state']
            if c_param_wo_rs == c_param_wo_rs_counter:
                rs_counter += 1
                if rs_counter > 10 and non_zero_score_counter == 0:
                    logger("Skip further run since too many times zero score from random state")
                    continue
            else:
                c_param_wo_rs_counter = c_param_wo_rs
                rs_counter = 0
                non_zero_score_counter = 0
            logger(f"Trying params = {c_params} {i+1}/{len(params_used)}")

            mixed_data, mixed_descriptor = get_inputs_gmm(X)
            c_params['measurement'] = mixed_descriptor
            model_params = c_params
            if IS_DEBUG:
                model_params['progress_bar'] = 2
            model = create_model(**model_params)
            model.fit(mixed_data, Y)
            score_fn = lambda Y=Y: model.score(mixed_data, Y, verbose=T) 
            c_score = score_fn()
            train_scores += [c_score]
            c_score_val = score_fn(Y)
            val_scores += [c_score_val]
            logger(f"train score = {c_score:1.4e}; val score = {c_score_val:1.4e}")

            if c_score > 0.04:
                non_zero_score_counter += 1

        best_val_idxs = try_multiindex(val_scores, max(val_scores))
        best_train_idxs = try_multiindex(train_scores, max(train_scores))

        is_best_val_train_different = len(try_sdui(best_val_idxs, best_train_idxs)) > 0

        logger(f"USING TRAIN Y") 
        logger(f"best model params = {[ params_used[i] for i in best_train_idxs]}")
        logger(f"best score = {[ round(train_scores[i],3) for i in best_train_idxs]}")

        if is_best_val_train_different:
            logger(f"USING VAL Y (will still select based on train scores)") 
            logger(f"best model params = {[ params_used[i] for i in best_val_idxs]}")
            logger(f"best score = {[ round(val_scores[i],3) for i in best_val_idxs]}")        

    # final model(s)
    out_files = []
    n_fms = len(best_train_idxs)
    logger(f"Going to fit {n_fms} final models")
    for i,m_i in enumerate(best_train_idxs):
        c_params = params_used[m_i]
        c_params['progress_bar'] = 2
        mixed_data, mixed_descriptor = get_inputs_gmm(X)
        c_params['measurement'] = mixed_descriptor

        model = create_model(**c_params)
        logger(f"Fitting final model ({i+1}/{n_fms}) with selected hyperparams... {c_params}")
        model.fit(mixed_data, Y)
        score_fn = lambda Y=Y: model.score(mixed_data, Y, verbose=T) 
        c_score = score_fn(Y)
        logger(f"SCORE={c_score:1.4e}...")
        if Y is not None:
            cluster_metrics  = compute_metrics_per_cluster(model, mixed_data, Y)
        else:
            cluster_metrics = None
        file_str_outcome_used = "ou_" if c_params['use_outcome'] else ""
        out_file = f'{file_str_outcome_used}_{c_params["n_components"]}_{outfile}'
        if pickle_exists(out_file):
            out_file = f"{out_file}{random.randint(1e8,1e10)}"
        out_files += [out_file]
        save_pickle(out_file, { "model" : model,
                                "X" : model_inputs['X'] ,
                                "Y" : Y,
                                "Y" : Y,
                                "params": c_params,
                                "cluster_metrics": cluster_metrics
                                    } )
    logger("run_it DONE")
    return out_files

def get_inputs_gmm(X):
    vars_to_use = try_sd(cns(X), protected_cols)
    X = X[vars_to_use]
    fit_cns = cns(X)
    bin_cols = [x for x in fit_cns if nuniq(X[x]) == 2]
    cont_cols = [x for x in fit_cns if x not in bin_cols]
    mixed_data, mixed_descriptor = None, None

    if bin_cols and cont_cols:
        mixed_data, mixed_descriptor = get_mixed_descriptor(
                dataframe = X,
                gaussian_full = cont_cols,
                bernoulli = bin_cols
        )
    elif cont_cols:
        mixed_data, mixed_descriptor = get_mixed_descriptor(
                dataframe = X,
                gaussian_full = cont_cols
        )
    elif bin_cols:
        mixed_data, mixed_descriptor = get_mixed_descriptor(
                dataframe = X,
                bernoulli = bin_cols
        )
    return mixed_data, mixed_descriptor


def quick_score_model_config(X, Y, c_params, create_model, verbose=F):
    c_vars = cns(X)
    mixed_data, mixed_descriptor = get_inputs_gmm(X)
    c_params['measurement'] = mixed_descriptor

    model_params = c_params
    if IS_DEBUG:
        model_params['progress_bar'] = 2
    model = create_model(**model_params)
    model.fit(mixed_data, Y)


    score = model.score(X[c_vars], Y, verbose=verbose, use_bic=F)
    bic = model.score(X[c_vars], Y, verbose=F, use_bic=T)
    return {"score": score, "bic": bic}

def compute_metrics_per_cluster(model, X, Y):
    incis, supps, npos, mass = _get_outcome_inc_and_count_per_cluster(model, X, Y, verbose=T)
    logger(f"Found clusters with incidence ranging from {min(incis)*100:0.2f}% to {max(incis)*100:0.2f}%")
    cluster_metrics = {
        "incis" : incis,
        "supps" : supps,
        "npos": npos,
        "mass": mass
    }
    best_best_metrics = {}
    for m_i, m_nm in enumerate(cluster_metrics.keys()):
        mvs = cluster_metrics[m_nm]
        best_idx, best_val = try_select_best_metric(mvs)
        other_metric_vals = { k: [v[b_i] for b_i in best_idx] for k,v in cluster_metrics.items() if k != m_nm }
        if m_nm in best_best_metrics:
            if best_val > best_best_metrics[m_nm][0]:
                best_best_metrics[m_nm][0] = best_val
                best_best_metrics[m_nm][1] = best_idx
                best_best_metrics[m_nm][3] = other_metric_vals
                
        else:
            best_best_metrics[m_nm] = [best_val, best_idx, other_metric_vals]

    return best_best_metrics



def _get_outcome_inc_and_count_per_cluster(model, X, Y, verbose=F):
    if type(Y) == pd.DataFrame:
        Y = Y.iloc[:, 0].values
    Y_pred = model.predict_class(X)
    inci, supps, npos, mass = get_clusters_inc_supp(Y, Y_pred, verbose=F)
    c_score = model.score(X, Y)
    y_1 = np.max(Y)
    base_inci = sum([1 for y in Y if y == y_1]) / len(Y)
    
    if verbose:
        logger(f"Clustering SCORE:{c_score:0.3f}")
        [logger(f"Interesting cluster ({i+1}/{len(supps)})\t\tI:{inci[i]:0.3f}\tS:{s:.2e}\tM:{mass[i]:0.3f}\tN+:{npos[i]:0.3f}") 
                for i,s in enumerate(supps) if is_cluster_interesting(inci[i], mass[i], npos[i], base_inci)] 
    return inci, supps, npos, mass





