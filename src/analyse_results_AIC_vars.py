from try_utils import *
from constants import *
from sys import exit

IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"

check_if_debugging(IS_DEBUG)

outfile_infix = f"{subsampled_str}"
logfile = f'{os.path.basename(__file__)[:-3]}_{outfile_infix}.log'
logger = get_logger_fn(logfile)
logger(f"Starting ...")

import random
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import analyse_results_util as ar_util

def _backwardsAIC(m, X, y):
    c_aic = m.aic
    best_aic = c_aic
    best_m = m
    vr_to_remove = []
    for vr in try_sd(cns(X), ['const']):
        c_m = _fit_model(y, X.drop(vr, axis=1))
        if c_m.aic < best_aic:
            best_aic = c_m.aic
            best_m = c_m
            vr_to_remove = [vr]
    return try_sd(cns(X), vr_to_remove), best_aic

def _stepwiseBackwardsAIC(X, y, return_m=F):
    
    c_vars = cns(X)
    # return c_vars
    prev_vars = []
    idxs_0s = np.where(y == 0)[0]
    idxs_1s = np.where(y == 1)[0]
    sample_prop = 1
    if len(c_vars) > 30:
        sample_prop = 1
    n_0s = int(len(idxs_0s)*sample_prop)
    n_1s = int(len(idxs_1s)*sample_prop)
    logger(f"Using subsampled input of {sample_prop} to reduce compute time")

    np.random.shuffle(idxs_0s)
    idxs_0s = idxs_0s[:n_0s]

    np.random.shuffle(idxs_1s)
    idxs_1s = idxs_1s[:n_1s]

    x_idxs = idxs_0s.tolist() + idxs_1s.tolist()

    s_Y = pd.Series([0]*n_0s + [1]*n_1s)

    sample_idxs = list(range(len(x_idxs)))
    np.random.shuffle(sample_idxs)
    x_idxs_shuffled = [x_idxs[i] for i in sample_idxs]
    s_X = X.iloc[x_idxs_shuffled]
    s_Y = s_Y.iloc[sample_idxs]

    s_Y = s_Y.reset_index(drop=T)
    s_X = s_X.reset_index(drop=T)

    m = _fit_model(s_Y, s_X)
    while try_sdui(c_vars, prev_vars) != []:
        prev_vars =  c_vars
        c_vars, c_aic = _backwardsAIC(m, s_X[c_vars], s_Y)
        logger(f"_stepwiseBackwardsAIC: Removed: {try_sd(prev_vars, c_vars)};  {len(c_vars)}/{ncol(X)} vars remaining. current AIC={c_aic:1.4e}")
    if return_m:
        return _fit_model(y, X[c_vars])
    return c_vars

def _get_c_X(cohort, cvar):
    c_X = cohort[cvar]
    for c in cns(c_X):
        if type(c_X[c].values[0]) != np.float64:
            c_X[c] = c_X[c].astype(int)
    return c_X
        

def _fit_model(y, c_X):
    c_X  = sm.add_constant(c_X)
    logitModel = sm.Logit(y, c_X)
    m = logitModel.fit(disp=0)
    return m

def _get_cv_folds(y, c_X, nfolds =4):
    cv = StratifiedKFold(n_splits = nfolds, shuffle=T, random_state = 42)
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    for train_idx, test_idx in cv.split(c_X, y):
        x_train = c_X.iloc[train_idx]
        x_trains = x_trains + [x_train]

        y_train = y.iloc[train_idx]
        y_trains = y_trains + [y_train]

        x_test = c_X.iloc[test_idx]
        x_tests = x_tests + [x_test]

        y_test = y.iloc[test_idx]
        y_tests = y_tests + [y_test]
    return x_trains,y_trains,x_tests,y_tests


def analyse_AIC_newvars():
    # Note -  it is not a good idea to combine two representations of the cluster vars at once 
    # i.e., boolean absolute labels (1 record = exactly one cluster label) , vs soft labels (1 record = 10% of cluster a, 30% of cluster b, etc...)
    # setup two experiments: one where clusters are absolute (ABS) , and one where clusters are probabilistic (PRB)
    tmp = ar_util.load_cohort(override_saved_file=F)
    cohort = tmp['cohort']
    gmm_vars = tmp['vars_used']
    
    clust_stats = ar_util.calc_cluster_stats(cohort)
    baseline_inci = vals(clust_stats[clust_stats['gmm_lab'] == 'ALL']['inci'])[0]
    hi_hf_cluster_cols = vals(clust_stats[clust_stats['inci'] > 2*baseline_inci]['gmm_lab'])


    gmm_labs = try_regex_multi(cns(cohort), 'gmm_cls')
    prob_gmm_labs = try_regex_multi(gmm_labs, '_prob')
    abs_gmm_labs = try_sd(gmm_labs, prob_gmm_labs)

    protected_cols =  [ar_util.ns.id_col, ar_util.ns.outcome_col] #[ar_util.ns.id_col, ar_util.ns.outcome_col, ar_util.ns.target_hf_col] + gmm_labs + gmm_vars
    all_cols = sorted(list(set(protected_cols + prob_gmm_labs + abs_gmm_labs + gmm_vars + [ar_util.ns.target_hf_col] + ar_util.ns.targetHF_cols)))
    cohort = cohort.drop(try_sd(cns(cohort), all_cols), axis =1)
    cohort['t_coronary_artery_disease'] = cohort['coronary_artery_disease']
    cohort['t_chronic_kidney_disease'] = cohort['chronic_kidney_disease']
    cohort['t_copd'] = cohort['copd']
    cohort['t_stroke'] = cohort['stroke']
    cohort['t_diabetes_mellitus'] = cohort['diabetes_mellitus']
    cohort['t_valvular_heart_disease'] = cohort['valvular_heart_disease']
    cohort['t_hypertension'] = cohort['hypertension']
    cohort['t_atrial_fibrillation'] = cohort['atrial_fibrillation']
    assert try_sdui(all_cols , cns(cohort)) == []


    #Do the new clusters add anything on top of the tHF vars?
    #@    cluster_vars_and_targetHF_vars

    #Do the new clusters add anything on top of the vars they were derived from?
    #@[ABS|PROB]_cluster_ID  [i]
    #@cluster_vars [ii]
    #@[ABS|PROB]_cluster_vars_and_cluster_ID [iii]
    # case [iii] > [ii] > [i]  && [iii] has vars both types of vars  ==> yes 
    # case [iii] <= [ii] ==> no 
        # IF YES 
        # i.e. ,new clusters DO ADD on top of the vars they were derived from
        #Do the new cluster [probs/labels] + [their composing vars] add anything on top of the tHF vars?
        # @[ABS|PROB]_cluster_vars_and_cluster_ID_and_targetHF_vars [i]
        # case [i] has all 3 types of vars ==> yes
        # case [i] has NO tHF vars ==> yes, and in fact they capture everything from tHF already
        # case [i] has only tHF vars ==> no

    #Do the new cluster probs add anything on top of the cluster abs labels?
    # @[ABS|PROB]_cluster_ID [i]
    # case [i].PROB > [i].ABS ==> yes
    # case [i].PROB <= [i].ABS ==> no
    # note: can not do cluster_ID_and_cluster_prob due to multicollinearity! have to just see which one of the two above has higher AIC 

    # NOTE: the manuscript's Table 2 reports four predictor setups: (i)
    # TARGET-HF only ("THF_COLS", below), (ii) cluster/phenotyping variables
    # only ("cluster_vars"), (iii) TARGET-HF + cluster membership
    # ("ABS_targetHF_vars_and_cluster_ID"), and (iv) TARGET-HF + cluster
    # membership + cluster variables ("ABS_cluster_vars_and_cluster_ID_and_
    # targetHF_vars"). Only setup (iii) is currently active below — the
    # others are commented out and would need to be re-enabled to reproduce
    # the full Table 2 comparison.
    setup_vars = {
        # "cluster_vars" : gmm_vars,
        # "cluster_vars_and_targetHF_vars" : gmm_vars + ar_util.ns.targetHF_cols,

        # ABS
        # "ABS_cluster_ID" : abs_gmm_labs,
        "ABS_targetHF_vars_and_cluster_ID" : ar_util.ns.targetHF_cols + abs_gmm_labs,
        # "ABS_cluster_vars_and_cluster_ID" : gmm_vars + abs_gmm_labs,
        # "ABS_cluster_vars_and_cluster_ID_and_targetHF_vars" : gmm_vars + abs_gmm_labs + ar_util.ns.targetHF_cols,

        # # PROB
        # "PROB_cluster_ID" : prob_gmm_labs,
        # "PROB_targetHF_vars_and_cluster_ID" : ar_util.ns.targetHF_cols + prob_gmm_labs,
        # "PROB_cluster_vars_and_cluster_ID" : gmm_vars + prob_gmm_labs,
        # "PROB_cluster_vars_and_cluster_ID_and_targetHF_vars" : gmm_vars + prob_gmm_labs + ar_util.ns.targetHF_cols,
        # "THF_COLS" : ar_util.ns.targetHF_cols,
    }

    setup_vars = { k:sorted(list(set(v))) for k,v in setup_vars.items()}


    cached_vars_results = {

    }
    # try_table(cohort[ar_util.ns.outcome_col])
    y = cohort[ar_util.ns.outcome_col].astype(int)
    res_df = {}
    is_var_dup = lambda v,vars : v.startswith('t_') and v[2:] in vars
    filter_var_dups = lambda cvars : [v for v in cvars if not is_var_dup(v, cvars)]

    #  run backwards AIC for each var setup
    for setup_nm,cvar in setup_vars.items():
        cvar  = filter_var_dups(cvar)
        c_X = _get_c_X(cohort, cvar)
        dup_age_idx = [i for i,c in  enumerate(cns(c_X)) if c == 'decades_age' ]
        if len(dup_age_idx) > 1:
            dup_age_idx = dup_age_idx[1:]
            c_X = c_X.drop( [c_X.columns[i] for i in dup_age_idx], axis=1)

        # find vars of best reduced model
        rdc_vars = []
        if setup_nm in cached_vars_results:
            logger(f"using cached_vars_results for {setup_nm}")
            rdc_vars = cached_vars_results[setup_nm]
        else:
            rdc_vars = _stepwiseBackwardsAIC(c_X, y)

        c_X = c_X[rdc_vars]
        full_m = _fit_model(y, c_X) # use for AIC
        logger(f"AIC for {setup_nm}:: \n {cvar} = {full_m.aic:0.3e}...")
        res_df[setup_nm] = { 
            "vars": ";".join(sorted(rdc_vars)),
            "AIC" : full_m.aic 
            }


    # per var setup, compute AUC for vars left from backwards AIC
    for setup_nm in setup_vars.keys():
        rdc_vars = res_df[setup_nm]['vars'].split(';')
        logger(f"AUC for {setup_nm}:: = {0}")
        c_X = _get_c_X(cohort, rdc_vars)
        # calculate AUC of best reduced model
        nfolds = 4
        x_trains,y_trains,x_tests,y_tests = _get_cv_folds(y, c_X, nfolds=nfolds)
        aucs = []
        for i in range(nfolds):
            m = _fit_model(y_trains[i], x_trains[i])
            preds = m.predict(sm.add_constant(x_tests[i]))
            aucs += [roc_auc_score(y_tests[i], preds)]
        res_df[setup_nm]["AUC_mean"] = np.mean(aucs)
        res_df[setup_nm]["AUC_sd"] = np.std(aucs)

    # compute summary stats    
    res_df = pd.DataFrame(res_df)
    res_df = res_df.T
    res_df = res_df.sort_values(by=['AIC'])
    all_vars_atc = try_regex_multi(all_cols, '_atc_')
    n_vars_atc = len(all_vars_atc)
    all_vars_icpc = try_regex_multi(all_cols, '_icpc_')
    n_vars_icpc = len(all_vars_icpc)
    all_vars_txt = try_regex_multi(all_cols, '_text_')
    n_vars_txt = len(all_vars_txt)


    all_vars_gmm = try_regex_multi(all_cols, 'gmm_cls')
    all_vars_gmm_prob = try_regex_multi(all_vars_gmm, '_prob')
    all_vars_gmm_id = try_sd(all_vars_gmm, all_vars_gmm_prob)

    n_vars_gmm_id = len(all_vars_gmm_id) 
    n_vars_gmm_prob= len(all_vars_gmm_prob) 
    assert n_vars_gmm_id == n_vars_gmm_prob

    n_vars_tHF = len(ar_util.ns.targetHF_cols)

    all_t_vars_tHF = [f't_{i}' for i in try_sd(ar_util.ns.targetHF_cols, 'decades_age') ] + ['decades_age']

    res_df['nvars'] = [ len(x.split(";")) for x in vals(res_df['vars'])]
    res_df['nvars_atc'] = [ f"{len(try_regex_multi(x.split(';'), '_atc_'))}/{n_vars_atc}" for x in vals(res_df['vars'])]
    res_df['nvars_icpc'] = [ f"{len(try_regex_multi(x.split(';'), '_icpc_'))}/{n_vars_icpc}" for x in vals(res_df['vars'])]
    res_df['nvars_txt'] = [ f"{len(try_regex_multi(x.split(';'), '_text_'))}/{n_vars_txt}" for x in vals(res_df['vars'])]
    res_df['nvars_gmm_ids'] = [ f"{len(try_regex_multi(x.split(';'), 'gmm_cls'))}/{n_vars_gmm_id}" for x in vals(res_df['vars'])]
    res_df['nvars_tHF'] = [ f"{len([i for i in x.split(';') if i in try_su(ar_util.ns.targetHF_cols, all_t_vars_tHF) ])}/{n_vars_tHF}" for x in vals(res_df['vars'])]

    all_vars_gmm_other = try_sd(gmm_vars, all_vars_txt + all_vars_atc + all_vars_icpc + ar_util.ns.targetHF_cols)
    n_vars_gmm_other = len(all_vars_gmm_other)
    res_df['nvars_gmm_other'] = [f"{len([i for i in x.split(';') if i in all_vars_gmm_other ])}/{n_vars_gmm_other}" for x in vals(res_df['vars'])]


    # re-order cols
    cols_ordered = ['setup_nm', 'AIC', 
        'AUC_mean', 'AUC_sd',
        'nvars',
        'nvars_icpc',
        'nvars_atc',
        'nvars_txt',
        'nvars_gmm_ids',
        'nvars_gmm_other',
        'nvars_tHF',
        'vars'
        ]
    res_df = res_df.reset_index(drop=T)
    res_df['setup_nm'] = res_df.iloc[:,0]
    #assert try_sdui(cols_ordered, cns(res_df)) == []
    res_df = res_df[cols_ordered]
    
    #res_df.columns.values = [" ".join(i.split("_")) for i in cns(res_df)]
    res_df.to_excel("excel/targetHF_cluster_vars_AIC.xlsx")

    return 0

analyse_AIC_newvars()