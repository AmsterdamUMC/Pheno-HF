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
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"
full_filename_pkl = lambda fname : f"{fname}{subsampled_str}.pkl"
full_filename_tsv = lambda fname, batch_n: f"{fname}{subsampled_str}b{batch_n}.tsv"
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


apply_flwp_cens_window = lambda pid, flwp_end, all_HFpos_ids : flwp_end if pid not in all_HFpos_ids else flwp_end - FOLLOW_UP_HFPOS_CENS_WINDOW
# <= 0 : happened before end of follow-up, >= -2*DAYS_IN_YEAR : happened after start of follow-up
is_dt_during_flwp = lambda pat, dt: dt >= COHORT_TIME_START_DAYS  and  dt - pat['follow_up_LAST'] <= 0 and dt - pat['follow_up_LAST'] >= -2*DAYS_IN_YEAR
is_med_dt_during_flwp = lambda pat, m: is_dt_during_flwp(pat, m['medication_datetime'])
is_ep_dt_during_flwp = lambda pat, ep: is_dt_during_flwp(pat, ep['episode_start_date']) or any([is_dt_during_flwp(pat, j['journal_datetime']) for j in ep['JOURNALS']])
is_j_dt_during_flwp = lambda pat, j: is_dt_during_flwp(pat, j['journal_datetime']) 

flwp_filter_eps = lambda p: [ep for ep in p['Episodes']  if is_ep_dt_during_flwp(p, ep)]
flwp_filter_js = lambda p, js: [j for j in js if is_j_dt_during_flwp(p, j)]

split_code_rowname = lambda rn, code_str_prefix="icpc_ep": ["_".join(rn.split("_")[:2]), rn.split("_")[-1:][0]][::-1] if rn.startswith(code_str_prefix)  else rn.split("_")[::-1]

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

logger = get_default_logger_fn(__file__, override=False)

ns_name = get_ns_name(__file__)
ns = getattr(namespaces, ns_name)



def calc_cluster_stats(df, sortby = 'inci'):
    gmm_labs = try_regex_multi(cns(df), 'gmm_cls')
    prob_gmm_labs = try_regex_multi(gmm_labs, '_prob')
    abs_gmm_labs = try_sd(gmm_labs, prob_gmm_labs)
    reference_cat_df = df #  any rows that dont have any other cat column values set to True

    def derive_row_df(df, c_df, c_lab):
        ct = try_table(c_df[ns.outcome_col]).to_dict()
        c_inci = ct[T] / ( ct[T] + ct[F] )
        return {'gmm_lab': c_lab, 'inci': c_inci, 'size' : nrow(c_df), 'size_percent' : nrow(c_df)/nrow(df)}
            
    incis_df = []
    for c_lab in abs_gmm_labs:
        c_df = df[df[c_lab] == T]
        reference_cat_df = reference_cat_df[reference_cat_df[c_lab] == F]
        incis_df = incis_df + [derive_row_df(df, c_df, c_lab)]

    
    incis_df = incis_df + [
                    derive_row_df(df, reference_cat_df, 'gmm_cls_REF'),
                    derive_row_df(df, df, 'ALL'),
                 ]
    
    incis_df = pd.DataFrame(incis_df)
    incis_df = incis_df.sort_values(by = [sortby], ascending=F)
    return incis_df

def try_dict_to_df(x, cols_to_keep=None):
    if cols_to_keep is None:
        cols_to_keep = list(list(x.values())[0].keys())

    x_trimmed = {k:{vk:vs[vk] for vk in cols_to_keep} for k,vs in x.items()}
    p_ids = list(x_trimmed.keys())
    df = pd.DataFrame.from_records(list(x_trimmed.values()))
    return df


def get_nested_gmm_clsvars(X, model, comp_gmms):
    vars_used = model.feature_names_in
    comp_X = None
    comp_vars = []
    for comp_gmm in comp_gmms:
        tmp = read_pickle(comp_gmm)
        c_gmm = tmp['model']
        comp_vars += c_gmm.feature_names_in
        c_X = tmp['X'][ ['id'] + c_gmm.feature_names_in]
        c_X['id'] = c_X['id'].apply(convert_pat_id_float2str)
        if comp_X is None:
            comp_X = c_X[['id']]

        c_preds = c_gmm.predict_class(c_X.drop('id', axis=1))
        c_X['c_preds'] = c_preds
        comp_nm = comp_gmm.split("_")[2]
        for c_clust in range(c_gmm.n_components):
            c_nm = f"c{c_clust}_{comp_nm}"
            if c_nm in vars_used:
                c_X[c_nm] = [c == c_clust for  c in c_preds]
                comp_X = pd.merge(comp_X, c_X[ ['id',c_nm]], on='id', how='inner')
            
    X = pd.merge(X, comp_X, on='id', how='inner')
    X['id'] = X['id'].apply(convert_pat_id_str2float)
    return X, comp_vars


 

def load_cohort(cohort_fn = 'analyse_results_cohort.pkl', override_saved_file=F, with_gmm_dummies=T):
    tmp = None
    gmm_file = ns.__dict__[ns.res_mod]['GMM_file']
    cohort_fn = f"{ns.res_mod}_{cohort_fn}"
    if pickle_exists(cohort_fn) and not override_saved_file:
        tmp = read_pickle(cohort_fn)
    else:
        tmp = read_pickle(gmm_file)
        model = tmp['model']
        n_gmm_classes = model.n_components
        vars_used = list(model.feature_names_in)
        X = tmp['X']
        X_in = None
        if ns.res_mod == 'nested':
            logger("DEPRECATED!!!! ns.res_mod == nested ")
            X, _ = get_nested_gmm_clsvars(X, model,
                    comp_gmms = ['_7_other_nested_gmm_output.pkl', '_8_icpc_nested_gmm_output.pkl',
                                '_10_atc_nested_gmm_output.pkl', '_12_txt_nested_gmm_output.pkl'] 
                                )

        X_in = X[vars_used]
        Y_pred = model.predict_class(X_in)
        Y_proba = model.predict_proba_class(X_in) # shape = (n, n_components)
        assert len(try_table(Y_pred)) == n_gmm_classes
        logger(f"X attrs = {try_print_list(cns(X))}")
        gmm_classes = X[[ 'id', 'follow_up_LAST', 'deceased_1'] + vars_used  ].copy()
        gmm_classes['id_str'] = gmm_classes['id'].apply(convert_pat_id_float2str)
        gmm_classes['gmm_cls'] = Y_pred
        gmm_prob_vars = [f'gmm_cls_{c_gmm_cls}_prob' for c_gmm_cls in range(1, n_gmm_classes)]
        for c_gmm_cls in range(1, n_gmm_classes): # ignore one of the probas (0-class) as all probs sum to 1
            gmm_classes[gmm_prob_vars[c_gmm_cls-1]] = Y_proba[:,c_gmm_cls]

        cluster_sizes = gmm_classes[['id_str', 'gmm_cls']]
        cluster_sizes = cluster_sizes.drop_duplicates(ignore_index=T).groupby(['gmm_cls']).size().reset_index(name='clust_size')
        cluster_sizes = cluster_sizes.to_dict()['clust_size']
        gmm_classes['prac_id'] = gmm_classes.id_str.apply(lambda x : x.split('|')[1])
        x = read_pickle(ns.pats_dict_file)
        cols_for_targethf = ['age_days', 
                    'sex',
                    't_cvd_in_family',
                    't_coronary_artery_disease',
                    't_atrial_fibrillation',
                    't_heart_murmur',
                    't_valvular_heart_disease',
                    't_hypertension',
                    't_stroke',
                    't_copd',
                    't_diabetes_mellitus',
                    't_chronic_kidney_disease',
                    't_alcohol_abuse',
                    't_tobacco_use',
                    't_obesity',
                    't_material_deprivation',
                    't_AF',
                    't_VHD',
                    't_HF',
                    't_min',
                    't_birth'
            ]
        p_ids = list(x.keys())
        X['id_str'] = [convert_pat_id_float2str(v) for v in vals(X.id)]
        df = try_dict_to_df(x, cols_for_targethf)
        df['id_str'] = p_ids
        df = df[df.id_str.isin(X.id_str)]
        df = pd.merge(df, X[['id_str', 'follow_up_LAST']], on ='id_str', how='inner')
        # df = df[df['t_HF'] < df['follow_up_LAST']]
        df['follow_up_effective'] = df['follow_up_LAST']
        df['event'] = ~pd.isnull(df.t_HF)
        df.loc[df['event'] == T, ['follow_up_effective']] = df.loc[df['event'] == T, ['follow_up_LAST']] - FOLLOW_UP_HFPOS_CENS_WINDOW

        cohort = df.copy()
        cohort['male'] = df['sex'] != 'V'
        cohort['decades_age'] = round((df['follow_up_LAST'] +  df['age_days'] ) / 3653) # decades
        try_table(cohort['decades_age'])

        # apply censoring
        df.loc[df['t_cvd_in_family'] >= df['follow_up_effective'], ['t_cvd_in_family']] = np.nan
        df.loc[df['t_coronary_artery_disease'] >= df['follow_up_effective'], ['t_coronary_artery_disease']] = np.nan
        df.loc[df['t_atrial_fibrillation'] >= df['follow_up_effective'], ['t_atrial_fibrillation']] = np.nan
        df.loc[df['t_heart_murmur'] >= df['follow_up_effective'], ['t_heart_murmur']] = np.nan
        df.loc[df['t_valvular_heart_disease'] >= df['follow_up_effective'], ['t_valvular_heart_disease']] = np.nan
        df.loc[df['t_hypertension'] >= df['follow_up_effective'], ['t_hypertension']] = np.nan
        df.loc[df['t_stroke'] >= df['follow_up_effective'], ['t_stroke']] = np.nan
        df.loc[df['t_copd'] >= df['follow_up_effective'], ['t_copd']] = np.nan
        df.loc[df['t_diabetes_mellitus'] >= df['follow_up_effective'], ['t_diabetes_mellitus']] = np.nan
        df.loc[df['t_chronic_kidney_disease'] >= df['follow_up_effective'], ['t_chronic_kidney_disease']] = np.nan
        df.loc[df['t_alcohol_abuse'] >= df['follow_up_effective'], ['t_alcohol_abuse']] = np.nan
        df.loc[df['t_tobacco_use'] >= df['follow_up_effective'], ['t_tobacco_use']] = np.nan
        df.loc[df['t_obesity'] >= df['follow_up_effective'], ['t_obesity']] = np.nan
        df.loc[df['t_material_deprivation'] >= df['follow_up_effective'], ['t_material_deprivation']] = np.nan
  
        # binarize
        cohort['cvd_in_family'] = ~pd.isnull(df['t_cvd_in_family'])
        cohort['coronary_artery_disease'] = ~pd.isnull(df['t_coronary_artery_disease'])
        cohort['atrial_fibrillation'] = ~pd.isnull(df['t_atrial_fibrillation'])
        cohort['heart_murmur'] = ~pd.isnull(df['t_heart_murmur'])
        cohort['valvular_heart_disease'] = ~pd.isnull(df['t_valvular_heart_disease'])
        cohort['hypertension'] = ~pd.isnull(df['t_hypertension'])
        cohort['stroke'] = ~pd.isnull(df['t_stroke'])
        cohort['copd'] = ~pd.isnull(df['t_copd'])
        cohort['diabetes_mellitus'] = ~pd.isnull(df['t_diabetes_mellitus'])
        cohort['chronic_kidney_disease'] = ~pd.isnull(df['t_chronic_kidney_disease'])
        cohort['alcohol_abuse'] = ~pd.isnull(df['t_alcohol_abuse'])
        cohort['tobacco_use'] = ~pd.isnull(df['t_tobacco_use'])
        cohort['obesity'] = ~pd.isnull(df['t_obesity'])
        cohort['material_deprivation'] = ~pd.isnull(df['t_material_deprivation'])

        # YES! confirmed no information leakage, i.e., did we ensure to mask conditions that occurred after end of follow-up?
        cohort['time_to_event'] = 0.25 # 3-months (1/4 of year) , censoriship window
        assert nrow(X) == nrow(cohort)
        targethf_preds = calc_TARGETHF_scores(cohort)
        cohort['targetHF_score'] = targethf_preds
        cohort = cohort.drop('time_to_event', axis=1)
        try_print_list(cns(cohort))
        vars_used = list(model.feature_names_in)
        vars_used = try_sd(vars_used, ['age_days'])
        # also filters only records that we used in experiments
        gmm_vars = ["id_str", "gmm_cls"] + gmm_prob_vars + vars_used
        cohort = pd.merge(cohort, gmm_classes[gmm_vars], on = 'id_str', how = 'inner').reset_index(drop=T)
        logger(f'N = {nrow(cohort)} records in cohort.')
        vars_used =  vars_used + ['decades_age']
        #cohort = cohort.drop('age_days', axis=1) # age_days doesnt ahve this issue..
        logger(f'N = {nrow(cohort)} records in cohort with age not missing.')
        tmp = { 'cohort': cohort,
                'n_gmm_classes' : n_gmm_classes,
                'gmm_file' : gmm_file,
                'vars_used' : vars_used
                }
        
        if override_saved_file or not pickle_exists(cohort_fn):
            save_pickle(cohort_fn, tmp)

    if with_gmm_dummies:
        tmp['cohort'] = pd.get_dummies(tmp['cohort'], columns=['gmm_cls'], drop_first=T)
    return tmp


#@Cacheable
def init__gmm_results(gmm_file = ns.__dict__[ns.res_mod]['GMM_file']):
    tmp = read_pickle(gmm_file)
    model = tmp['model']
    X = tmp['X']
    Y = tmp['Y']
    vars_used = list(model.feature_names_in)
    X_in = X[vars_used]
    Y_pred = model.predict_class(X_in)
    Y_proba = model.predict_proba_class(X_in) # shape = (n, n_components)
    logger(f"X attrs = {try_print_list(cns(X))}")

    gmm_df = X[[ 'id', 'follow_up_LAST', 'deceased_1']  ].copy() # Note: DO NOT TRUST MORTALITY, data is unreliable
    gmm_df['id_str'] = gmm_df['id'].apply(convert_pat_id_float2str)
    gmm_df['gmm_cls'] = Y_pred
    gmm_df['event'] = Y
    
    return model, vars_used, gmm_df