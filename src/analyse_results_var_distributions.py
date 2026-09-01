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
import analyse_results_util as ar_util
pickle_exists = lambda f: try_pickle_exists(f, subsampled=SUBSAMPLE_DATA)
read_pickle = lambda f: try_read_pickle(f, subsampled=SUBSAMPLE_DATA)
save_pickle = lambda f, o: try_save_pickle(f, o, subsampled=SUBSAMPLE_DATA)
cached_call = lambda fn, override_cache=F, **kwargs : try_cached_call(fn, io_r=read_pickle, io_c=pickle_exists, io_w=save_pickle, override_cache=override_cache, **kwargs)


#from Top2Vec import Top2Vec
#umap_file = EMBEDDING_MODEL_METADATA['doc2vec'][f'umap_embeddings_file{subsampled_str}']['0_24_epj_text_'] 
#hdbscan_file = EMBEDDING_MODEL_METADATA['doc2vec'][f'hdbscan_labels_file{subsampled_str}']['0_24_epj_text_']
print("s") #
# why do we need the model here? so we can map each doc to its label 

# [23.Jun.2025]
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

def analyse_var_distrs():
    # get cluster ids/probs 
    model, vars_used, gmm_df = cached_call(ar_util.init__gmm_results, override_cache=T)

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

    # tmp = read_pickle("_15_temp_gmm_output.pkl")
    # model = tmp['model']
    # vars_used = list(model.feature_names_in)
    # X = tmp['X']
    # X_in = X[vars_used]
    # Y_pred = model.predict_class(X_in)
    # Y_proba = model.predict_proba_class(X_in) # shape = (n, n_components)

    # logger(f"X attrs = {try_print_list(cns(X))}")

    # gmm_df = X[[ 'id', 'follow_up_LAST', 'deceased_1'] + vars_used  ].copy()
    # gmm_df['id_str'] = gmm_df['id'].apply(convert_pat_id_float2str)
    # gmm_df['gmm_cls'] = Y_pred

    # cluster_sizes = gmm_df[['id_str', 'gmm_cls']]
    # cluster_sizes = cluster_sizes.drop_duplicates(ignore_index=T).groupby(['gmm_cls']).size().reset_index(name='clust_size')
    # cluster_sizes = cluster_sizes.to_dict()['clust_size']
    # gmm_df['prac_id'] = gmm_df.id_str.apply(lambda x : x.split('|')[1])
    # x = read_pickle(f"pats_dict_merged.pkl")
    # cols_for_targethf = ['age_days',
    #             'sex',
    #             't_cvd_in_family',
    #             't_coronary_artery_disease',
    #             't_atrial_fibrillation',
    #             't_heart_murmur',
    #             't_valvular_heart_disease',
    #             't_hypertension',
    #             't_stroke',
    #             't_copd',
    #             't_diabetes_mellitus',
    #             't_chronic_kidney_disease',
    #             't_alcohol_abuse',
    #             't_tobacco_use',
    #             't_obesity',
    #             't_material_deprivation',
    #             't_AF',
    #             't_VHD',
    #             't_HF',
    #             't_min',
    #             't_max',
    #             't_birth'
    #     ]
    # x_trimmed = {k:{vk:vs[vk] for vk in cols_for_targethf} for k,vs in x.items()}
    # p_ids = list(x_trimmed.keys())
    # df = pd.DataFrame.from_records(list(x_trimmed.values()))
    # cohort = df.copy()
    # cohort['male'] = df['sex'] != 'V'
    # cohort = cohort[['male']]

    # cohort['decades_age'] = round((df['t_max'] +  df['age_days'] ) / 3653) # decades
    # cohort['cvd_in_family'] = ~pd.isnull(df['t_cvd_in_family'])
    # cohort['coronary_artery_disease'] = ~pd.isnull(df['t_coronary_artery_disease'])
    # cohort['atrial_fibrillation'] = ~pd.isnull(df['t_atrial_fibrillation'])
    # cohort['heart_murmur'] = ~pd.isnull(df['t_heart_murmur'])
    # cohort['valvular_heart_disease'] = ~pd.isnull(df['t_valvular_heart_disease'])
    # cohort['hypertension'] = ~pd.isnull(df['t_hypertension'])
    # cohort['stroke'] = ~pd.isnull(df['t_stroke'])
    # cohort['copd'] = ~pd.isnull(df['t_copd'])
    # cohort['diabetes_mellitus'] = ~pd.isnull(df['t_diabetes_mellitus'])
    # cohort['chronic_kidney_disease'] = ~pd.isnull(df['t_chronic_kidney_disease'])
    # cohort['alcohol_abuse'] = ~pd.isnull(df['t_alcohol_abuse'])
    # cohort['tobacco_use'] = ~pd.isnull(df['t_tobacco_use'])
    # cohort['obesity'] = ~pd.isnull(df['t_obesity'])
    # cohort['material_deprivation'] = ~pd.isnull(df['t_material_deprivation'])

    # # YES! confirmed no information leakage, i.e., did we ensure to mask conditions that occurred after end of follow-up?
    # cohort['time_to_event'] = 2 # two years follow-up # 
    # cohort['event'] = ~pd.isnull(df['t_HF'])

    # cohort['id_str'] = p_ids
    # targethf_preds = calc_TARGETHF_scores(cohort)
    # cohort['targetHF_score'] = targethf_preds
    # cohort = cohort.drop('time_to_event', axis=1)
    try_print_list(cns(cohort))

    vars_used = list(model.feature_names_in)


    # also filters only records that we used in experiments
    gmm_vars = ["id_str", "gmm_cls"] + vars_used
    cohort = pd.merge(cohort, gmm_df[["id_str", "gmm_cls"]], on = 'id_str', how = 'inner').reset_index(drop=T)
    logger(f'N = {nrow(cohort)} records in cohort.')

    cohort = pd.get_dummies(cohort, columns=['gmm_cls'], drop_first=F)

    outcome_col = 'event'
    target_hf_col = 'targetHF_score'
    id_col = 'id_str'


    hi_hf_cluster_cols = [ f"gmm_cls_{c}" for c in  ['2', '1', '0', '8', '11']]
    cluster_cols = [c for c in cns(cohort) if try_regex('gmm_cls', c)]
    protected_cols = [id_col, outcome_col, target_hf_col] + cluster_cols + vars_used
    cohort['age_days'] = cohort['decades_age']
    try_table(cohort[outcome_col])
    y = cohort[outcome_col].astype(int)
    X = cohort
    X = X.loc[:, ~X.columns.duplicated()].copy() # remove dup cols

    fig_counter = 1
    for c_var in vars_used:
        if c_var != 'age_days':
            continue
        for c_gmm_cls in [2, 1, 0, 8, 11]:
            plt.figure(fig_counter)
            fig_counter+=1
            fig, axes = plt.subplots(1,1 , figsize=(12,5))
            #c_var = vars_used[0]
            #c_gmm_cls = 13
            is_binary = len(set(X[c_var])) <= 3
            if is_binary:
                continue
            c_X = X[X[c_var] != 0] 
            c_vals = vals(c_X[c_var])
            c_vals_gmm = vals(c_X[c_X[f'gmm_cls_{c_gmm_cls}'] == 1][c_var])
            c_vals_idxs = list(range(0, len(c_vals)))
            random.shuffle(c_vals_idxs)
            c_vals_subsampled = [c_vals[i] for i in c_vals_idxs[:len(c_vals_gmm)]]
            n_0s = nrow(X) - nrow(c_X)
            n_0s_gmm = X[X[f'gmm_cls_{c_gmm_cls}'] == 1]
            X_gmm = nrow(X[X[f'gmm_cls_{c_gmm_cls}'] == 1])
            n_0s_gmm = nrow(n_0s_gmm[n_0s_gmm[c_var] == 0])
            
            axes.hist(c_vals_subsampled, bins =40, alpha=0.9, color='blue', edgecolor='none', log=F, label='ALL')

            axes.hist(c_vals_gmm, bins = 40, alpha=0.4, color='red', edgecolor='none', log=F, label=f'GMM C{c_gmm_cls}')
            axes.set_xlabel(c_var)
            axes.set_ylabel('Freq')
            axes.set_title(f'GMM class = {c_gmm_cls} (%0s ALL = {100*n_0s/nrow(X):0.1f}; %0s GMM = {100*n_0s_gmm/X_gmm:0.1f})')
            axes.legend()

            # axes[1].boxplot([c_vals, c_vals_gmm], patch_artist=T, whis=[5, 95])
            # axes[1].set_xticklabels(['ALL', f'GMM C{c_gmm_cls}'])
            # axes[1].set_ylabel(c_var)
            # axes[1].set_title('Box plot')

            plt.savefig(f'plots/sandbox_generic/hist_{c_var}_gmm{c_gmm_cls}.png')

    return 0

analyse_var_distrs()