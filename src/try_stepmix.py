from sklearn.utils.validation import (check_random_state, check_is_fitted, _check_sample_weight)
from stepmix.stepmix import StepMix
import numpy as np
import pandas as pd
from constants import (T, F)

from try_utils import (try_table, try_expand, try_reduce, tty, try_multiindex, nrow, cns)
from try_utils import (get_default_logger_fn, is_cluster_interesting)
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

logger = get_default_logger_fn(__file__, override=False)

class StepMixBICScore(StepMix):

    def __init__( 
            self,
            use_outcome= T,
            auto_weight_class = 0,
            n_components=2,
            *,
            n_steps=1,
            measurement="bernoulli",
            structural="gaussian_unit",
            assignment="modal",
            correction=None,
            abs_tol=1e-10,
            rel_tol=0.00,
            max_iter=1000,
            n_init=1,
            save_param_init=False,
            init_params="random",
            random_state=None,
            verbose=0,
            progress_bar=1,
            measurement_params=None,
            structural_params=None
            ):
        self.use_outcome = use_outcome
        self.auto_weight_class = auto_weight_class

        super().__init__(
            n_components=n_components,
            n_steps=n_steps,
            measurement=measurement,
            structural=structural,
            assignment=assignment,
            correction=correction,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            n_init=n_init,
            save_param_init=save_param_init,
            init_params=init_params,
            random_state=random_state,
            verbose=verbose,
            progress_bar=progress_bar,
            measurement_params=measurement_params,
            structural_params=structural_params)

    def get_class_weights(self, Y):
        sample_weight = None
        
        if self.auto_weight_class and Y is not None:
            max_y = np.nanmax(Y)
            Y = [max_y+1 if np.isnan(y) else y for y in Y] # assign nans a new category , assume nan means patient had hf before last time bin

            is_binary = len(set(Y)) == 2
            
            weight_strength = self.auto_weight_class
            neg_label = tty(Y).index[0]

            n = len(Y)
            n_pos = sum([1 if y != neg_label else 0 for y in Y])
            n_neg = n - n_pos
            pos_w = 1*(1-weight_strength) + (n_neg/n)*weight_strength
            neg_w = 1*(1-weight_strength) + (n_pos/n)*weight_strength
            w_sum = pos_w+neg_w
            pos_w /= w_sum
            neg_w /= w_sum
            sample_weight = [ neg_w if v == neg_label else pos_w for v in Y]


            
            #             continue                    
            #         c_n_y = sum([1 if y == c_y else 0 for y in Y])
            #         c_ny_prop = n_pos/c_n_y
            #         y_props += [c_ny_prop]
            #         y_ws += [(pos_w*c_ny_prop)/2]

            #     prop_sum = sum(y_props)
            #     w_sum = sum(y_ws)
            #     y_ws = list(np.array(y_ws)/w_sum)

            #     sample_weight = [ y_ws[y_cats.index(y)] for y in  Y ]
            #tty(sample_weight)
                
        return sample_weight


    def fit(self, X, Y=None, sample_weight=None, y=None):
        sample_weight = self.get_class_weights(Y if Y is not None else y)  # use weights even if not fitting a structural model
        if not self.use_outcome:
            Y = None
            y = None
        self.feature_names_in = cns(X)
        return super().fit(X, Y, sample_weight, y) 

    
    def orig_score(self, X, Y=None, sample_weight=None):
        check_is_fitted(self)
        if Y is not None:
            sample_weight = self.get_class_weights(Y)
            sample_weight = _check_sample_weight(sample_weight, X, dtype='float64', copy=T)
        X,Y = self._check_x_y(X,Y)

        if not self.use_outcome:
            Y = None

        avg_ll, _ = self._e_step(X, Y=Y, sample_weight = sample_weight)

        return avg_ll

    def score(self, X, Y=None, sample_weight=None, verbose=F, specific_cats=[], use_bic=F, Y_bic =None):
        
        # if not self.use_outcome:
        #     Y = None
        #     y = None
        # bic = -2 * self.orig_score(X,Y) * X.shape[0] + self.n_parameters * np.log(X.shape[0])
        
        n = nrow(X)
        if type(Y) == pd.DataFrame:
            Y = Y.iloc[:, 0].values
        min_y = np.nanmin(Y)
        no_Y = Y is None or min(Y) == max(Y)
        Y_bic = Y_bic if Y_bic is not None else Y
        if no_Y:
            aic = -2 * self.orig_score(X) * X.shape[0] + self.n_parameters * 2
            bic = -2 * self.orig_score(X) * X.shape[0] + self.n_parameters * np.log(X.shape[0])
            return -bic if use_bic else -aic
        else:
            bic = -2 * self.orig_score(X, Y_bic) * n + self.n_parameters * np.log(n)
            aic = -2 * self.orig_score(X,Y_bic) * n + self.n_parameters * 2
        
        max_y = np.max(Y)
        cats = list(set(Y))
        n_cats = len(cats)
        assert n_cats > 1
        labels = self.predict_class(X)
        # stratified multiple resample to compute sil score 
        # x0s = X[Y == 0]
        # x1s = X[Y == 1]
        # n_samples = 10
        # n0s = x0s.shape[0]
        # n1s = x1s.shape[0]
        
        # n0s_sample = n1s
        # n1s_sample = n1s
        # sil_scores = []
        # for i in range(n_samples):
        #     c_x0s = resample(x0s, n_samples = n0s_sample)
        #     c_x1s = resample(x1s, n_samples = n1s_sample)
        #     c_x = np.vstack([c_x0s, c_x1s])
        #     #c_y = np.concatenate([np.zeros(x0s.shape[0]), (np.zeros(x1s.shape[0])+1) ])
        #     c_labels = self.predict_class(c_x)
        #     sil_sc = silhouette_score(c_x, c_labels, metric="cosine")
        #     sil_scores+= [sil_sc]

        # logger(f"sil score (sd) = {np.mean(sil_scores):0.2f}({np.std(sil_scores):0.2f})") 
        # sil_sc #  interpretation of score : 1 = best; 0 = rubbish;  <0 = re-evaluate life goals
        # return np.mean(sil_scores)
        
        uniq_labs = sorted(set(labels))
        idxs_per_lab = [[i for i,l in enumerate(labels) if l == c_l] for c_l in uniq_labs]
        sizes = pd.Series(labels).value_counts().sort_index().to_numpy()
        masses = sizes / n
        cat_sizes = try_table(Y)
        null_cat = list(cat_sizes.to_dict().keys())[cat_sizes.argmax()]
        if specific_cats != []:
            nn_cats = specific_cats
        else:
            nn_cats = [ [c] for c in cats if c != null_cat]
        c_scores = []
        c_nposes = []
        
        for nn_cat in nn_cats:
            nPoss = np.array([ sum([ 1 if v in nn_cat else 0 for v in np.take(Y, c_idxs) ] ) for c_idxs in idxs_per_lab])
            npos_all = sum(nPoss)
            c_nposes += [npos_all]
            base_inci = npos_all / len(Y)
            inci_per_lab = nPoss / sizes
            npos_mases = [ (idx,p,m) for idx,i,m,p in zip(range(len(nPoss)),inci_per_lab,masses,nPoss) if is_cluster_interesting(i, m, p, base_inci) ] # find clusters with sufficient mass and incidence

            incis_masses = sorted( [(x,m) for x,m  in zip(inci_per_lab,masses)],  reverse=T, key=lambda l:l[0])
            cutoff = 0
            c_inci = 0
            c_mass = 0
            prev_inci = c_inci
            prev_mass = c_mass 
            while cutoff < len(incis_masses):
                prev_inci = c_inci
                prev_mass = c_mass
                cutoff+=1
                c_mass = sum([m for _,m in incis_masses[:cutoff]])
                c_inci = sum([i*m for i,m in incis_masses[:cutoff]]) / c_mass
                if c_inci < base_inci*3:
                    break
            cutoff-=1
            percent_pos_capured = prev_mass*prev_inci*n / npos_all*100
            c_score = percent_pos_capured
            if verbose:
                dbg = sorted([ (round(i,3),round(m,3),p) for i,m,p in zip(inci_per_lab,masses,nPoss) ], key=lambda x: -x[0])
                incis_str = "; ".join([ f"{100*im[0]:0.1f}%({100*im[1]:0.1f}%) {'<|CUTOFF' if idx == cutoff else ''}" for idx,im  in  enumerate(incis_masses)])
                logger(f"Incidences (mass) {incis_str}")
                logger(f"highest incidence clusters = (in/ms/npos){dbg[:2]} base inci {base_inci*100:0.1f}%")
            n_clusters = len(npos_mases)
            if npos_mases == []:
                npos_mases = [(-1,0,1e-16)]
            c_npos = sum([p for _,p,_ in npos_mases])
            c_coverage = (c_npos/npos_all)  # cvr:= what % of all positive cases does the current cluster capture?
            c_mass = sum([m for _,_,m in npos_mases])
            c_idxs = [i for i,_,_ in npos_mases]
            c_inci = c_npos/(c_mass*n)
            # c_score = (c_coverage*20 + c_inci )/21  # if coverage drops by 5%, thats similar score change to losing 1% incidence, e.g., C30% I5% =~ C25% I6% =~ C35% I4% 
            # c_score -= max(n_clusters-4, 0)*0.01  # penalize for too many overly-specific clusters (overfitting)
            c_scores += [c_score]
            if verbose:
                logger(f"cat{nn_cat}: (SCR:{c_score:0.2f}); cvr:{c_coverage*100:0.2f}%; ({n_clusters})clstrs:{c_idxs}; mss:{c_mass*100:0.2f}%; inc:{c_inci*100:0.2f}%; bs_inc:{base_inci*100:0.2f}%")
        if len(c_scores) == 1:
            c_score = c_scores[0]
        else:
            ws = [ (sum(c_nposes)-x)/sum(c_nposes) for x in c_nposes]
            ws = (1+np.array(ws) )/(1+len(ws))
            c_score = np.average(c_scores, weights=ws)
        # what if we just score on incidence and mass?
        #logger(f"BIC: {bic:1.3e}")
        logger(f"BIC:{bic:1.3e} AIC:{aic:1.3e} SCORE:{c_score:0.3f}")
        # Cluster-count/model selection is AIC-driven by default (per manuscript
        # Methods: "optimal number of clusters was selected using the Akaike
        # Information Criterion"), with BIC available via use_bic=True. Both
        # follow the 'higher score is better' convention used elsewhere in this
        # codebase (see best_train_idxs = max(...) in GaussMMStepMix.py).
        return -bic if use_bic else -aic
        # return c_score

