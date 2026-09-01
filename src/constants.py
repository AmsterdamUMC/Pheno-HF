import pandas as pd
from pathlib import Path
import numpy as np


SENS_ANALYSIS_UNBIAS_FLWP_START = False
# NOTE: the manuscript's Sensitivity Analysis restricts to patients with
# >=3 consultations and >=6 months of observation (matching the inline
# comments in try_utils.py's __init_pats_dict_time_unbiased, e.g. "require
# at least 3 consults per record" / "minimum observation window (6
# months)") -- but as set here, MIN_N_CONSULTS/MIN_FLWP_WIN_DAYS don't
# apply either restriction. Left unchanged pending confirmation of intended
# values; bump to 3 / round(6*30.44) if reproducing that analysis exactly.
MIN_N_CONSULTS = 1
MIN_FLWP_WIN_DAYS = 1

DAYS_IN_YEAR = 365.2422

VAR_OUTCOME = 'adj_HF_diag'
VAR_FOLLOW_UP_DATE = 'follow_up_LAST'
VAR_OBSERVATION_END = 'follow_up_effective'

DATE_ARBITRARY_OFFSET_YEAR = 1970
DATE_ARBITRARY_OFFSET_TIMESTAMP = pd.Timestamp(f'{DATE_ARBITRARY_OFFSET_YEAR}-01-01')
DATE_ARBITRARY_OFFSET_DAYS = round(DATE_ARBITRARY_OFFSET_TIMESTAMP.timestamp() / (60*60*24))

FOLLOW_UP_HFPOS_CENS_WINDOW = 90 # 3 months
FOLLOW_UP_PERIOD_DAYS = 730 + FOLLOW_UP_HFPOS_CENS_WINDOW # two years + 3 months delay window

DATE_LAST_POSSIBLE_FOLLOWUP_DAYS = (pd.Timestamp('2022-12-31').timestamp() / (60*60*24)) - DATE_ARBITRARY_OFFSET_DAYS

ADJ_TIME_START_YEAR = 2017 # select patients with new diagnosis only after this year
ADJ_TIME_START_TIMESTAMP = pd.Timestamp(f'{ADJ_TIME_START_YEAR}-01-01')

# Cohort-selection
COHORT_TIME_START_YEAR = 2010 # assumes start jan 1 # 2015 new start
COHORT_TIME_END_YEAR = 2023  # assumes end dec 31 #  2019 new end

COHORT_TIME_START_TIMESTAMP = pd.Timestamp(f'{COHORT_TIME_START_YEAR}-01-01')
COHORT_TIME_END_TIMESTAMP = pd.Timestamp(f'{COHORT_TIME_END_YEAR}-12-31')

COHORT_TIME_END_DAYS = round(COHORT_TIME_END_TIMESTAMP.timestamp() / (60*60*24))
COHORT_TIME_START_DAYS = round(COHORT_TIME_START_TIMESTAMP.timestamp() / (60*60*24))

THRESHOLD_MIN_AGE_DAYS = 35*DAYS_IN_YEAR # i.e., 35 years 

# a year is too much, could we consider doing it by month? 
# i.e., debug = two times 2 years (so collect from 0-24 months, predict 25-48 months)
# no debug = 2 times 1 year and 1 times 2 year (so collect from 0-12, 13-24, predict 25-48)
# UPDATE 26-mar 2024: we want (from most recent time to last) : 1y (prediction period) , 6m, , 12m, 18m, 24m, 36m, 48m (but we want these overlapping!)
# TIME_BINS_DAYS = [[730]*2, [365]*2 + [730]*1] # used as TIME_BINS_DAYS[IS_DEBUG]
# Update 1-apr 2025: use one single bin, and one window for censorship , treat [0][1] as boundries for ep_date - first_ep_dt
TIME_BINS_DAYS = [ (0, FOLLOW_UP_PERIOD_DAYS- FOLLOW_UP_HFPOS_CENS_WINDOW), (FOLLOW_UP_PERIOD_DAYS-FOLLOW_UP_HFPOS_CENS_WINDOW, FOLLOW_UP_PERIOD_DAYS)]  # [(period_start, period_end)+]


# USE_MEASUREMENTS = False  # Note: these have now be replaced by namespace input args for each script
# USE_MEDICATIONS = False
SHOULD_SOEP_SEPERATE_TEXTS = False # if True (default), keep S/O/E/P/X as separate text vars; if False, use Noman's approach: one var containing ep_desc+S+O+E+P+X concatenated
d2v_dm = 0  # Doc2Vec training algorithm: 0 = DBOW (currently used), 1 = DM
# DBOW ignores word order in a doc, more mem efficient and faster, can be ok accuracy with large-scale or low-resource settings

OHE_REMOVE_FIRST_CATEGORY = True

# Medication Measurements filename infix, note: getting replaced by namespace arg
MM_INFIX = "" #  f"{'_msrs' if USE_MEASUREMENTS else ''}{'_meds' if USE_MEDICATIONS else ''}"
REDUCE_N_TOPICS = True

ICPC_CODES_HF = ['K77', 'K77.01', 'K77.02', 'K84.03']
ICPC_TEXT_REGEXES_HF = [r"hart *falen", r"cardio *m[yi]opath?ie", r"dec(\.?|omp\w*\.?) *cordis" ]

infile_Lukas_adjudicated = 'all_adjudicated_HF_Lukas.pkl'
infile_ICPC_AF_HF_VHD_tagged = ''

# Hyperparams
STEPMIX_MINIMUM_CLUSTER_MASS = 0.005

# for text embedding
EMBEDDING_MODELS = ['doc2vec', 'universal-sentence-encoder-multilingual-large', 'universal-sentence-encoder-multilingual', 'paraphrase-multilingual-MiniLM-L12-v2',
						'NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers', 'RobertaForMaskedLM_embedding_model']
EMBEDDING_MODELS_TO_TRY = ['doc2vec', 'RobertaForMaskedLM_embedding_model', 'NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers']

EMBEDDING_MODEL_METADATA = {
						'doc2vec': {
							'short_name': 'd2v',
							'is_sbert': False,
							'is_callable': False,
							'is_custom' : False,
							# main experiemtns results
							'doc_embeddings_file' : {},#{ "0_24_epj_text_" : "try_embeddings_d2v2025-12-31 18:00:14.421074.pkl" },
							'umap_embeddings_file' : {}, #{"0_24_epj_text_" : "try_umap_embeddings_2026-01-03 11:32:00.359326.pkl"}, 
							'umap_knn_cache_file' : {}, #{ "0_24_epj_text_" : "umap_knn_dists.pkl" },
							'hdbscan_labels_file' : {}, # {"0_24_epj_text_" : "try_hdbscan_labs_2026-01-03 11:33:07.398915.pkl"}, 


							'doc_embeddings_file_SUBSAMPLED' : {},#{"0_24_epj_text_" : "try_embeddings_d2v2025-04-07 10:13:26.922378.pkl" },
							'umap_embeddings_file_SUBSAMPLED' : {}, #{ "0_24_epj_text_" : "try_umap_embeddings_2025-04-08 13:07:24.292203.pkl"},
							'umap_knn_cache_file_SUBSAMPLED' : {}, #{ "0_24_epj_text_" : "umap_knn_dists_3.pkl" },
							'hdbscan_labels_file_SUBSAMPLED' : {} #{ "0_24_epj_text_" : "try_hdbscan_labs_2025-04-08 13:07:37.093942.pkl" }
								},
						'universal-sentence-encoder-multilingual-large': {
							'short_name': 'use-mling-lrg',
							'is_sbert': False,
							'is_callable': False,
							'is_custom' : False,
							'doc_embeddings_file' : {},	
							'umap_embeddings_file' : {},
							'hdbscan_labels_file' : {},
							'doc_embeddings_file_SUBSAMPLED' : { },
							'umap_embeddings_file_SUBSAMPLED' : { },
							'hdbscan_labels_file_SUBSAMPLED' : { }
									},
						'universal-sentence-encoder-multilingual': {
							'short_name': 'use-mling',
							'is_sbert': False,
							'is_callable': False,
							'is_custom' : False,
							'doc_embeddings_file' : {},
							'umap_embeddings_file' : {},
							'hdbscan_labels_file' : {},
							'doc_embeddings_file_SUBSAMPLED' : { },
							'umap_embeddings_file_SUBSAMPLED' : { },
							'hdbscan_labels_file_SUBSAMPLED' : { }
									},
						'paraphrase-multilingual-MiniLM-L12-v2': {
							'short_name': 'parphras-mling',
							'is_sbert': True,
							'is_callable': False,
							'is_custom' : False,
							'doc_embeddings_file' : {},
							'umap_embeddings_file' : {},
							'hdbscan_labels_file' : {},
							'doc_embeddings_file_SUBSAMPLED' : { },
							'umap_embeddings_file_SUBSAMPLED' : { },
							'hdbscan_labels_file_SUBSAMPLED' : { }
									},
						'NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers': {
							'short_name': 'sbertnl',
							'is_sbert': True,
							'is_callable': False,
							'is_custom' : True,
							'doc_embeddings_file' : {},
							'umap_embeddings_file' : {},
							'hdbscan_labels_file' : {}
									},
						'RobertaForMaskedLM_embedding_model': {
							'short_name': 'medrobnl',
							'is_sbert': False,
							'is_callable': True,
							'is_custom' : True,
							'doc_embeddings_file' : {},
							'umap_embeddings_file' : {},
							'hdbscan_labels_file' : {},
							'doc_embeddings_file_SUBSAMPLED' : { },
							'umap_embeddings_file_SUBSAMPLED' : { },
							'hdbscan_labels_file_SUBSAMPLED' : { }
									}
						}


# Paths
# Configurable via env vars so this repo isn't tied to any one deployment's
# mount layout. Defaults match the original GP-EHR extract naming used in
# this project's docker/run setup (see README "Data layout").
import os
data_dir = Path(os.environ.get("PHENO_HF_DATA_DIR", "/data/current_extract"))
pqt_dir = data_dir/"parquet"
csv_dir = data_dir/"csv"
old_data_dir = Path(os.environ.get("PHENO_HF_PREV_DATA_DIR", "/data/previous_extract"))
old_pqt_dir = old_data_dir/"parquet"

# Aliases
T = True 
F = False 

# Main analysis-specific
SURROGATE_OUTCOME_COLUMNS_EPISODE_LEVEL = ['text_HF', 'icpc_HF']

# Pre-processing specific
TOPIC_DISTANCE_COLUMN_REGEX = ".*t\d+_.*_dist$"

STEP1_BATCH_SIZE = 10000
STEP1D1_BATCH_SIZE = 50000
PATIENT_COLS = ['age_days', 'patient_type', 'sex', 'postal_code', 'reg_date', 'dereg_date', 'dereg_cause', 'anonymous', 'missing',
					VAR_OUTCOME, VAR_FOLLOW_UP_DATE]
EPISODE_COLS = ['episode_start_date',
	'episode_end_date',
	'icpc_episode',
	'episode_attention',
	'episode_problem',
	'episode_status',
	'episode_description',
	'cvd_in_family',
	'coronary_artery_disease',
	'atrial_fibrillation',
	'heart_murmur',
	'valvular_heart_disease',
	'hypertension',
	'stroke',
	'copd',
	'diabetes_mellitus',
	'chronic_kidney_disease',
	'alcohol_abuse',
	'tobacco_use',
	'obesity',
	'material_deprivation',
	'text_HF',
	'icpc_HF'
	]
JOURNAL_COLS = ['journal_datetime', 'contact_type', 'icpc_episode', 'icpc_journal', 'icpc_s', 'icpc_o', 'icpc_e', 'icpc_p', 'icpc_x', 'text_s', 'text_o', 'text_e', 'text_p', 'text_x', 'dyspnea', 'edema', 'chest_complaints', 'palpitations', 'dizziness', 'syncope', 'tiredness']
MEDICATION_COLS = ['medication_datetime', 'atc_code', 'medication_txt']
MEASUREMENT_COLS = ['measurement_datetime', 'nhgnummer', 'measurement_txt']

MAPPING_COMPOSITE_KEYS = { 
                            'episode' : { 'key_nm' : 'ptnt_prc_epi_id',
                                          'key_comp' : ['person_id', 'practice_id', 'episode_id']
                                        },
                            'patient' : { 'key_nm' : 'ptnt_prc_id',
                                          'key_comp' : ['person_id', 'practice_id']
                                        },
                            'journal' : { 'key_nm' : 'ptnt_prc_jrnl_id',
                                          'key_comp' : ['person_id', 'practice_id', 'journal_id']
                                        },
							'measurement' : { 'key_nm' : 'ptnt_prc_msrm_id',
											  'key_comp' : ['person_id', 'practice_id', 'measurement_id']
											},
							'medication' : { 'key_nm' : 'ptnt_prc_med_id',
											  'key_comp' : ['person_id', 'practice_id', 'medication_id']
											},
                            # composite of composites
                            'journals_episodes' : { 'key_nm' : 'ptnt_prc_epi_id',
                                                    'key_comp' : ['ptnt_prc_id', 'episode_id']
                                                 }
                        }

CSV_DTYPES = {
	"episodes":  
		{
		"Unnamed: 0" : 'int64',
		"person_id" : 'int64',
		"practice_id" : 'int64',
		"episode_id" : 'int64',
		"import_id" : 'int64',
		"episode_start_date" : str,
		"episode_end_date" : str,
		"icpc_episode" : str,
		"episode_attention" : np.float64,
		"episode_problem" : np.float64,
		"episode_status" : str,
		"episode_description" : str,
		"cvd_in_family" : np.int32,
		"coronary_artery_disease" : np.int32,
		"atrial_fibrillation" : np.int32,
		"heart_murmur" : np.int32,
		"valvular_heart_disease" : np.int32,
		"hypertension" : np.int32,
		"stroke" : np.int32,
		"copd" : np.int32,
		"diabetes_mellitus" : np.int32,
		"chronic_kidney_disease" : np.int32,
		"alcohol_abuse" : np.int32,
		"tobacco_use" : np.int32,
		"obesity" : np.int32,
		"material_deprivation" : np.int32,
		"text_HF" : np.int32,
		"icpc_HF" : np.int32,
		"tokenized_episode_description" : str,
		"ptnt_prc_id" : str,
		"ptnt_prc_epi_id" : str,
		"ptnt_prc_epi_id" : str 
		},
	"journals":
		{ 
		"Unnamed: 0" : 'int64',
		"person_id" : 'int64',
		"practice_id" : 'int64',
		"journal_id" : 'int64',
		"import_id" : 'int64',
		"journal_datetime" : str,
		"episode_start_date" : str,
		"episode_id" : np.float64,
		"contact_id" : np.float64,
		"contact_type" : str,
		"icpc_episode" : str,
		"icpc_journal" : str,
		"icpc_s" : str,
		"icpc_o" : str,
		"icpc_e" : str,
		"icpc_p" : str,
		"icpc_x" : str,
		"text_s" : str,
		"text_o" : str,
		"text_e" : str,
		"text_p" : str,
		"text_x" : str,
		"dyspnea" : np.int32,
		"edema" : np.int32,
		"chest_complaints" : np.int32,
		"palpitations" : np.int32,
		"dizziness" : np.int32,
		"syncope" : np.int32,
		"tiredness" : np.int32,
		"ptnt_prc_id" : str,
		"ptnt_prc_epi_id" : str,
		"ptnt_prc_jrnl_id" : str
		},
	"measurements": {
		"Persoon_id" : 'int64',
		"Praktijk_id" : 'int64',
		"Import_id" : 'int64',
		"Bepaling_id" : 'int64',
		"Contact_id" : np.float64,
		"Journaal_id" : np.float64,
		"Episode_id" : np.float64,
		"Datum" : str,
		"Episode_icpc" : str,
		"NHGnummer" : str,
		"Vraagtype" : str,
		"Uitslag_type" : str,
		"Soort" : str,
		"Omschrijving" : str,
		"Eenheid" : str,
		"Referentie_min" : str,
		"Referentie_max" : str,
		"Afwijkende_uitslag" : str,
		"Memo_mat_bijz" : str,
		"Bijzonder" : str,
		"Materiaal" : str,
		"Memo" : str,
		"Opmerking2" : str,
		"Toelichting2" : str,
		"Uitslag2" : str,
		"Uitslag_tekst2" : str
					},
	"medications": {
		"Persoon_id": 'int64',
		"Praktijk_id": 'int64',
		"Import_id": 'int64',
		"Recept_id": 'int64',
		"Contact_id": np.float64,
		"Journaal_id": np.float64,
		"Episode_id": np.float64,
		"Episode_icpc": str,
		"Specialisme": str,
		"Actueel": str,
		"Voorschrijfdatum": str,
		"Afleverdatum": str,
		"Einddatum": str,
		"Atc_code": str,
		"Atc_omschrijving": str,
		"Omschrijving": str,
		"Hoeveelheid": str,
		"Aflever_eenheid": str,
		"Product_sterkte": str,
		"Chronisch": str,
		"Volgnummer_herhaling": str,
		"Zindex_nummer": str,
		"Prk": str,
		"Gpk": str,
		"Hpk": str,
		"Toedieningsweg": str,
		"Gebruiksvoorschrift2": str,
		"Vrije_tekst2": str
	}

}

TINY_OFFSET = np.finfo(np.float64).tiny*10000000


TXT_ENTRY_SEP = " <<>> "