from types import SimpleNamespace as nspace

	
ns_adjudication_preprocessor = nspace(**{
	"required_extra_args" : ["target_condition"]
	})

ns_00_CohortExtraction = nspace(**{
	"required_extra_args" : []
	})

ns_ty_merge_anh_aha = nspace(**{
	"required_extra_args" : ["target_condition"]
	})

ns_0_pre_process = nspace(**{
	"required_extra_args" : []
	})

ns_1_pre_process = nspace(**{
	"required_extra_args" : ["target_condition", 'tagging_mode']
	})

ns_1d1_pre_process = nspace(**{
	"required_extra_args" : ["include_medications", 'include_measurements']
	})

ns_2_pre_process = nspace(**{
	"required_extra_args" : ["infiles", "outfile"]
	})

ns_2d1_pre_process_text = nspace(**{
	"required_extra_args" : ["include_medications", "include_measurements","include_medications_text", 'include_measurements_text', 'infile', 'outfile']
	})

ns_A_runner = nspace(**{
	"required_extra_args" : ["include_medications_text", 'include_measurements_text', "infile", "outfile", "interim_out_file"]
	})

ns_3_pre_process = nspace(**{
	"required_extra_args" : ["include_medications", 'include_measurements', "infile", "outfile", "reuse_batchfiles", "derive_specific_attrs"]
	})

ns_4_pre_process = nspace(**{
	"required_extra_args" : ["infile", "infile_nbatches", "append_attributes", "infile_append", "infile_pats_dict", "outfile"]
	})

ns_4d1_pre_process = nspace(**{
	"required_extra_args" : ["infile", "outfile", "plot_histograms"]
	})

ns_5_dim_reduce = nspace(**{
	"required_extra_args" : ["infile", "infile_colnames", "infile_code_lookups", "outfile"]
	})

ns_dim_reduce_utils = nspace(**{
	"required_extra_args" : []
	})

ns_B_runner = nspace(**{
	"required_extra_args" : ["infile", "outfile", "use_text_vars"]
	})

ns_GMM_preprocess = nspace(**{
	"required_extra_args" :  ["infile", "infile_code_lookups", "outfile", "plot_histograms"]
	})

ns_C_runner = nspace(**{
	"required_extra_args" : ["infile", "outfile"]
	})

ns_C_runner2 = nspace(**{
	"required_extra_args" : ["infiles", "outfile"]
	})

ns_analyse_results = nspace(**{
	"required_extra_args" : ["infile", "outfile", 
		"goodness_of_fit_metric", 
		"infile_scale_maxabs_lookup",
		"infile_scale_robust_lookup",
		"infile_fs1_ohe_lookup",
		"infile_t2v_model",
		"plot_wordclouds"]
	})


ns_analyse_results_util = nspace(**{
	"required_extra_args" : [],
	'res_mod' : '10x4',
	'nested' : {
		'GMM_file' : "_12_nested_gmm_output.pkl"
	},
	'10x4' : {
		'GMM_file' : "_12_non-nested_nested_gmm_output.pkl"
	},
	'pats_dict_file' : "pats_dict_merged.pkl",
	'id_col' : 'id_str',
	'outcome_col' : 'event',
	'target_hf_col' : 'targetHF_score',
	'targetHF_cols' : [ 
		'decades_age', 
		'heart_murmur',
		'coronary_artery_disease',
		'chronic_kidney_disease',
		'male',
		'obesity',
		'copd',
		'atrial_fibrillation',
		'diabetes_mellitus',
		'valvular_heart_disease',
		'hypertension',
		'tobacco_use',
		'cvd_in_family',
		'material_deprivation',
		'alcohol_abuse',
		'stroke' 
	]
})


_locals = dir()


def get_ns_name(target_filepath):
	# print(_locals)
	script_name = target_filepath.split("/")[-1].split(".")[0]
	# print(script_name)
	return [ x for x in _locals if x == f'ns_{script_name}'][0]


def parse_ns_val_bool(v):
	v_map = { 'T' : True,
			  'True' : True,
			  'F': False,
			  'False': False
			  }
	return v_map[v] if v in v_map else v
