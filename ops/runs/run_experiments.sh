root_dir="/app"
cd "$root_dir/src"

INCLUDE_MEDICATIONS=T
INCLUDE_MEASUREMENTS=F
INCLUDE_MEDICATIONS_TEXT=F
INCLUDE_MEASUREMENTS_TEXT=F

me=$(basename "$0")
logfile="$root_dir/log/pipeline/$me.log"
echo pipeline output written to $logfile
echo begin > $logfile

echo "***EXPERIMENT PIPELINE***"
echo -e " <BEGIN> 1_pre_process -=> 1d1_pre_process -=> 2_pre_process "
echo -e " 	-=> 2d1_pre_process_text -=> A_runner -=> 3_pre_process "
echo -e " 		-=> 4_pre_process -=> 4d1_pre_process -=> 5_dim_reduce"
echo -e " 			 -=> B_runner -=> GMM_preprocess -=> C_runner "
echo -e " 			 	-=> analyse_results"
echo -e " <END>\n\n"


echo -e "1_pre_process.py :"
echo -e "	- tag patients, add episodes+consults to patient-centered dicts"
echo -e "1d1_pre_process.py :"
echo -e "	- apply filters for: cohort, features; add meds+measures to patient-centered dicts"
echo -e "2_pre_process.py :"
echo -e "	- merge two patient-level dicts from ANH and AHA"
echo -e "2d1_pre_process_text.py :"
echo -e "	- parse docs per patient; apply cohort masking of text;"
echo -e "A_runner.py :"
echo -e "	- further massage text; run Top2Vec; generate t2v topic-based metrics;"
echo -e "3_pre_process.py :"
echo -e "	- flatten patient representation for ALL non-text vars "
echo -e "4_pre_process.py :"
echo -e "	- clean-up after 3_pre_process. Write output to single sparse matrix."
echo -e "4d1_pre_process.py :"
echo -e "	- MaxAbsScaler, age derived based on last follow-up date and birth date, remove nans"
echo -e "5_dim_reduce.py :"
echo -e "	- group variables for sparse coded variables (icpc, ATC)"
echo -e "B_runner.py :"
echo -e "	- perform variable selection"
echo -e "GMM_preprocess.py :"
echo -e "	- RobustScaler"
echo -e "C_runner.py :"
echo -e "	- run Gaussian Mixture Model"
echo -e "analyse_results.py :"
echo -e "	- generate summary of clusters from GMM "


# Only used for sensitivity analysis when matching control flwp start dates with cases
# Otherwise start from 1_pre_process
# python -u 0_pre_process.py $1 $2  >> $logfile && \



# python -u 1_pre_process.py $1 $2  \
# 	tagging_mode=F  \
# 	target_condition= >> $logfile && \
# python -u 1d1_pre_process.py $1 $2 \
# 	include_medications=$INCLUDE_MEDICATIONS \
# 	include_measurements=$INCLUDE_MEASUREMENTS >> $logfile && \
# python -u 2_pre_process.py $1 $2 \
# 	infiles="pats_dict_AHA_1d1;pats_dict_ANH_1d1" \
# 	outfile=pats_dict_merged >> $logfile && \
# python -u 2d1_pre_process_text.py $1 $2  \
# 	include_medications=$INCLUDE_MEDICATIONS \
# 	include_measurements=$INCLUDE_MEASUREMENTS \
# 	include_medications_text=$INCLUDE_MEDICATIONS_TEXT \
# 	include_measurements_text=$INCLUDE_MEASUREMENTS_TEXT \
# 	infile=pats_dict_merged \
# 	outfile=pats_dict_text >> $logfile && \
# # ***************************************************
# # Hey there bud, whatcha doin?
# # you about to re-run UMAP there are ya?
# # DID YOU UPDATE constants.py FOR THE KNN CACHE?
# # didn't think so, go now!
# # ***************************************************
# python -u A_runner.py $1 $2 \
# 	include_medications_text=$INCLUDE_MEDICATIONS_TEXT \
# 	include_measurements_text=$INCLUDE_MEDICATIONS_TEXT \
# 	infile=pats_dict_text \
# 	outfile=temp_pats_dict_merged_text_emb4 \
# 	interim_out_file=tv2_mdist.1_nneigh200_4 >> $logfile 2>&1 && \
# python -u 3_pre_process.py $1 $2 \
#  include_medications=$INCLUDE_MEDICATIONS \
#  include_measurements=$INCLUDE_MEASUREMENTS \
#  infile=temp_pats_dict_merged_text_emb4 \
#  outfile=fs_flat \
#  reuse_batchfiles=T \
#  derive_specific_attrs= >> $logfile && \
# python -u 4_pre_process.py $1 $2 \
#  infile=fs_flat \
#  append_attributes= \
#  infile_append= \
#  infile_nbatches=cluster_fs1_n_batches \
#  infile_pats_dict=pats_dict_merged\
#  outfile=temp_cluster_fs1_csr >> $logfile && \
# python -u 4d1_pre_process.py $1 $2\
#  infile=temp_cluster_fs1_csr \
#  outfile=temp_cluster_fs1_crs_scaled \
#  plot_histograms=F >> $logfile && \
# python -u 5_dim_reduce.py $1 $2 \
#  infile=temp_cluster_fs1_crs_scaled \
#  infile_colnames=temp_cluster_fs1_csr_colnames \
#  infile_code_lookups=cluster_fs1_lookup_dicts \
#  outfile=temp_cluster_fs1_reduced >> $logfile && \
# python -u B_runner.py $1 $2 \
#  infile=temp_cluster_fs1_reduced \
#  outfile=10x4_cluster_fs1_selected  \
#  use_text_vars=T>> $logfile && \
# python -u GMM_preprocess.py $1 $2 \
#  infile=10x4_cluster_fs1_selectedn_etors5000_maxepth5_minplit100_maxodes200  \
#  infile_code_lookups=cluster_fs1_lookup_dicts \
#  outfile=10x4_cluster_fs1_df_scaled \
#  plot_histograms=T >> $logfile && \
# python -u C_runner.py $1 $2  \
#  infile=10x4_cluster_fs1_df_scaled \
#  outfile=gmm_output  >> $logfile && \
python -u analyse_results.py $1 $2 \
  infile=_12_non-nested_gmm_output \
  outfile=12_analyse_results \
  goodness_of_fit_metric=inci \
  infile_scale_maxabs_lookup=temp_cluster_fs1_crs_scaled_scale_multipliers \
  infile_scale_robust_lookup=10x4_cluster_fs1_df_scaled_scale_multipliers \
  infile_fs1_ohe_lookup=cluster_fs1_lookup_dicts \
  infile_t2v_model=temp_pats_dict_merged_text_emb4 \
  plot_wordclouds=T >> $logfile && \
echo done >> $logfile 2>&1

# GMM_preprocess
# no subsample
#  infile=cluster_fs1_selectedn_etors5000_maxepth5_minplit100_maxodes200 \ 
# yes subsample 
# infile=cluster_fs1_selectedn_etors100_maxepth2_minplit20_maxodes20