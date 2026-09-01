root_dir="/app"
cd "$root_dir/src"

python -u sandbox_generic.py T F && \
# python -u analyse_results_ICPC_codes.py T F && \
# python -u analyse_results_AIC_vars.py T F && \
# python -u analyse_results_ATC_codes.py T F && \
# python -u analyse_results_PRAC_ids.py T F && \
# python -u analyse_results_TARGETHF_score.py T F && \
# python -u analyse_results_missed_HF.py T F && \
# python -u analyse_results_var_distributions.py T F && \
echo DONE