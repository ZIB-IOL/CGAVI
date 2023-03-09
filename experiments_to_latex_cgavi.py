from global_ import tos_, border_type_, data_sets_performance_gb_, data_sets_performance_bb_
from src.experiment_analysis.experiment_analysis import table_results

hp_bpcg_wihb_gb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'term_ordering_strategy': tos_, 'border_type': "gb",
                   'inverse_hessian_boost': 'weak'}
hp_cgavi_ihb_gb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'term_ordering_strategy': tos_, 'border_type': "gb",
                   'inverse_hessian_boost': 'true'}
hp_agdavi_ihb_gb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'term_ordering_strategy': tos_, 'border_type': "gb",
                    'inverse_hessian_boost': 'true'}
hp_abm_gb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'term_ordering_strategy': tos_, 'border_type': "gb",
             'inverse_hessian_boost': 'false'}
hp_vca = {'algorithm': 'vca'}
hp_svm = {'algorithm': 'svm'}

hp_bpcg_wihb_bb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'term_ordering_strategy': tos_, 'border_type': "bb",
                   'inverse_hessian_boost': 'weak'}
hp_cgavi_ihb_bb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'term_ordering_strategy': tos_, 'border_type': "bb",
                   'inverse_hessian_boost': 'true'}
hp_agdavi_ihb_bb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'term_ordering_strategy': tos_, 'border_type': "bb",
                    'inverse_hessian_boost': 'true'}
hp_abm_bb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'term_ordering_strategy': tos_, 'border_type': "bb",
             'inverse_hessian_boost': 'false'}
hps_no_svm = [hp_bpcg_wihb_gb, hp_cgavi_ihb_gb, hp_agdavi_ihb_gb, hp_abm_gb, hp_vca]
hps = hps_no_svm + [hp_svm]
print("############################################################################################")
print("# Performance")
print("############################################################################################")

for key in ["time_train", "time_test", "time_hyper", "error_train", "error_test", 'avg_degree', "G + O",
            "total_sparsity"]:
    print("-------------------------------------------------------------------------")
    print("key: ", key)
    if key in ['avg_degree', "G + O", "total_sparsity"]:
        table_results(data_sets_performance_gb_, hps_no_svm, key, ordering=False)
    else:
        table_results(data_sets_performance_gb_, hps, key, ordering=False)
print("############################################################################################")

print("")
print("")
print("")
print("")
print("")
print("")

print("############################################################################################")
print("# Borders")
print("############################################################################################")
hps_bases = [hp_bpcg_wihb_gb, hp_bpcg_wihb_bb, hp_cgavi_ihb_gb, hp_cgavi_ihb_bb, hp_agdavi_ihb_gb, hp_agdavi_ihb_bb,
             hp_abm_gb, hp_abm_bb]
for key in ["time_train", "time_test", "time_hyper", "error_train", "error_test", 'avg_degree', "G + O",
            "total_sparsity"]:
    print("-------------------------------------------------------------------------")
    print("key: ", key)
    table_results(data_sets_performance_bb_, hps_bases, key, ordering=False)
print("############################################################################################")
