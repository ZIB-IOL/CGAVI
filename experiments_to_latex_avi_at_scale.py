from global_ import tos_, border_type_, data_sets_performance_
from src.experiment_analysis.experiment_analysis import table_results

hp_cg_ihb_rev = {'algorithm': 'oavi', 'oracle_type': 'CG', 'term_ordering_strategy': "rev pearson",
                 'border_type': border_type_, 'inverse_hessian_boost': 'full'}

hp_cg_ihb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'term_ordering_strategy': tos_, 'border_type': border_type_,
             'inverse_hessian_boost': 'full'}
hp_agd_ihb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'term_ordering_strategy': tos_, 'border_type': border_type_,
              'inverse_hessian_boost': 'full'}
hp_bpcg_wihb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'term_ordering_strategy': tos_, 'border_type': border_type_,
                'inverse_hessian_boost': 'weak'}
hp_abm = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'term_ordering_strategy': tos_, 'border_type': border_type_,
          'inverse_hessian_boost': 'false'}
hp_vca = {'algorithm': 'vca'}
hp_svm = {'algorithm': 'svm'}
hps_no_svm = [hp_cg_ihb, hp_agd_ihb, hp_bpcg_wihb, hp_abm, hp_vca]
hps = hps_no_svm + [hp_svm]

print("############################################################################################")
print("# Term Ordering")
print("############################################################################################")
table_results(data_sets_performance_, [hp_cg_ihb, hp_cg_ihb_rev], "error_test", ordering=True)
print("############################################################################################")
print("")
print("")
print("############################################################################################")
print("# Performance")
print("############################################################################################")

for key in ["time_train", "time_test", "time_hyper", "error_train", "error_test", 'avg_degree', "G + O",
            "total_sparsity"]:
    print("-------------------------------------------------------------------------")
    print("key: ", key)
    if key in ['avg_degree', "G + O", "total_sparsity"]:
        table_results(data_sets_performance_, hps_no_svm, key, ordering=False)
    else:
        table_results(data_sets_performance_, hps, key, ordering=False)

print("###########################################################################################")