from global_ import tos_, border_type_
from src.experiment_analysis.experiment_analysis import table_results

hp_pcg = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'term_ordering_strategy': tos_, 'border_type': border_type_,
          'inverse_hessian_boost': 'false'}
hp_agd = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'term_ordering_strategy': tos_, 'border_type': border_type_,
          'inverse_hessian_boost': 'false'}
hp_abm = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'term_ordering_strategy': tos_, 'border_type': border_type_,
          'inverse_hessian_boost': 'false'}
hp_avi = {'algorithm': 'avi', 'term_ordering_strategy': tos_, 'border_type': border_type_, }
hp_vca = {'algorithm': 'vca'}
hp_svm = {'algorithm': 'svm'}
hps_no_svm = [hp_pcg, hp_agd, hp_abm, hp_avi, hp_vca]
hps = hps_no_svm + [hp_svm]
data_sets = ['bank']

print("############################################################################################")
print("# Performance")
print("############################################################################################")

for key in ["time_train", "time_test", "time_hyper", "error_train", "error_test", 'avg_degree', "G + O",
            "total_sparsity"]:
    print("-------------------------------------------------------------------------")
    print("key: ", key)
    if key in ['avg_degree', "G + O", "total_sparsity"]:
        table_results(data_sets, hps_no_svm, key, ordering=False)
    else:
        table_results(data_sets, hps, key, ordering=False)
print("############################################################################################")