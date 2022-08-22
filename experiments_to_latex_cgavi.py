from global_ import tos_, border_type_, data_sets_performance_gb_, data_sets_performance_bb_
from src.experiment_analysis.experiment_analysis import table_results

hp_pcg = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'term_ordering_strategy': tos_, 'border_type': "gb",
          'inverse_hessian_boost': 'false'}

hp_agd = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'term_ordering_strategy': tos_, 'border_type': "gb",
          'inverse_hessian_boost': 'false'}

hp_abm = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'term_ordering_strategy': tos_, 'border_type': "gb",
          'inverse_hessian_boost': 'false'}

hp_avi = {'algorithm': 'avi', 'term_ordering_strategy': tos_, 'border_type': "gb", }
hp_vca = {'algorithm': 'vca'}
hp_svm = {'algorithm': 'svm'}
hp_bpcg = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'term_ordering_strategy': tos_, 'border_type': "gb",
           'inverse_hessian_boost': 'false'}

hp_pcg_bb = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'term_ordering_strategy': tos_, 'border_type': "bb",
             'inverse_hessian_boost': 'false'}
hp_agd_bb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'term_ordering_strategy': tos_, 'border_type': "bb",
          'inverse_hessian_boost': 'false'}
hp_abm_bb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'term_ordering_strategy': tos_, 'border_type': "bb",
          'inverse_hessian_boost': 'false'}
hps_no_svm = [hp_pcg, hp_agd, hp_abm, hp_vca]
hps = hps_no_svm + [hp_svm]


# data_sets_performance_gb_ = ['bank', 'credit', 'htru']
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



hps_bases = [hp_pcg, hp_pcg_bb, hp_agd, hp_agd_bb, hp_abm, hp_abm_bb]

for key in ["time_train", "time_test", "time_hyper", "error_train", "error_test", 'avg_degree', "G + O",
            "total_sparsity"]:
    print("-------------------------------------------------------------------------")
    print("key: ", key)
    table_results(data_sets_performance_bb_, hps_bases, key, ordering=False)
print("############################################################################################")
