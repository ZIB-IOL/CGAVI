from global_ import tos_, psis_, Cs_, data_sets_performance_
from src.experiment_analysis.experiment_analysis import table_results

from src.experiment_setups.experiment_setups import perform_experiments

taus_ = [0.0]
psis_ = [0.00005]
hp_avi = {'algorithm': 'avi', 'term_ordering_strategy': 'deglex', 'psi': psis_, 'C': Cs_, 'tau': taus_}
hp_pearson = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': 'pearson',
                  'inverse_hessian_boost': 'full'}


perform_experiments(data_sets_performance_, hp_pearson, saving=False)
# perform_experiments(data_sets_performance_, hp_pearson_rev, saving=True)
# table_results(data_sets, [hp_cg_ihb, hp_cg_ihb_rev, hp_agd_ihb,], "error_test", ordering=True)


#
#
# data_sets = ['bank']
#
# for key in ["time_train", "time_test", "time_hyper", "error_train", "error_test", 'avg_degree', "G + O", "total_sparsity"]:
#     print("-------------------------------------------------------------------------")
#     print("key: ", key)
#     if key in ['avg_degree', "G + O", "total_sparsity"]:
#         table_results(data_sets, hps_no_svm, key, ordering=False)
#     else:
#         table_results(data_sets, hps, key, ordering=False)
# print("############################################################################################")




# Gröbner:
# -----------------------------------------------------------------------------------------------------
# Testing
#                                0
# oracle_type                   CG
# term_ordering_strategy   pearson
# inverse_hessian_boost       full
# psi                       0.0005
# C                           10.0
# time_train              0.077631
# time_test               0.002241
# accuracy_train               1.0
# accuracy_test                1.0
# avg_sparsity                 0.0
# total_sparsity               0.0
# number_of_polynomials         22
# number_of_zeros                0
# number_of_entries            204
# number_of_terms               18
# degree                  2.545455
# time_hyper              1.582884
# -----------------------------------------------------------------------------------------------------



# Border
# -----------------------------------------------------------------------------------------------------
# Testing
#                                0
# oracle_type                   CG
# term_ordering_strategy   pearson
# inverse_hessian_boost       full
# psi                       0.0005
# C                           10.0
# time_train              0.103767
# time_test               0.002394
# accuracy_train               1.0
# accuracy_test                1.0
# avg_sparsity                 0.0
# total_sparsity               0.0
# number_of_polynomials         38
# number_of_zeros                0
# number_of_entries            368
# number_of_terms               18
# degree                  2.761905
# time_hyper              1.874777
# -----------------------------------------------------------------------------------------------------



























# Gröbner
# -----------------------------------------------------------------------------------------------------
# Testing
#                                0
# oracle_type                   CG
# term_ordering_strategy   pearson
# inverse_hessian_boost       full
# psi                      0.00005
# C                           10.0
# time_train              0.127333
# time_test                 0.0029
# accuracy_train               1.0
# accuracy_test                1.0
# avg_sparsity                 0.0
# total_sparsity               0.0
# number_of_polynomials         31
# number_of_zeros                0
# number_of_entries            464
# number_of_terms               30
# degree                       3.0
# time_hyper              1.996859
# -----------------------------------------------------------------------------------------------------

# Border
# -----------------------------------------------------------------------------------------------------
# Testing
#                                0
# oracle_type                   CG
# term_ordering_strategy   pearson
# inverse_hessian_boost       full
# psi                      0.00005
# C                           10.0
# time_train              0.211967
# time_test               0.004274
# accuracy_train               1.0
# accuracy_test                1.0
# avg_sparsity                 0.0
# total_sparsity               0.0
# number_of_polynomials         57
# number_of_zeros                0
# number_of_entries            908
# number_of_terms               32
# degree                  3.484848
# time_hyper              2.673084
# -----------------------------------------------------------------------------------------------------
