from global_ import psis_, Cs_, tos_, degrees_, Cs_single_, data_sets_performance_gb_, data_sets_performance_bb_
from src.experiment_setups.experiment_border_types import comparison_border_types
from src.experiment_setups.experiment_setups import perform_experiments

# Algorithms
hp_bpcg_wihb_gb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_,
                   'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'weak'}
hp_bpcg_wihb_bb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_,
                   'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'weak'}
hp_cgavi_ihb_gb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_,
                   'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'full'}
hp_cgavi_ihb_bb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_,
                   'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'full'}
hp_agd_ihb_gb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                 'border_type': "gb", 'inverse_hessian_boost': 'full'}
hp_agd_ihb_bb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                 'border_type': "bb", 'inverse_hessian_boost': 'full'}
hp_abm_gb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_,
             'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'false'}
hp_abm_bb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_,
             'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'false'}
hp_vca = {'algorithm': 'vca', 'psi': psis_, 'C': Cs_}
hp_svm = {'algorithm': 'svm', 'C': Cs_, 'avg_degree': degrees_}

# Performance experiment
hps_gb = [hp_bpcg_wihb_gb, hp_cgavi_ihb_gb, hp_agd_ihb_gb, hp_abm_gb, hp_vca, hp_svm]

# for hp in hps_gb:
#     perform_experiments(data_sets_performance_gb_, hp, saving=True)

# Border experiment
hps_bb = [hp_bpcg_wihb_bb, hp_cgavi_ihb_bb, hp_agd_ihb_bb, hp_abm_bb]
# for hp in hps_bb:
#     perform_experiments(data_sets_performance_bb_, hp, saving=True)

# Border plots
hp_bpcg_wihb_gb_plot = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                        'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'weak'}
hp_bpcg_wihb_bb_plot = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                        'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'weak'}
hp_cgavi_ihb_gb_plot = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_single_,
                        'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'full'}
hp_cgavi_ihb_bb_plot = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_single_,
                        'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'full'}
hp_agd_ihb_gb_plot = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_single_,
                      'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'full'}
hp_agd_ihb_bb_plot = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_single_,
                      'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'full'}
hp_abm_gb_plot = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_single_,
                  'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'false'}
hp_abm_bb_plot = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_single_,
                  'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'false'}

comparison_border_types("comparison_border_types",
                        [hp_cgavi_ihb_gb_plot, hp_cgavi_ihb_bb_plot, hp_bpcg_wihb_gb_plot, hp_bpcg_wihb_bb_plot,
                         hp_abm_gb_plot, hp_abm_bb_plot], data_sets_performance_gb_)

