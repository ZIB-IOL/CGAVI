from global_ import psis_, Cs_, tos_, degrees_, data_sets_performance_, taus_, border_type_, Cs_single_, \
    data_sets_performance_gb_, data_sets_performance_bb_
from src.experiment_setups.experiment_border_types import comparison_border_types
from src.experiment_setups.experiment_setups import perform_experiments


def performance():
    hp_bpcg_wihb_gb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                       'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'weak'}
    hp_bpcg_wihb_bb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                       'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'weak'}
    hp_cgavi_ihb_gb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_single_,
                       'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'true'}
    hp_cgavi_ihb_bb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_single_,
                       'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'true'}
    hp_agd_ihb_gb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                     'border_type': "gb", 'inverse_hessian_boost': 'true'}
    hp_agd_ihb_bb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                     'border_type': "bb", 'inverse_hessian_boost': 'true'}
    hp_abm_gb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_single_,
                 'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'false'}
    hp_abm_bb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_single_,
                 'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'false'}
    hp_vca = {'algorithm': 'vca', 'psi': psis_, 'C': Cs_}
    hp_svm = {'algorithm': 'svm', 'C': Cs_, 'avg_degree': degrees_}

    hps_gb = [hp_bpcg_wihb_gb, hp_cgavi_ihb_gb, hp_agd_ihb_gb, hp_abm_gb, hp_vca, hp_svm]
    hps_bb = [hp_bpcg_wihb_bb, hp_cgavi_ihb_bb, hp_agd_ihb_bb, hp_abm_bb]
    for hp in hps_gb:
        perform_experiments(data_sets_performance_gb_, hp, saving=True)
    for hp in hps_bb:
        perform_experiments(data_sets_performance_bb_, hp, saving=True)


def border_type():
    hp_bpcg_wihb_gb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                       'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'weak'}
    hp_bpcg_wihb_bb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                       'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'weak'}
    hp_abm_gb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'C': Cs_single_, 'term_ordering_strategy': tos_,
                 'border_type': "gb", 'inverse_hessian_boost': 'false'}
    hp_abm_bb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'C': Cs_single_, 'term_ordering_strategy': tos_,
                 'border_type': "bb", 'inverse_hessian_boost': 'false'}

    comparison_border_types("comparison_border_types", [hp_bpcg_wihb_gb, hp_bpcg_wihb_bb, hp_abm_gb, hp_abm_bb],
                            data_sets_performance_bb_)

if __name__ == '__main__':
    performance()
    border_type()
