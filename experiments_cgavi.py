from global_ import psis_, Cs_, tos_, degrees_, data_sets_performance_, taus_, border_type_, Cs_single_, \
    data_sets_comparison_border_types_, data_sets_performance_gb_, data_sets_performance_bb_
from src.experiment_setups.experiment_border_types import comparison_border_types
from src.experiment_setups.experiment_setups import perform_experiments


def performance():
    hp_agd_gb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                 'border_type': "gb", 'inverse_hessian_boost': 'false'}
    hp_agd_bb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                 'border_type': "bb", 'inverse_hessian_boost': 'false'}

    hp_pcg_gb = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'psi': psis_, 'C': Cs_single_,
                 'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'false'}
    hp_pcg_bb = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'psi': psis_, 'C': Cs_single_,
                 'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'false'}

    hp_abm_gb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_,
                 'C': Cs_single_, 'term_ordering_strategy': tos_, 'border_type': "gb",
                 'inverse_hessian_boost': 'false'}
    hp_abm_bb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_single_,
                 'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'false'}

    hp_vca = {'algorithm': 'vca', 'psi': psis_, 'C': Cs_}
    hp_svm = {'algorithm': 'svm', 'C': Cs_, 'avg_degree': degrees_}

    hp_bpcg_gb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                  'term_ordering_strategy': tos_, 'border_type': "gb", 'inverse_hessian_boost': 'false'}
    hp_bpcg_bb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_single_,
                  'term_ordering_strategy': tos_, 'border_type': "bb", 'inverse_hessian_boost': 'false'}

    hp_avi_gb = {'algorithm': 'avi', 'C': Cs_single_, 'psi': psis_, 'tau': taus_, 'term_ordering_strategy': tos_,
                 'border_type': "gb"}
    hp_avi_bb = {'algorithm': 'avi', 'C': Cs_single_, 'psi': psis_, 'tau': taus_, 'term_ordering_strategy': tos_,
                 'border_type': "bb"}

    hps_gb = [hp_pcg_gb, hp_agd_gb, hp_abm_gb,  hp_vca, hp_svm, hp_bpcg_gb,  hp_avi_gb]
    hps_bb = [hp_pcg_bb, hp_agd_bb, hp_abm_bb, hp_bpcg_bb, hp_avi_bb]
    for hp in hps_gb:
        perform_experiments(data_sets_performance_gb_, hp, saving=True)
    for hp in hps_bb:
        perform_experiments(data_sets_performance_bb_, hp, saving=True)


def border_type():
    hp_pcg_gb = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'C': Cs_single_, 'term_ordering_strategy': tos_,
                 'border_type': "gb", 'inverse_hessian_boost': 'weak'}
    hp_pcg_bb = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'C': Cs_single_, 'term_ordering_strategy': tos_,
                 'border_type': "bb", 'inverse_hessian_boost': 'weak'}
    hp_abm_gb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'C': Cs_single_, 'term_ordering_strategy': tos_,
                 'border_type': "gb", 'inverse_hessian_boost': 'false'}
    hp_abm_bb = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'C': Cs_single_, 'term_ordering_strategy': tos_,
                 'border_type': "bb", 'inverse_hessian_boost': 'false'}

    comparison_border_types("comparison_border_types", [hp_pcg_gb, hp_pcg_bb, hp_abm_gb, hp_abm_bb],
                            data_sets_comparison_border_types_)


if __name__ == '__main__':
    performance()
    border_type()
