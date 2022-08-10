import numpy as np
from global_ import psis_single_, psis_, Cs_, degrees_, data_sets_performance_, data_sets_time_trials_, \
    Cs_single_, border_type_, tos_
from src.auxiliary_functions.auxiliary_functions import G_O_bound_vectorized
from src.experiment_setups.experiment_setups import perform_experiments
from src.experiment_setups.experiment_termination_theory_v_practice import termination_theory_v_practice
from src.experiment_setups.experiment_time_trials import time_trials
from src.plotting.plotting_functions import plotter


def term_ordering_comparsion():
    hp_pearson = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': 'pearson',
                  'border_type': border_type_, 'inverse_hessian_boost': 'full'}
    hp_pearson_rev = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_, 'border_type': border_type_,
                      'term_ordering_strategy': 'rev pearson', 'inverse_hessian_boost': 'full'}

    perform_experiments(data_sets_performance_, hp_pearson, saving=True)
    perform_experiments(data_sets_performance_, hp_pearson_rev, saving=True)


def performance():
    hp_cg_ihb = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                 'border_type': border_type_, 'inverse_hessian_boost': 'full'}
    hp_agd_ihb = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                  'border_type': border_type_, 'inverse_hessian_boost': 'full'}
    hp_bpcg_wihb = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                    'border_type': border_type_, 'inverse_hessian_boost': 'weak'}
    hp_abm = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
              'border_type': border_type_, 'inverse_hessian_boost': 'false'}
    hp_vca = {'algorithm': 'vca', 'psi': psis_, 'C': Cs_}
    hp_svm = {'algorithm': 'svm', 'C': Cs_, 'avg_degree': degrees_}

    hps = [hp_cg_ihb, hp_agd_ihb, hp_bpcg_wihb, hp_abm, hp_vca, hp_svm]
    for hp in hps:
        perform_experiments(data_sets_performance_, hp, saving=True)


def termination_graphic():
    ns = [1, 5, 10, 25, 50]
    legend = []
    data = []
    x = np.logspace(-5, -1, 9)
    for i in range(0, len(ns)):
        n = ns[i]
        data.append(G_O_bound_vectorized(x, n))
        legend.append(r"number of features $n=$" + r" ${}$".format(str(n)))

    plotter('termination', data, x, legend, x_label=r'Vanishing parameter $\psi$',
            y_label=r'Theoretical bound on $|\mathcal{G}| + |\mathcal{O}|$', x_scale='log', y_scale='log')


def time_BPCG_v_IHB_v_WIHB():
    hp_BPCG = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_single_, 'C': Cs_single_,
               'border_type': border_type_, 'border_type': border_type_, 'term_ordering_strategy': tos_,
               'inverse_hessian_boost': 'false'}
    hp_WIHB = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_single_, 'C': Cs_single_,
               'border_type': border_type_, 'term_ordering_strategy': tos_, 'inverse_hessian_boost': 'weak'}
    hp_IHB = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_single_, 'C': Cs_single_,
              'border_type': border_type_, 'term_ordering_strategy': tos_, 'inverse_hessian_boost': 'full'}

    hyperparameters = [hp_BPCG, hp_WIHB, hp_IHB]
    time_trials("time_ihb", hyperparameters, data_sets_time_trials_)


def time_comparison():
    hp_CGAVI_IHB = {'algorithm': 'oavi', 'oracle_type': 'CG', 'psi': psis_, 'C': Cs_,
                    'term_ordering_strategy': tos_, 'border_type': border_type_, 'inverse_hessian_boost': 'full'}
    hp_AGDAVI_IHB = {'algorithm': 'oavi', 'oracle_type': 'AGD', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
                     'border_type': border_type_, 'inverse_hessian_boost': 'full'}
    hp_abm = {'algorithm': 'oavi', 'oracle_type': 'ABM', 'psi': psis_, 'C': Cs_, 'term_ordering_strategy': tos_,
              'border_type': border_type_, 'inverse_hessian_boost': 'false'}
    hp_vca = {'algorithm': 'vca', 'psi': psis_, 'C': Cs_}

    hyperparameters = [hp_CGAVI_IHB, hp_AGDAVI_IHB, hp_abm, hp_vca]
    time_trials("time_algos", hyperparameters, data_sets_time_trials_)


def time_PCG_v_BPCG():
    hp_PCG = {'algorithm': 'oavi', 'oracle_type': 'PCG', 'psi': psis_single_, 'C': Cs_single_,
              'border_type': border_type_, 'term_ordering_strategy': tos_, 'inverse_hessian_boost': 'false'}
    hp_BPCG = {'algorithm': 'oavi', 'oracle_type': 'BPCG', 'psi': psis_single_, 'C': Cs_single_,
               'border_type': border_type_, 'term_ordering_strategy': tos_, 'inverse_hessian_boost': 'false'}

    hyperparameters = [hp_PCG, hp_BPCG]
    time_trials("time_pcg_v_bpcg", hyperparameters, data_sets_time_trials_)


if __name__ == '__main__':
    # term_ordering_comparsion()
    performance()
    termination_graphic()
    termination_theory_v_practice(10000, list(range(2, 25, 3)), psis_single_[0])
    time_BPCG_v_IHB_v_WIHB()
    time_comparison()
    time_PCG_v_BPCG()
