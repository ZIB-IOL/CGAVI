import numpy as np
import cupy as cp
from global_ import n_runs_
from src.auxiliary_functions.auxiliary_functions import G_O_bound
from src.feature_transformations.oracle_avi import OracleAVI
from src.plotting.plotting_functions import plotter
from global_ import n_seed_


def termination_theory_v_practice(m, ns, psi):
    """Checks whether the theoretical bound on |G| + |O| is close to what we observe in practice."""
    cp.random.seed(n_seed_)
    np.random.seed(n_seed_)
    O_G_practice = None
    for run in range(n_runs_):
        O_G_practice_one_run = []
        for n in ns:
            X = np.random.rand(m, n)
            X = cp.asarray(X)
            oracle_avi = OracleAVI(psi=psi,
                                   oracle_type='CG',
                                   term_ordering_strategy='pearson',
                                   inverse_hessian_boost='full')
            oracle_avi.fit(X)
            _, _, _, number_of_polynomials, number_of_terms, _ = oracle_avi.evaluate_transformation()
            practice = int(number_of_polynomials + number_of_terms)

            O_G_practice_one_run.append(practice)
        O_G_practice_one_run = np.array([O_G_practice_one_run])

        if not isinstance(O_G_practice, np.ndarray):
            O_G_practice = O_G_practice_one_run
        else:
            O_G_practice = np.vstack((O_G_practice, O_G_practice_one_run))

    O_G_practice_mean = np.mean(O_G_practice, axis=0)
    O_G_practice_std = np.std(O_G_practice, axis=0)
    theory = np.array([[min(m * (n + 1), G_O_bound(psi, n)) for n in ns]])
    n_quadrupled = np.array([[n ** 4 for n in ns]])
    means = [O_G_practice_mean.flatten(), theory.flatten(), n_quadrupled.flatten()]
    stds = [O_G_practice_std.flatten(), np.zeros(len(theory)), np.zeros(len(n_quadrupled))]

    algorithm_names = [r'$\texttt{CGAVI}$', "theoretical bound", r'$n^4$']
    plotter('termination_theory_v_practice_m_{m}_psi_{psi}'.format(m=m, psi=psi), means, ns,
            algorithm_names, std_data=stds, data_set_name='random', x_label=r'Number of features $n$',
            y_label=r'$|\mathcal{G}| + |\mathcal{O}|$', y_scale='log')
