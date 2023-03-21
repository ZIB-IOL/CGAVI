import cupy as cp
import numpy as np
from global_ import n_runs_, n_seed_
from src.auxiliary_functions.auxiliary_functions import translate_names
from src.experiment_setups.experiment_setups import fetch_and_prepare_data, ExperimentSetups
from src.plotting.plotting_functions import plotter


def comparison_border_types(title: str, hyperparameters: list, data_set_names, ordering: bool = False):
    """Compares different types of borders."""
    cp.random.seed(n_seed_)
    np.random.seed(n_seed_)

    psis = np.logspace(-0, -3.5, 10).tolist()
    psis.reverse()

    # collect the names of all algorithms
    algorithm_names = []
    for hp in hyperparameters:
        name = translate_names(hp, tex=True, ordering=ordering, border_type=True)
        algorithm_names.append(name)

    for idx in range(len(data_set_names)):
        data_set_name = data_set_names[idx]
        mean_polynomials, std_polynomials, mean_terms, std_terms = [], [], [], []

        for hp in hyperparameters:
            one_algorithm_polynomials, one_algorithm_terms = None, None
            for run in range(n_runs_):
                X_train, y_train, X_test, y_test = fetch_and_prepare_data(data_set_name, proportion=0.6)
                one_run_n_polynomials, one_run_n_terms = [], []
                for psi in psis:
                    hp["psi"] = [psi]
                    # set up the experiment environment
                    pc = ExperimentSetups(data_set_name, hp)
                    pc.X_train, pc.y_train = X_train, y_train
                    pc.X_test, pc.y_test = X_test, y_test

                    # tune hyperparameters on small subset of data
                    df, _ = pc.hyperparamter_tuning()

                    # perform one run and store the results in one_run_n_polynomials and one_run_n_polynomials_test

                    try:
                        out = pc.final_training(df, count_svm=False)
                        one_run_n_polynomials.append(float(out['number_of_polynomials']))
                        one_run_n_terms.append(float(out['number_of_terms']))
                    except:
                        one_run_n_polynomials.append(np.nan)
                        one_run_n_terms.append(np.nan)

                # add the run results to the one_algorithm_polynomials and one_algorithm_terms arrrays
                if not isinstance(one_algorithm_polynomials, np.ndarray):
                    one_algorithm_polynomials = np.array([one_run_n_polynomials])
                else:
                    one_algorithm_polynomials = np.vstack(
                        (one_algorithm_polynomials, np.array([one_run_n_polynomials])))
                if not isinstance(one_algorithm_terms, np.ndarray):
                    one_algorithm_terms = np.array([one_run_n_terms])
                else:
                    one_algorithm_terms = np.vstack((one_algorithm_terms, np.array([one_run_n_terms])))

            for i in range(0, one_algorithm_polynomials.shape[1]):
                if np.isnan(one_algorithm_polynomials[:, i]).any():
                    one_algorithm_polynomials = one_algorithm_polynomials[:, :i]
                    one_algorithm_terms = one_algorithm_terms[:, :i]
                    break

            one_algorithm_std_polynomials = np.std(one_algorithm_polynomials, axis=0)
            one_algorithm_mean_polynomials = np.mean(one_algorithm_polynomials, axis=0)

            one_algorithm_std_terms = np.std(one_algorithm_terms, axis=0)
            one_algorithm_mean_terms = np.mean(one_algorithm_terms, axis=0)

            mean_polynomials.append(one_algorithm_mean_polynomials.flatten())
            std_polynomials.append(one_algorithm_std_polynomials.flatten())
            mean_terms.append(one_algorithm_mean_terms.flatten())
            std_terms.append(one_algorithm_std_terms.flatten())
        if idx != len(data_set_names) - 1:
            pass_algorithm_names = None
        else:
            pass_algorithm_names = algorithm_names
        plotter(str(title + "_G"), mean_polynomials, psis, pass_algorithm_names, data_set_name,
                std_data=None,
                x_label=r"Vanishing parameter $\psi$", y_label=r"$|\mathcal{G}|$", x_scale="log", two_versions=True)
        plotter(str(title + "_O"), mean_terms, psis, pass_algorithm_names, data_set_name, std_data=None,
                x_label=r"Vanishing parameter $\psi$", y_label=r"$|\mathcal{O}|$", x_scale="log", two_versions=True)
