import cupy as cp
import numpy as np
from global_ import n_splits_, n_runs_, n_seed_
from src.auxiliary_functions.auxiliary_functions import translate_names, fd
from src.experiment_setups.experiment_setups import fetch_and_prepare_data, ExperimentSetups
from src.plotting.plotting_functions import plotter


def time_trials(title: str, hyperparameters: list, data_set_names, ordering: bool = False, border_type: bool = False):
    """Performs the time trials experiments, that is, compares running times of algorithms."""
    cp.random.seed(n_seed_)
    np.random.seed(n_seed_)

    # collect the names of all algorithms
    algorithm_names = []
    for hp in hyperparameters:
        name = translate_names(hp, tex=True, ordering=ordering, border_type=border_type)
        algorithm_names.append(name)

    for idx in range(len(data_set_names)):
        data_set_name = data_set_names[idx]
        mean_train, std_train, mean_test, std_test = [], [], [], []
        X_train, _, X_test, _ = fetch_and_prepare_data(data_set_name)
        X = cp.vstack((X_train, X_test))
        m = X.shape[0]
        # clean-up
        del X
        # prepare the sample points for which we want to compare algorithm performance
        n_samples = list(np.linspace(int(m / n_splits_), m, n_splits_, dtype=int))

        for hp in hyperparameters:
            one_algorithm_train, one_algorithm_test = None, None
            for run in range(n_runs_):
                X_train, y_train, X_test, y_test = fetch_and_prepare_data(data_set_name, proportion=0.25)
                X, y = np.vstack((X_train, X_test)), np.vstack((fd(y_train), fd(y_test)))
                # set up the experiment environment
                pc = ExperimentSetups(data_set_name, hp)
                pc.X_train, pc.y_train = fd(X[:10000, :]), fd(y[:10000])

                # tune hyperparameters on small subset of data
                df, _ = pc.hyperparamter_tuning()

                # clean-up
                del X_train, y_train, X_test, y_test

                # perform one run and store the results in one_run_train and one_run_train_test
                one_run_train, one_run_test = [], []
                for sample in n_samples:
                    pc.X_train, pc.y_train = X[:sample], y[:sample]
                    pc.X_test, pc.y_test = X[:sample], y[:sample]
                    try:
                        out = pc.final_training(df, count_svm=False)
                        one_run_train.append(float(out['time_train']))
                        one_run_test.append(float(out['time_test']))
                    except:
                        one_run_train.append(np.nan)
                        one_run_test.append(np.nan)

                del pc
                # add the run results to the one_algorithm_train and one_algorithm_test arrrays
                if not isinstance(one_algorithm_train, np.ndarray):
                    one_algorithm_train = np.array([one_run_train])
                else:
                    one_algorithm_train = np.vstack((one_algorithm_train, np.array([one_run_train])))
                if not isinstance(one_algorithm_test, np.ndarray):
                    one_algorithm_test = np.array([one_run_test])
                else:
                    one_algorithm_test = np.vstack((one_algorithm_test, np.array([one_run_test])))

            for i in range(0, one_algorithm_train.shape[1]):
                if np.isnan(one_algorithm_train[:, i]).any():
                    one_algorithm_train = one_algorithm_train[:, :i]
                    one_algorithm_test = one_algorithm_test[:, :i]
                    break

            one_algorithm_std_train = np.std(one_algorithm_train, axis=0)
            one_algorithm_mean_train = np.mean(one_algorithm_train, axis=0)

            one_algorithm_std_test = np.std(one_algorithm_test, axis=0)
            one_algorithm_mean_test = np.mean(one_algorithm_test, axis=0)

            mean_train.append(one_algorithm_mean_train.flatten())
            std_train.append(one_algorithm_std_train.flatten())
            mean_test.append(one_algorithm_mean_test.flatten())
            std_test.append(one_algorithm_std_test.flatten())
        if idx != len(data_set_names) - 1:
            pass_algorithm_names = None
        else:
            pass_algorithm_names = algorithm_names
        plotter(title, mean_train, n_samples, pass_algorithm_names, data_set_name, std_data=std_train)
