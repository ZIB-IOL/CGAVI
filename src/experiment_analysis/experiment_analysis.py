from src.auxiliary_functions.auxiliary_functions import translate_names
from src.data_frames.data_frames import load_data_frame
import pandas as pd
import numpy as np


def get_from_key(algorithm_name, data_set_name, key: str):
    """Returns the mean and standard deviation associated with a certain key for algorithm_name on data_set_name.

    The key is in ['time_train', 'time_test','error_train', 'error_test', 'avg_degree', 'total_sparsity',
    'number_of_polynomials_and_terms', 'time_hyper'].
    """
    from global_ import n_runs_

    list_of_values = []
    for i in range(1, n_runs_ + 1):
        df_tmp = load_data_frame(algorithm_name, data_set_name, experiment="performance", test=True, number=i)
        if key == "G + O":
            value = int(df_tmp["number_of_polynomials"]) + int(df_tmp["number_of_terms"])
        elif key == "error_train":
            value = (1 - df_tmp["accuracy_train"]) * 100
        elif key == "error_test":
            value = (1 - df_tmp["accuracy_test"]) * 100
        else:
            value = float(df_tmp[key])
        list_of_values.append(value)

    array_of_values = np.array(list_of_values)
    mean = float(np.mean(array_of_values))
    std = float(np.std(array_of_values))
    if key in ['time_train', 'time_test', 'time_hyper']:
        mean = np.format_float_scientific(mean, 1)
        std = np.format_float_scientific(std, 1)
    elif key in ['error_train', 'error_test', 'total_sparsity', 'avg_degree', 'number_of_polynomials_and_terms', 'G + O']:
        mean = "{:.2f}".format(mean)
        std = "{:.2f}".format(std)
    return mean, std


def table_results(data_sets, algorithms, key, ordering=False):
    """Creates a latex table of the form
        Algorithms      data_sets[1]    ...       data_sets[-1]
        algorithms[1]       *           ...             *
        ...
        algorithms[-1]      *           ...             *
    and the entries correspond to the key.
    The key is in ["error train", "error test", "time train", "time test", "time hyper", 'avg_degree', "total sparsity",
    "G + O"].
    """
    algorithm_names = [translate_names(name, ordering=True) for name in algorithms]
    algorithm_names_tex = [translate_names(name, ordering=ordering, tex=True) for name in algorithms]
    columns = ['', 'Algorithms'] + data_sets
    df = pd.DataFrame(columns=columns)
    df = df.reset_index(drop=True)
    for i in range(len(algorithm_names)):
        df_tmp = ['', algorithm_names_tex[i]] + [get_from_key(algorithm_names[i], name, key)[0] for name in data_sets]
        df_tmp = pd.DataFrame([df_tmp], columns=columns)
        df = df.reset_index(drop=True)
        df = df.append(df_tmp, ignore_index=True)
    df = df.style.hide_index()
    print(df.to_latex())
