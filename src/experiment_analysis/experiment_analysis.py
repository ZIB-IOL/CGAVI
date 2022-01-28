from global_ import number_of_runs_
from src.data_frames.data_frames import load_data_frame
import pandas as pd
import numpy as np


def print_results(latex: bool = False):
    """Prepare results as (LaTex) table."""
    columns_other = ['data set',
                     'algorithm',
                     'error train',
                     'error train std',
                     'error test',
                     'error test std',
                     'time train',
                     'time train std',
                     'time test',
                     'time test std',
                     'hyperparameter optimization time',
                     'hyperparameter optimization time std',
                     'polynomials',
                     'polynomials std',
                     'average sparsity',
                     'average sparsity std',
                     'total sparsity',
                     'total sparsity std',
                     'zeros',
                     'zeros std',
                     'non zero entries',
                     'non zero entries std',
                     'entries',
                     'entries std',
                     'terms',
                     'terms std',
                     '|O| + |G|',
                     '|O| + |G| std',
                     'max degree',
                     'max degree std']

    columns_svm = ['data set',
                   'algorithm',
                   'error train',
                   'error train std',
                   'error test',
                   'error test std',
                   'time train',
                   'time train std',
                   'time test',
                   'time test std',
                   'hyperparameter optimization time',
                   'hyperparameter optimization time std',
                   'max degree',
                   'max degree std']

    data_set_names = ['banknote', 'cancer', 'htru2', 'iris', 'seeds', 'sonar', 'spam', 'voice', 'wine']
    algorithm_names = ['l1_cgavi', 'l2_cgavi', 'agdavi', 'avi', 'vca', 'svm']

    for name in columns_other:

        if ' std' not in name and name not in ['data set', 'algorithm']:
            print("------------------------------------------------------------------------------------------------")
            print(name)
            print("------------------------------------------------------------------------------------------------")
            cols = ['placeholder', 'algorithm'] + data_set_names
            df = pd.DataFrame(columns=cols)

            for algorithm_name in algorithm_names:
                results = ['placeholder', algorithm_name]
                if algorithm_name is not 'svm':
                    columns = columns_other
                else:
                    columns = columns_svm
                for data_set_name in data_set_names:
                    try:
                        float_val = float(prep(data_set_name, algorithm_name, columns)[name])
                        if name in ['error train', 'error test', 'average sparsity',
                                    'total sparsity', 'max degree']:
                            if name in 'time test':
                                float_val *= 1000
                            rounded_val = "{:.2f}".format(float_val)
                        elif name in ['polynomials', 'zeros', 'terms', '|O| + |G|', ]:
                            rounded_val = int(prep(data_set_name, algorithm_name, columns)[name])
                        elif name in ['non zero entries', 'entries', 'time train', 'time test',
                                      'hyperparameter optimization time']:
                            rounded_val = np.format_float_scientific(prep(data_set_name, algorithm_name, columns)[name],
                                                                     1)
                        else:
                            print(name)
                    except:
                        rounded_val = 'N/A'
                    results = results + [rounded_val]
                df = df.append(pd.DataFrame([results], columns=cols))
            if latex:
                df = df.style.hide_index()
                print(df.to_latex())
            else:
                print(df.T)
            print("------------------------------------------------------------------------------------------------")
            print(' ')


def svm_eval(latex: bool = True):
    """Prepare rebuttal results as LaTex table."""

    data_set_names = ['banknote', 'cancer', 'htru2', 'iris', 'seeds', 'sonar', 'spam', 'voice', 'wine']
    algorithm_names = ['svm']


def prep(data_set_name, algorithm_name, columns):
    """Prep data."""
    df = load_data_frame(algorithm_name, data_set_name, test=True, number=1)
    df['Algorithm'] = algorithm_name
    df['Data set'] = data_set_name
    df = df.reset_index(drop=True)
    for i in range(2, number_of_runs_ + 1):
        df_tmp = load_data_frame(algorithm_name, data_set_name, test=True, number=i)
        df_tmp['Algorithm'] = algorithm_name
        df_tmp['Data set'] = data_set_name
        df_tmp = df_tmp.reset_index(drop=True)
        df = df.append(df_tmp, ignore_index=True)

    classification_error_train_mean = ((1 - df['accuracy_train']) * 100).mean()
    classification_error_train_std = ((1 - df['accuracy_train']) * 100).std()

    classification_error_test_mean = ((1 - df['accuracy_test']) * 100).mean()
    classification_error_test_std = ((1 - df['accuracy_test']) * 100).std()

    time_train_mean = df['time_train'].mean()
    time_train_std = df['time_train'].std()

    time_test_mean = df['time_test'].mean()
    time_test_std = df['time_test'].std()

    hyperparameter_optimization_time_mean = df['hyperparameter_optimization_time'].mean()
    hyperparameter_optimization_time_std = df['hyperparameter_optimization_time'].std()

    degree_mean = df['degree'].mean()
    degree_std = df['degree'].std()

    if algorithm_name is not 'svm':

        number_of_polynomials_mean = df['number_of_polynomials'].mean()
        number_of_polynomials_std = df['number_of_polynomials'].std()

        avg_sparsity_mean = df['avg_sparsity'].mean()
        avg_sparsity_std = df['avg_sparsity'].std()

        total_sparsity_mean = df['total_sparsity'].mean()
        total_sparsity_std = df['total_sparsity'].std()

        number_of_zeros_mean = df['number_of_zeros'].mean()
        number_of_zeros_std = df['number_of_zeros'].std()

        number_of_non_zeros_mean = (df['number_of_entries'] - df['number_of_zeros']).mean()
        number_of_non_zeros_std = (df['number_of_entries'] - df['number_of_zeros']).std()

        number_of_entries_mean = df['number_of_entries'].mean()
        number_of_entries_std = df['number_of_entries'].std()

        number_of_terms_mean = df['number_of_terms'].mean()
        number_of_terms_std = df['number_of_terms'].std()

        O_plus_G_mean = (df['number_of_polynomials'] + df['number_of_terms']).mean()
        O_plus_G_std = (df['number_of_polynomials'] + df['number_of_terms']).std()

        results = pd.DataFrame([[data_set_name, algorithm_name, classification_error_train_mean,
                                 classification_error_train_std, classification_error_test_mean,
                                 classification_error_test_std, time_train_mean, time_train_std, time_test_mean,
                                 time_test_std, hyperparameter_optimization_time_mean,
                                 hyperparameter_optimization_time_std, number_of_polynomials_mean,
                                 number_of_polynomials_std, avg_sparsity_mean, avg_sparsity_std,
                                 total_sparsity_mean, total_sparsity_std, number_of_zeros_mean,
                                 number_of_zeros_std, number_of_non_zeros_mean, number_of_non_zeros_std,
                                 number_of_entries_mean, number_of_entries_std, number_of_terms_mean,
                                 number_of_terms_std, O_plus_G_mean, O_plus_G_std,
                                 degree_mean, degree_std]],
                               columns=columns)
    else:
        results = pd.DataFrame([[data_set_name, algorithm_name, classification_error_train_mean,
                                 classification_error_train_std, classification_error_test_mean,
                                 classification_error_test_std, time_train_mean, time_train_std, time_test_mean,
                                 time_test_std, hyperparameter_optimization_time_mean,
                                 hyperparameter_optimization_time_std, degree_mean, degree_std
                                 ]], columns=columns)
    return results


def round_significant_digits(number: float, no_digits: int = 2):
    """Rounds to no_digits significant digits."""
    rounded = float(np.format_float_positional(number, precision=no_digits, unique=False, fractional=False, trim='k'))
    return rounded
