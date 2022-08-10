import os
import sys
import time
import warnings
import pandas as pd
from sklearn import svm
import cupy as cp
import numpy as np
from global_ import n_seed_, n_runs_, cvs_
from src.auxiliary_functions.auxiliary_functions import fd, translate_names
from src.data_frames.data_frames import save_data_frame
from src.data_sets.data_set_creation import fetch_data_set
from src.data_sets.preprocessing import train_test_split, min_max_feature_scaling, split_into_classes, \
    unison_shuffled_copies
from src.feature_transformations.avi import AVI
from src.feature_transformations.oracle_avi import OracleAVI
from src.feature_transformations.vca import VCA


class ExperimentSetups:
    """Class managing the entire experimental setup for performance comparison.

    Args:
        name_data_set: str
        hyperparameters: dict
        cv: int, Optional
            Number of cross validation splits. (Default is cvs_.)

    Methods:
        fetch_and_prepare_data()
            Downloads and prepares the data.
        get_chunks(X_train_chunks: np.ndarray, y_train_chunks: np.ndarray, idx: int)
            Turns the idx chunk into the validation set and the other chunks into the train set.
        hyperparameter_tuning()
            Tunes the hyperparameters.
        final_training(df: pd.DataFrame, count_svm: bool)
            Performs the final retraining with the best hyperparameters.
        cross_validation(saving: bool)
            Performs cross validation.
        hyperparameter_combinations_avi()
            Tests all hyperparameter combinations for AVI and linear kernel SVM and returns a data frame containing the
            results.
        hyperparameter_combinations_oavi()
            Tests all hyperparameter combinations for OAVI and linear kernel SVM and returns a data frame
            containing the results.
        hyperparameter_combinations_svm()
            Tests all hyperparameter combinations for polynomial kernel SVM and returns a data frame containing the
            results.
        hyperparameter_combinations_vca()
            Tests all hyperparameter combinations for VCA and linear kernel SVM and returns a data frame containing the
            results.
        final_training_avi(data_frame: pd.DataFrame, count_svm: bool)
            Performs the final retraining for AVI and linear kernel SVM, with the optimal hyperparameter choices.
        final_training_oavi(data_frame: pd.DataFrame,count_svm: bool)
            Performs the final retraining for OAVI and linear kernel SVM, with the optimal hyperparameter choices.
        final_training_svm(data_frame: pd.DataFrame)
            Performs the final retraining for VCA and linear kernel SVM, with the optimal hyperparameter choices.
        final_training_vca(data_frame: pd.DataFrame,count_svm: bool)
            Performs the final retraining for VCA and linear kernel SVM, with the optimal hyperparameter choices.
    """
    def __init__(self, name_data_set: str, hyperparameters: dict, cv: int = cvs_):
        self.data_set_name = name_data_set
        self.algorithm_name = hyperparameters["algorithm"]
        self.X_train, self.y_train, self.X_test, self.y_test = self.fetch_and_prepare_data()

        self.hyperparameters = hyperparameters
        self.cv = cv
        self.experiment = "performance"

    def fetch_and_prepare_data(self):
        """Downloads and prepares the data."""
        return fetch_and_prepare_data(self.data_set_name)

    def get_chunks(self, X_train_chunks: np.ndarray, y_train_chunks: np.ndarray, idx: int):
        """Turns the idx chunk into the validation set and the other chunks into the train set.

        Args:
            X_train_chunks: np.ndarray
            y_train_chunks: np.ndarray
            idx: int

        Returns:
            X_t: np.ndarray
                Train set.
            y_t: np.ndarray
                Train labels.
            X_v: np.ndarray
                Validation set.
            y_v: np.ndarray
                Validation labels.
        """
        if idx == 0:
            X_t = np.vstack((X_train_chunks[1:]))
            y_t = np.vstack((y_train_chunks[1:]))
        elif idx == len(X_train_chunks) - 1:
            X_t = np.vstack((X_train_chunks[:len(X_train_chunks) - 1]))
            y_t = np.vstack((y_train_chunks[:len(X_train_chunks) - 1]))
        else:
            X_t = np.vstack((X_train_chunks[:idx]))
            y_t = np.vstack((y_train_chunks[:idx]))
        X_v = X_train_chunks[idx]
        y_v = y_train_chunks[idx]
        return fd(X_t), fd(y_t), fd(X_v), fd(y_v)

    def hyperparamter_tuning(self):
        """Tunes the hyperparameters."""
        df = None
        timer = time.time()
        if self.algorithm_name == "avi":
            df = self.hyperparameter_combinations_avi()
        elif self.algorithm_name == "oavi":
            df = self.hyperparameter_combinations_oavi()
        elif self.algorithm_name == "svm":
            df = self.hyperparameter_combinations_svm()
        elif self.algorithm_name == "vca":
            df = self.hyperparameter_combinations_vca()
        timer = time.time() - timer

        return df, timer

    def final_training(self, df: pd.DataFrame, count_svm: bool = True):
        """Performs the final retraining with the best hyperparameters."""
        df_test = None
        if self.algorithm_name == "avi":
            df_test = self.final_training_avi(df, count_svm)
        elif self.algorithm_name == "oavi":
            df_test = self.final_training_oavi(df, count_svm)
        elif self.algorithm_name == "svm":
            df_test = self.final_training_svm(df)
        elif self.algorithm_name == "vca":
            df_test = self.final_training_vca(df, count_svm)
        return df_test

    def cross_validation(self, saving: bool = True):
        """Performs cross validation."""
        # train
        df, timer = self.hyperparamter_tuning()
        assert df is not None
        if saving:
            save_data_frame(df, translate_names(self.hyperparameters), self.data_set_name, self.experiment,
                            test=False, number=-1)

        # test
        df_test = self.final_training(df)
        assert df_test is not None, "Why is this None?"
        df_test['time_hyper'] = timer
        if self.data_set_name == 'synthetic':
            df_test['samples_train'] = self.X_train.shape[0]
            df_test['samples_test'] = self.X_test.shape[0]
        if saving:

            save_data_frame(df_test, translate_names(self.hyperparameters), self.data_set_name, self.experiment,
                            test=True, number=-1)

        print("-----------------------------------------------------------------------------------------------------")
        print("Training")
        print(df.T)
        print("-----------------------------------------------------------------------------------------------------")
        print("Testing")
        print(df_test.T)
        print("-----------------------------------------------------------------------------------------------------")

        return df_test

    def hyperparameter_combinations_avi(self):
        """Tests all hyperparameter combinations for AVI and linear kernel SVM and returns a data frame containing the
        results.

        Returns:
            df: pd.DataFrame
        """
        term_ordering_strategy = self.hyperparameters["term_ordering_strategy"]
        border_type = self.hyperparameters["border_type"]
        psis = self.hyperparameters["psi"]
        taus = self.hyperparameters["tau"]
        Cs = self.hyperparameters["C"]

        columns = ['term_ordering_strategy', 'border_type', 'psi', 'tau', 'C', 'time_t', 'time_v', 'accuracy_t',
                   'accuracy_v', 'avg_sparsity', 'total_sparsity', 'number_of_polynomials', 'number_of_zeros',
                   'number_of_entries', 'number_of_terms', 'avg_degree']
        df = pd.DataFrame(columns=columns, dtype=float)
        for psi in psis:
            for tau in taus:
                for C in Cs:
                    (time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg, tot_sparsity_avg, n_poly_avg,
                     n_zeros_avg, n_entries_avg, n_terms_avg, degree_avg) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    X_train_chunks = np.array_split(self.X_train, self.cv)
                    y_train_chunks = np.array_split(self.y_train, self.cv)
                    for idx in range(0, len(X_train_chunks)):
                        X_t, y_t, X_v, y_v = self.get_chunks(X_train_chunks, y_train_chunks, idx=idx)
                        avi = AVI(psi=psi, tau=tau, term_ordering_strategy=term_ordering_strategy,
                                  border_type=border_type)
                        (time_t, time_v, acc_t, acc_v, sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
                         degree
                         ) = feature_transformation_for_linear_svm(avi, C, fd(X_t), fd(y_t), fd(X_v), fd(y_v))
                        time_t_avg += time_t / self.cv
                        time_v_avg += time_v / self.cv
                        acc_t_avg += acc_t / self.cv
                        acc_v_avg += acc_v / self.cv
                        sparsity_avg += sparsity / self.cv
                        tot_sparsity_avg += tot_sparsity / self.cv
                        n_poly_avg += n_poly / self.cv
                        n_zeros_avg += n_zeros / self.cv
                        n_entries_avg += n_entries / self.cv
                        n_terms_avg += n_terms / self.cv
                        degree_avg += degree / self.cv

                    row = pd.DataFrame([[term_ordering_strategy, border_type, psi, tau, C, time_t_avg, time_v_avg,
                                         acc_t_avg, acc_v_avg, sparsity_avg, tot_sparsity_avg, n_poly_avg, n_zeros_avg,
                                         n_entries_avg, n_terms_avg, degree_avg]], columns=columns)

                    df = df.append(row, ignore_index=True)
        return df

    def hyperparameter_combinations_oavi(self):
        """Tests all hyperparameter combinations for OAVI and linear kernel SVM and returns a data frame
        containing the results.

        Returns:
            df: pd.DataFrame
        """
        oracle_type = self.hyperparameters["oracle_type"]
        term_ordering_strategy = self.hyperparameters["term_ordering_strategy"]
        border_type = self.hyperparameters["border_type"]
        inverse_hessian_boost = self.hyperparameters["inverse_hessian_boost"]
        psis = self.hyperparameters["psi"]
        Cs = self.hyperparameters["C"]

        columns = ['oracle_type', 'term_ordering_strategy', 'border_type', 'inverse_hessian_boost', 'psi', 'C',
                   'time_t', 'time_v', 'accuracy_t', 'accuracy_v', 'avg_sparsity', 'total_sparsity',
                   'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms', 'avg_degree']
        df = pd.DataFrame(columns=columns, dtype=float)
        for psi in psis:
            for C in Cs:
                (time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg, tot_sparsity_avg, n_poly_avg,
                 n_zeros_avg, n_entries_avg, n_terms_avg, degree_avg) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                X_train_chunks = np.array_split(self.X_train, self.cv)
                y_train_chunks = np.array_split(self.y_train, self.cv)
                for idx in range(0, len(X_train_chunks)):
                    X_t, y_t, X_v, y_v = self.get_chunks(X_train_chunks, y_train_chunks, idx=idx)
                    oracle_avi = OracleAVI(psi=psi, oracle_type=oracle_type,
                                           term_ordering_strategy=term_ordering_strategy, border_type=border_type,
                                           inverse_hessian_boost=inverse_hessian_boost)
                    (time_t, time_v, acc_t, acc_v, sparsity, tot_sparsity, n_poly, n_zeros, n_entries,
                     n_terms, degree) = feature_transformation_for_linear_svm(
                        oracle_avi, C, fd(X_t), fd(y_t), fd(X_v), fd(y_v))
                    time_t_avg += time_t / self.cv
                    time_v_avg += time_v / self.cv
                    acc_t_avg += acc_t / self.cv
                    acc_v_avg += acc_v / self.cv
                    sparsity_avg += sparsity / self.cv
                    tot_sparsity_avg += tot_sparsity / self.cv
                    n_poly_avg += n_poly / self.cv
                    n_zeros_avg += n_zeros / self.cv
                    n_entries_avg += n_entries / self.cv
                    n_terms_avg += n_terms / self.cv
                    degree_avg += degree / self.cv

                row = pd.DataFrame([[oracle_type, term_ordering_strategy, border_type, inverse_hessian_boost, psi, C,
                                     time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg, tot_sparsity_avg,
                                     n_poly_avg, n_zeros_avg, n_entries_avg, n_terms_avg, degree_avg]],
                                   columns=columns)
                df = df.append(row, ignore_index=True)

        return df

    def hyperparameter_combinations_svm(self):
        """Tests all hyperparameter combinations for polynomial kernel SVM and returns a data frame containing the
        results.

        Returns:
            df: pd.DataFrame
        """
        Cs = self.hyperparameters["C"]
        degrees = self.hyperparameters['avg_degree']

        columns = ['C', 'avg_degree', 'time_t', 'time_v', 'accuracy_t', 'accuracy_v']
        df = pd.DataFrame(columns=columns, dtype=float)

        for C in Cs:
            for degree in degrees:
                time_t_avg, time_v_avg, acc_t_avg, acc_v_avg = 0, 0, 0, 0
                X_train_chunks = np.array_split(self.X_train, self.cv)
                y_train_chunks = np.array_split(self.y_train, self.cv)
                for idx in range(0, len(X_train_chunks)):
                    X_t, y_t, X_v, y_v = self.get_chunks(X_train_chunks, y_train_chunks, idx=idx)

                    classifier = svm.SVC(kernel='poly', C=C, degree=degree, max_iter=10000, cache_size=6000)

                    # Train
                    timer = time.time()
                    classifier.fit(X_t, y_t.ravel())
                    acc_t_avg += classifier.score(X_t, y_t.ravel()) / self.cv
                    time_t_avg += time.time() - timer

                    # Test
                    timer = time.time()
                    acc_v_avg += classifier.score(X_v, y_v) / self.cv
                    time_v_avg += time.time() - timer

                row = pd.DataFrame([[C, degree, time_t_avg, time_v_avg, acc_t_avg, acc_v_avg]], columns=columns)

                df = df.append(row, ignore_index=True)

        return df

    def hyperparameter_combinations_vca(self):
        """Tests all hyperparameter combinations for VCA and linear kernel SVM and returns a data frame containing the
        results.

        Returns:
            df: pd.DataFrame
        """
        psis = self.hyperparameters["psi"]
        Cs = self.hyperparameters["C"]

        columns = ['psi', 'C', 'time_t', 'time_v', 'accuracy_t', 'accuracy_v', 'avg_sparsity', 'total_sparsity',
                   'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms', 'avg_degree']
        df = pd.DataFrame(columns=columns, dtype=float)

        for psi in psis:
            for C in Cs:
                (time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg, tot_sparsity_avg, n_poly_avg, n_zeros_avg,
                 n_entries_avg, n_terms_avg, degree_avg) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                X_train_chunks = np.array_split(self.X_train, self.cv)
                y_train_chunks = np.array_split(self.y_train, self.cv)
                for idx in range(0, len(X_train_chunks)):
                    X_t, y_t, X_v, y_v = self.get_chunks(X_train_chunks, y_train_chunks, idx=idx)
                    vca = VCA(psi=psi)
                    (time_t, time_v, acc_t, acc_v, sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms, degree
                     ) = feature_transformation_for_linear_svm(vca, C, fd(X_t), fd(y_t), fd(X_v), fd(y_v))
                    time_t_avg += time_t / self.cv
                    time_v_avg += time_v / self.cv
                    acc_t_avg += acc_t / self.cv
                    acc_v_avg += acc_v / self.cv
                    sparsity_avg += sparsity / self.cv
                    tot_sparsity_avg += tot_sparsity / self.cv
                    n_poly_avg += n_poly / self.cv
                    n_zeros_avg += n_zeros / self.cv
                    n_entries_avg += n_entries / self.cv
                    n_terms_avg += n_terms / self.cv
                    degree_avg += degree / self.cv

                row = pd.DataFrame([[psi, C, time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg,
                                     tot_sparsity_avg, n_poly_avg, n_zeros_avg, n_entries_avg, n_terms_avg,
                                     degree_avg]], columns=columns)
                df = df.append(row, ignore_index=True)

        return df

    def final_training_avi(self, data_frame: pd.DataFrame, count_svm: bool = True):
        """Performs the final retraining for AVI and linear kernel SVM, with the optimal hyperparameter choices."""
        columns = ['term_ordering_strategy', 'border_type', 'psi', 'tau', 'C', 'time_train', 'time_test',
                   'accuracy_train', 'accuracy_test', 'avg_sparsity', 'total_sparsity', 'number_of_polynomials',
                   'number_of_zeros', 'number_of_entries', 'number_of_terms', 'avg_degree']
        term_ordering_strategy = self.hyperparameters["term_ordering_strategy"]
        border_type = self.hyperparameters["border_type"]
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        psi = row['psi']
        tau = row['tau']
        C = row['C']
        avi = AVI(psi=psi, tau=tau, term_ordering_strategy=term_ordering_strategy, border_type=border_type)

        (time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
         degree) = feature_transformation_for_linear_svm(avi, C, self.X_train, self.y_train, self.X_test, self.y_test,
                                                         count_svm)

        df_test = pd.DataFrame([[term_ordering_strategy, border_type, psi, tau, C, time_train, time_test, acc_train,
                                 acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
                                 degree]], columns=columns)
        return df_test

    def final_training_oavi(self, data_frame: pd.DataFrame, count_svm: bool = True):
        """Performs the final retraining for OAVI and linear kernel SVM, with the optimal hyperparameter choices."""
        columns = ['oracle_type', 'term_ordering_strategy', 'border_type', 'inverse_hessian_boost', 'psi', 'C',
                   'time_train', 'time_test', 'accuracy_train', 'accuracy_test', 'avg_sparsity', 'total_sparsity',
                   'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms', 'avg_degree']
        oracle_type = self.hyperparameters["oracle_type"]
        term_ordering_strategy = self.hyperparameters["term_ordering_strategy"]
        border_type = self.hyperparameters["border_type"]
        inverse_hessian_boost = self.hyperparameters["inverse_hessian_boost"]
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        psi = row['psi']
        C = row['C']
        oracle_avi = OracleAVI(psi=psi, oracle_type=oracle_type, term_ordering_strategy=term_ordering_strategy,
                               border_type=border_type, inverse_hessian_boost=inverse_hessian_boost)

        (time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
         degree) = feature_transformation_for_linear_svm(oracle_avi, C, self.X_train, self.y_train, self.X_test,
                                                         self.y_test, count_svm)

        df_test = pd.DataFrame(
            [[oracle_type, term_ordering_strategy, border_type, inverse_hessian_boost, psi, C, time_train, time_test,
              acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms, degree]],
            columns=columns)
        return df_test

    def final_training_svm(self, data_frame: pd.DataFrame):
        """Performs the final retraining for VCA and linear kernel SVM, with the optimal hyperparameter choices."""
        columns = ['C', 'avg_degree', 'time_train', 'time_test', 'accuracy_train', 'accuracy_test']
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        C = row['C']
        degree = row['avg_degree']

        classifier = svm.SVC(kernel='poly', C=C, degree=degree, max_iter=10000, cache_size=6000)

        # Train
        timer = time.time()
        classifier.fit(self.X_train, self.y_train.ravel())
        acc_train = classifier.score(self.X_train, self.y_train.ravel())
        time_train = time.time() - timer

        # Test
        timer = time.time()
        acc_test = classifier.score(self.X_test, self.y_test)
        time_test = time.time() - timer

        df_test = pd.DataFrame([[C, degree, time_train, time_test, acc_train, acc_test]], columns=columns)
        return df_test

    def final_training_vca(self, data_frame: pd.DataFrame, count_svm: bool = True):
        """Performs the final retraining for VCA and linear kernel SVM, with the optimal hyperparameter choices."""
        columns = ['psi', 'C', 'time_train', 'time_test', 'accuracy_train', 'accuracy_test', 'avg_sparsity',
                   'total_sparsity', 'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms',
                   'avg_degree']
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        psi = row['psi']
        C = row['C']
        vca = VCA(psi=psi)

        (time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
         degree) = feature_transformation_for_linear_svm(vca, C, self.X_train, self.y_train, self.X_test, self.y_test,
                                                         count_svm=count_svm)

        df_test = pd.DataFrame([[psi, C, time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly,
                                 n_zeros, n_entries, n_terms, degree]],
                               columns=columns)

        return df_test


def fetch_and_prepare_data(name, proportion: float = 0.6):
    """Downloads and prepares the data."""
    X, y = fetch_data_set(name=name)
    X, y = unison_shuffled_copies(X, y)
    X_train, y_train, X_test, y_test = train_test_split(X, y, proportion=proportion)
    X_train, X_test = min_max_feature_scaling(X_train, X_test)
    return fd(X_train), fd(y_train), fd(X_test), fd(y_test)


def feature_transformation_for_linear_svm(feature_transformation, C: float, X_train: np.ndarray, y_train: np.ndarray,
                                          X_test: np.ndarray, y_test: np.ndarray, count_svm: bool = True):
    """Runs feature_transformation OAVI, ABM, AVI, or VCA and linear kernel SVM with the given hyperparameters.

    Args:
        feature_transformation
            OAVI, ABM, AVI, or VCA.
        C: float
            Regularization for the SVM.
        X_train: np.ndarray
        y_train: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        count_svm: bool, Optional
            Whether to add the time related to the SVM training to the counted time. (Default is True.)

    Returns:
        train_time: float
            Time needed to train feature_transformation and linear kernel SVM.
        test_time: float
            Time needed to evaluate feature_transformation and linear kernel SVM on the test set.
        train_accuracy: float
            Accuracy on the train set.
        test_accuracy: float
            Accuracy on the test set.
        avg_sparsity: float
            Average sparsity of the polynomials constructed by feature_transformation.
        total_sparsity: float
            Number of zeros in coefficient vectors divided by (total number of entries - total number of polynomials).
        number_of_polynomials: int
            Number of polynomials constructed by feature_transformation.
        number_of_zeros: int
            Number of coefficient vector entries that are zero of polynomials summed over all classes.
        number_of_entries: int
            Number of coefficient vector entries of polynomials summed over all classes.
        number_of_terms: int
            Total number of terms in O summed over all classes.
    """
    train_accuracy = 0
    test_accuracy = 0
    train_time = 0
    test_time = 0
    avg_sparsity = 0
    number_of_zeros = 0
    number_of_entries = 0
    number_of_polynomials = 0
    number_of_terms = 0
    degree = 0

    X_train_classes = split_into_classes(fd(X_train), fd(y_train))

    polynomials_train = []
    polynomials_test = []

    for X_train_class in X_train_classes:
        # Train
        timer = time.time()
        feature_transformation.fit(cp.array(X_train_class))
        X_train_polynomial_evaluations, _ = feature_transformation.evaluate(cp.array(X_train))
        polynomials_train.append(X_train_polynomial_evaluations)
        train_time += time.time() - timer

        # Test
        timer = time.time()
        X_test_polynomial_evaluations, _ = feature_transformation.evaluate(cp.array(X_test))
        polynomials_test.append(X_test_polynomial_evaluations)
        test_time += time.time() - timer

        (n_zeros, n_entries, sparsity, n_polynomials, n_terms, tmp_degree
         ) = feature_transformation.evaluate_transformation()
        avg_sparsity += sparsity * n_polynomials
        number_of_polynomials += n_polynomials
        number_of_zeros += n_zeros
        number_of_entries += n_entries
        number_of_terms += n_terms
        degree += tmp_degree * n_polynomials

    avg_sparsity /= number_of_polynomials
    total_sparsity = number_of_zeros / max((number_of_entries - number_of_polynomials), 1)
    degree /= number_of_polynomials

    # Train
    timer = time.time()
    cont = True
    for i in range(0, len(polynomials_train)):
        if polynomials_train[i] is None:
            cont = False
            break

    if cont:
        X_train_transformed = cp.asnumpy(fd(cp.hstack(polynomials_train)))
        train_time += time.time() - timer

        # Test
        timer = time.time()
        X_test_transformed = cp.asnumpy(fd(cp.hstack(polynomials_test)))
        test_time += time.time() - timer

        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

            classifier = svm.LinearSVC(C=C, penalty='l1', loss='squared_hinge', random_state=0, dual=False,
                                       max_iter=10000)

        if count_svm:
            # Train
            timer = time.time()
            classifier.fit(X_train_transformed, y_train.ravel())
            train_accuracy = classifier.score(X_train_transformed, y_train)
            train_time += time.time() - timer
            # if count_svm:
            #     train_time += time.time() - timer

            # Test
            timer = time.time()
            test_accuracy = classifier.score(X_test_transformed, y_test)
            test_time += time.time() - timer
            # if count_svm:
            #     test_time += time.time() - timer

    else:
        train_time = 1e+16
        test_time = 1e+16

    return (train_time, test_time, train_accuracy, test_accuracy, avg_sparsity, total_sparsity, number_of_polynomials,
            number_of_zeros, number_of_entries, number_of_terms, degree)


def perform_experiments(data_sets, paras: dict, saving: bool = False):
    for name in data_sets:
        cp.random.seed(n_seed_)
        np.random.seed(n_seed_)
        for i in range(0, n_runs_):
            pc = ExperimentSetups(name, paras)
            pc.cross_validation(saving=saving)

