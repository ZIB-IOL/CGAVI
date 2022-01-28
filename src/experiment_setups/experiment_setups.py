import os
import sys
import time
import warnings

import pandas as pd

from sklearn import svm
import cupy as cp
import numpy as np

from global_ import n_seed_, number_of_runs_
from src.auxiliary_functions.auxiliary_functions import fd
from src.data_frames.data_frames import save_data_frame
from src.data_sets.data_set_creation import fetch_data_set
from src.data_sets.preprocessing import train_test_split, min_max_feature_scaling, split_into_classes, \
    unison_shuffled_copies
from src.feature_transformations.avi import AVI
from src.feature_transformations.oracle_avi import OracleAVI
from src.feature_transformations.vca import VCA


class ExperimentSetups:
    """
    Class managing the entire experimental setup for performance comparison.

    Args:
        name_data_set: str
        hyperparameters: dict
        cv: int, Optional
            Number of cross validation splits. (Default is 5.)
        samples_per_class_synthetic: int, Optional
            (Default is 200.)
        noise_synthetic: int, Optional
            (Default is 0.1.)

    Methods:
        fetch_and_prepare_data()
            Downloads and prepares the data.
        get_chunks(X_train_chunks: np.ndarray, y_train_chunks: np.ndarray, idx: int)
            Turns the ith chunk into the validation set and the other chunks into the train set.
        cross_validation(saving: bool)
            Performs cross validation for the given algorithm.
        hyperparameter_combinations_avi()
            Tests all hyperparameter combinations for AVI and Linear Kernel SVM and returns a data frame containing the
            results.
        hyperparameter_combinations_cgavi()
            Tests all hyperparameter combinations for CGAVI and Linear Kernel SVM and returns a data frame
        containing the results.
        hyperparameter_combinations_agdavi()
            Tests all hyperparameter combinations for AGDAVI and Linear Kernel SVM and returns a data frame
        containing the results.
        hyperparameter_combinations_svm()
            Tests all hyperparameter combinations for Polynomial Kernel SVM and returns a data frame containing the
            results.
        hyperparameter_combinations_vca()
            Tests all hyperparameter combinations for VCA and Linear Kernel SVM and returns a data frame containing the
            results.
        final_training_cgavi()
            Performs the final retraining for CGAVI and Linear Kernel SVM, with the optimal hyperparameter choices.
        final_training_agdavi()
            Performs the final retraining for AGDAVI and Linear Kernel SVM, with the optimal hyperparameter choices.
        final_training_svm()
            Performs the final retraining for VCA and Linear Kernel SVM, with the optimal hyperparameter choices.
        final_training_vca()
            Performs the final retraining for VCA and Linear Kernel SVM, with the optimal hyperparameter choices.
    """

    def __init__(self, name_data_set: str, hyperparameters: dict, cv: int = 3, samples_per_class_synthetic: int = 200,
                 noise_synthetic: float = 0.1):
        self.data_set_name = name_data_set
        self.algorithm_name = hyperparameters["algorithm"]
        self.samples_per_class_synthetic = samples_per_class_synthetic
        self.noise_synthetic = noise_synthetic
        self.X_train, self.y_train, self.X_test, self.y_test = self.fetch_and_prepare_data()
        self.hyperparameters = hyperparameters
        self.cv = cv
        self.experiment = "performance"

    def fetch_and_prepare_data(self):
        """Downloads and prepares the data."""
        return fetch_and_prepare_data(self.data_set_name, self.samples_per_class_synthetic, self.noise_synthetic)

    def get_chunks(self, X_train_chunks: np.ndarray, y_train_chunks: np.ndarray, idx: int):
        """Turns the ith chunk into the validation set and the other chunks into the train set.

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

    def cross_validation(self, saving: bool = True):
        """
        Performs cross validation for the given algorithm.
        """
        df = None
        df_test = None

        if self.algorithm_name == "avi":
            timer = time.time()
            df = self.hyperparameter_combinations_avi()
            timer = time.time() - timer
        elif self.algorithm_name in ['l1_cgavi', 'l2_cgavi']:
            timer = time.time()
            df = self.hyperparameter_combinations_cgavi()
            timer = time.time() - timer
        elif self.algorithm_name == 'agdavi':
            timer = time.time()
            df = self.hyperparameter_combinations_agdavi()
            timer = time.time() - timer
        elif self.algorithm_name == "svm":
            timer = time.time()
            df = self.hyperparameter_combinations_svm()
            timer = time.time() - timer
        elif self.algorithm_name == "vca":
            timer = time.time()
            df = self.hyperparameter_combinations_vca()
            timer = time.time() - timer

        assert df is not None

        if saving:
            save_data_frame(df, self.algorithm_name, self.data_set_name, self.experiment, test=False, number=-1)

        if self.algorithm_name == "avi":
            df_test = self.final_training_avi(df)
        elif self.algorithm_name in ['l1_cgavi', 'l2_cgavi']:
            df_test = self.final_training_cgavi(df)
        elif self.algorithm_name == 'agdavi':
            df_test = self.final_training_agdavi(df)
        elif self.algorithm_name == "svm":
            df_test = self.final_training_svm(df)
        elif self.algorithm_name == "vca":
            df_test = self.final_training_vca(df)

        assert df_test is not None, "Why is this None?"

        df_test['hyperparameter_optimization_time'] = timer

        if self.data_set_name == 'synthetic':
            df_test['samples_train'] = self.X_train.shape[0]
            df_test['samples_test'] = self.X_test.shape[0]
        if saving:
            save_data_frame(df_test, self.algorithm_name, self.data_set_name, self.experiment, test=True, number=-1)

        print("-----------------------------------------------------------------------------------------------------")
        print("Training")
        print(df.T)
        print("-----------------------------------------------------------------------------------------------------")
        print("Testing")
        print(df_test.T)
        print("-----------------------------------------------------------------------------------------------------")

        return df_test

    def hyperparameter_combinations_avi(self):
        """Tests all hyperparameter combinations for AVI and Linear Kernel SVM and returns a data frame containing the
        results.

        Returns:
            df: pd.DataFrame
        """
        psis = self.hyperparameters["psi"]
        taus = self.hyperparameters["tau"]
        Cs = self.hyperparameters["C"]

        columns = ['psi', 'tau', 'C', 'time_t', 'time_v', 'accuracy_t', 'accuracy_v', 'avg_sparsity', 'total_sparsity',
                   'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms', 'degree']
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
                        avi = AVI(psi=psi, tau=tau)
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

                    row = pd.DataFrame([[psi, tau, C, time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg,
                                         tot_sparsity_avg, n_poly_avg, n_zeros_avg, n_entries_avg, n_terms_avg,
                                         degree_avg]], columns=columns)

                    df = df.append(row, ignore_index=True)
        return df

    def hyperparameter_combinations_cgavi(self):
        """Tests all hyperparameter combinations for CGAVI and Linear Kernel SVM and returns a data frame
        containing the results.

        Returns:
            df: pd.DataFrame
        """
        psis = self.hyperparameters["psi"]
        eps_factors = self.hyperparameters["eps_factor"]
        taus = self.hyperparameters["tau"]
        lmbdas = self.hyperparameters["lmbda"]
        Cs = self.hyperparameters["C"]
        tol = self.hyperparameters["tol"]

        columns = ['psi', 'eps', 'tau', 'lmbda', 'C', 'time_t', 'time_v', 'accuracy_t', 'accuracy_v', 'avg_sparsity',
                   'total_sparsity', 'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms',
                   'degree']
        df = pd.DataFrame(columns=columns, dtype=float)
        for psi in psis:
            for eps_factor in eps_factors:
                eps = psi * eps_factor
                for tau in taus:
                    for lmbda in lmbdas:
                        for C in Cs:
                            (time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg, tot_sparsity_avg, n_poly_avg,
                             n_zeros_avg, n_entries_avg, n_terms_avg, degree_avg) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                            X_train_chunks = np.array_split(self.X_train, self.cv)
                            y_train_chunks = np.array_split(self.y_train, self.cv)
                            for idx in range(0, len(X_train_chunks)):
                                X_t, y_t, X_v, y_v = self.get_chunks(X_train_chunks, y_train_chunks, idx=idx)
                                if self.algorithm_name == 'l1_cgavi':
                                    region_type = 'L1Ball'
                                elif self.algorithm_name == 'l2_cgavi':
                                    region_type = 'L2Ball'
                                oracle_avi = OracleAVI(psi=psi, eps=eps, tau=tau, lmbda=lmbda, region_type=region_type,
                                                       tol=tol)
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

                            row = pd.DataFrame([[psi, eps, tau, lmbda, C, time_t_avg, time_v_avg, acc_t_avg, acc_v_avg,
                                                 sparsity_avg, tot_sparsity_avg, n_poly_avg, n_zeros_avg, n_entries_avg,
                                                 n_terms_avg, degree_avg]], columns=columns)
                            df = df.append(row, ignore_index=True)

        return df

    def hyperparameter_combinations_agdavi(self):
        """Tests all hyperparameter combinations for AGDAVI and Linear Kernel SVM and returns a data frame
        containing the results.

        Returns:
            df: pd.DataFrame
        """
        psis = self.hyperparameters["psi"]
        lmbdas = self.hyperparameters["lmbda"]
        Cs = self.hyperparameters["C"]
        tol = self.hyperparameters["tol"]

        columns = ['psi', 'lmbda', 'C', 'time_t', 'time_v', 'accuracy_t', 'accuracy_v', 'avg_sparsity',
                   'total_sparsity', 'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms',
                   'degree']
        df = pd.DataFrame(columns=columns, dtype=float)
        for psi in psis:
            for lmbda in lmbdas:
                for C in Cs:
                    (time_t_avg, time_v_avg, acc_t_avg, acc_v_avg, sparsity_avg, tot_sparsity_avg, n_poly_avg,
                     n_zeros_avg, n_entries_avg, n_terms_avg, degree_avg) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                    X_train_chunks = np.array_split(self.X_train, self.cv)
                    y_train_chunks = np.array_split(self.y_train, self.cv)
                    for idx in range(0, len(X_train_chunks)):
                        X_t, y_t, X_v, y_v = self.get_chunks(X_train_chunks, y_train_chunks, idx=idx)
                        oracle_avi = OracleAVI(psi=psi, lmbda=lmbda, oracle_type='AGD', tol=tol)
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

                    row = pd.DataFrame([[psi, lmbda, C, time_t_avg, time_v_avg, acc_t_avg, acc_v_avg,
                                         sparsity_avg, tot_sparsity_avg, n_poly_avg, n_zeros_avg, n_entries_avg,
                                         n_terms_avg, degree_avg]], columns=columns)

                    df = df.append(row, ignore_index=True)

        return df

    def hyperparameter_combinations_svm(self):
        """Tests all hyperparameter combinations for Polynomial Kernel SVM and returns a data frame containing the
        results.

        Returns:
            df: pd.DataFrame
        """

        Cs = self.hyperparameters["C"]
        degrees = self.hyperparameters["degree"]

        columns = ['C', 'degree', 'time_t', 'time_v', 'accuracy_t', 'accuracy_v']
        df = pd.DataFrame(columns=columns, dtype=float)

        for C in Cs:
            for degree in degrees:
                time_t_avg, time_v_avg, acc_t_avg, acc_v_avg = 0, 0, 0, 0
                X_train_chunks = np.array_split(self.X_train, self.cv)
                y_train_chunks = np.array_split(self.y_train, self.cv)
                for idx in range(0, len(X_train_chunks)):
                    X_t, y_t, X_v, y_v = self.get_chunks(X_train_chunks, y_train_chunks, idx=idx)

                    classifier = svm.SVC(kernel='poly', C=C, degree=degree)

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
        """Tests all hyperparameter combinations for VCA and Linear Kernel SVM and returns a data frame containing the
        results.

        Returns:
            df: pd.DataFrame
        """
        psis = self.hyperparameters["psi"]
        Cs = self.hyperparameters["C"]

        columns = ['psi', 'C', 'time_t', 'time_v', 'accuracy_t', 'accuracy_v', 'avg_sparsity', 'total_sparsity',
                   'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms', 'degree']
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

    def final_training_avi(self, data_frame: pd.DataFrame):
        """Performs the final retraining for AVI and Linear Kernel SVM, with the optimal hyperparameter choices.
        """
        columns = ['psi', 'tau', 'C', 'time_train', 'time_test', 'accuracy_train', 'accuracy_test', 'avg_sparsity',
                   'total_sparsity', 'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms',
                   'degree']
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        psi = row['psi']
        tau = row['tau']
        C = row['C']
        avi = AVI(psi=psi, tau=tau)

        (time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
         degree) = feature_transformation_for_linear_svm(avi, C, self.X_train, self.y_train, self.X_test, self.y_test)

        df_test = pd.DataFrame([[psi, tau, C, time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity,
                                 n_poly, n_zeros, n_entries, n_terms, degree]], columns=columns)
        return df_test

    def final_training_cgavi(self, data_frame: pd.DataFrame):
        """Performs the final retraining for CGAVI and Linear Kernel SVM, with the optimal hyperparameter choices.
        """
        columns = ['psi', 'eps', 'tau', 'lmbda', 'C', 'time_train', 'time_test', 'accuracy_train', 'accuracy_test',
                   'avg_sparsity', 'total_sparsity', 'number_of_polynomials', 'number_of_zeros', 'number_of_entries',
                   'number_of_terms', 'degree']
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        psi = row['psi']
        eps = row['eps']
        tau = row['tau']
        lmbda = row['lmbda']
        tol = self.hyperparameters['tol']
        C = row['C']
        if self.algorithm_name == 'l1_cgavi':
            region_type = 'L1Ball'
        elif self.algorithm_name == 'l2_cgavi':
            region_type = 'L2Ball'
        oracle_avi = OracleAVI(psi=psi, eps=eps, tau=tau, lmbda=lmbda, region_type=region_type, tol=tol)

        (time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
         degree) = feature_transformation_for_linear_svm(oracle_avi, C, self.X_train, self.y_train, self.X_test,
                                                         self.y_test)

        df_test = pd.DataFrame(
            [[psi, eps, tau, lmbda, C, time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly,
              n_zeros, n_entries, n_terms, degree]], columns=columns)
        return df_test

    def final_training_agdavi(self, data_frame: pd.DataFrame):
        """Performs the final retraining for AGDAVI and Linear Kernel SVM, with the optimal hyperparameter choices.
        """
        columns = ['psi', 'lmbda', 'C', 'time_train', 'time_test', 'accuracy_train', 'accuracy_test',
                   'avg_sparsity', 'total_sparsity', 'number_of_polynomials', 'number_of_zeros', 'number_of_entries',
                   'number_of_terms', 'degree']
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        psi = row['psi']
        lmbda = row['lmbda']
        C = row['C']
        tol = self.hyperparameters["tol"]

        oracle_avi = OracleAVI(psi=psi, lmbda=lmbda, oracle_type='AGD', tol=tol)
        (time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
         degree) = feature_transformation_for_linear_svm(oracle_avi, C, self.X_train, self.y_train, self.X_test,
                                                         self.y_test)

        df_test = pd.DataFrame(
            [[psi, lmbda, C, time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros,
              n_entries, n_terms, degree]], columns=columns)
        return df_test

    def final_training_svm(self, data_frame: pd.DataFrame):
        """Performs the final retraining for VCA and Linear Kernel SVM, with the optimal hyperparameter choices."""
        columns = ['C', 'degree', 'time_train', 'time_test', 'accuracy_train', 'accuracy_test']
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        C = row['C']
        degree = row['degree']

        classifier = svm.SVC(kernel='poly', C=C, degree=degree)

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

    def final_training_vca(self, data_frame: pd.DataFrame):
        """Performs the final retraining for VCA and Linear Kernel SVM, with the optimal hyperparameter choices."""
        columns = ['psi', 'C', 'time_train', 'time_test', 'accuracy_train', 'accuracy_test', 'avg_sparsity',
                   'total_sparsity', 'number_of_polynomials', 'number_of_zeros', 'number_of_entries', 'number_of_terms',
                   'degree']
        idx = data_frame['accuracy_v'].idxmax()
        row = data_frame.iloc[idx]
        psi = row['psi']
        C = row['C']
        vca = VCA(psi=psi)

        (time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly, n_zeros, n_entries, n_terms,
         degree) = feature_transformation_for_linear_svm(vca, C, self.X_train, self.y_train, self.X_test, self.y_test)

        df_test = pd.DataFrame([[psi, C, time_train, time_test, acc_train, acc_test, avg_sparsity, tot_sparsity, n_poly,
                                 n_zeros, n_entries, n_terms, degree]],
                               columns=columns)

        return df_test


def fetch_and_prepare_data(name, samples_per_class_synthetic, noise_synthetic, proportion: float = 0.6):
    """Downloads and prepares the data."""
    X, y = fetch_data_set(name=name, samples_per_class_synthetic=samples_per_class_synthetic,
                          noise_synthetic=noise_synthetic)
    X, y = unison_shuffled_copies(X, y)
    X_train, y_train, X_test, y_test = train_test_split(X, y, proportion=proportion)
    X_train, X_test = min_max_feature_scaling(X_train, X_test)
    return fd(X_train), fd(y_train), fd(X_test), fd(y_test)


def feature_transformation_for_linear_svm(transformer, C: float, X_train: np.ndarray, y_train: np.ndarray,
                                          X_test: np.ndarray, y_test: np.ndarray):
    """Runs transformer (either AVI, Oracle AVI, or VCA) and Linear Kernel SVM with the given parameters.

    Args:
        transformer: instance of AVI, OracleAVI, or VCA
        C: float
        X_train: np.ndarray
        y_train: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray

    Returns:
        train_time: float
            Time needed to train transformer and Linear Kernel SVM.
        test_time: float
            Time needed to evaluate transformer and Linear Kernel SVM on the test set.
        train_accuracy: float
            Accuracy on the train set.
        test_accuracy: float
            Accuracy on the test set.
        avg_sparsity: float
            Average sparsity of the polynomials constructed by transformer.
        total_sparsity: float
            Number of zeros divided by (total number of entries - total number of polynomials).
        number_of_polynomials: int
            Number of polynomials constructed by transformer.
        number_of_zeros: int
            Number of coefficient vector entries that are zero of polynomials summed over all classes.
        number_of_entries: int
            Number of coefficient vector entries of polynomials summed over all classes.
        number_of_terms: int
            Total number of terms in O summed over all classes.
    """

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
        transformer.fit(cp.array(X_train_class))
        X_train_polynomial_evaluations, _ = transformer.evaluate(cp.array(X_train))
        polynomials_train.append(X_train_polynomial_evaluations)
        train_time += time.time() - timer

        # Test
        timer = time.time()
        X_test_polynomial_evaluations, _ = transformer.evaluate(cp.array(X_test))
        polynomials_test.append(X_test_polynomial_evaluations)
        test_time += time.time() - timer

        n_zeros, n_entries, sparsity, n_polynomials, n_terms, tmp_degree = transformer.evaluate_sparsity()
        avg_sparsity += sparsity * n_polynomials
        number_of_polynomials += n_polynomials
        number_of_zeros += n_zeros
        number_of_entries += n_entries
        number_of_terms += n_terms
        degree = max(tmp_degree, degree)

    avg_sparsity /= number_of_polynomials
    total_sparsity = number_of_zeros / max((number_of_entries - number_of_polynomials), 1)

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
                                       max_iter=1000)

        # Train
        timer = time.time()
        classifier.fit(X_train_transformed, y_train.ravel())
        train_accuracy = classifier.score(X_train_transformed, y_train)
        train_time += time.time() - timer

        # Test
        timer = time.time()
        test_accuracy = classifier.score(X_test_transformed, y_test)
        test_time += time.time() - timer
    else:
        test_time = 1000
        train_time = 1000
        train_accuracy = 0
        test_accuracy = 0

    return (train_time, test_time, train_accuracy, test_accuracy, avg_sparsity, total_sparsity, number_of_polynomials,
            number_of_zeros, number_of_entries, number_of_terms, degree)


def perform_experiments(data_sets, paras: dict, saving: bool = False, samples_per_class_synthetic: int = 200):
    for name in data_sets:
        cp.random.seed(n_seed_)
        np.random.seed(n_seed_)
        for i in range(0, number_of_runs_):
            pc = ExperimentSetups(name, paras, samples_per_class_synthetic=samples_per_class_synthetic)
            pc.cross_validation(saving=saving)


def create_logspace(low: float, high: float, n: int = 2, reverse: bool = True):
    """Creates a logspace from low to high with n elements."""
    space = list(np.logspace(np.log(low) / np.log(10), np.log(high) / np.log(10), n))
    if reverse:
        space.reverse()
    return space
