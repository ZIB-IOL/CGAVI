import numpy as np
from src.auxiliary_functions.auxiliary_functions import fd
from sklearn.preprocessing import MinMaxScaler


def unison_shuffled_copies(array_1: np.ndarray, array_2: np.ndarray):
    """Shuffles the rows of two numpy arrays in unison."""
    assert len(array_1) == len(array_2)
    p = np.random.permutation(array_1.shape[0])
    return fd(array_1[p, :]), fd(array_2[p, :])


def train_test_split(X: np.ndarray, y: np.ndarray, proportion: float = 0.8):
    """Splits data X and labels y into train and test data and labels, respectively.

    Args:
        X: np.ndarray
            Data.
        y: np.ndarray
            Labels.
        proportion: float, Optional
            The proportion of the data and labels that is used for training. (Default is 0.8.)

    Returns:
        X_train: np.ndarray
            Training data.
        y_train: np.ndarray
            Training labels.
        X_test: np.ndarray
            Test data.
        y_test: np.ndarray
            Test labels.
    """
    cutoff = int(fd(X).shape[0] * proportion)
    X_train = fd(X[:cutoff, :])
    y_train = fd(fd(y)[:cutoff, :])
    X_test = fd(X[cutoff:, :])
    y_test = fd(fd(y)[cutoff:, :])
    return X_train, y_train, X_test, y_test


def min_max_feature_scaling(X_train: np.ndarray, X_test: np.ndarray):
    """Applies min-max feature scaling based on X_train to both X_train and X_test.

    Args:
        X_train: np.ndarray
            Training data.
        X_test: np.ndarray
            Test data.

    Returns:
        X_train_scaled: np.ndarray
            The data set X_train with min-max feature scaling based on X_train.
        X_test_scaled: np.ndarray
            The data set X_test with min-max feature scaling based on X_train.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def split_into_classes(X: np.ndarray, y: np.ndarray):
    """Splits the data X into a list containing the subsets corresponding to samples of one class."""
    values = set(y.flatten().tolist())
    values = list(values)
    X_values = []
    for value in values:
        X_value = X[(y.flatten() == value).tolist(), :]
        X_values.append(X_value)
    return X_values


