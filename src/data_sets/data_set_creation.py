from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow import keras
import pandas as pd
import numpy as np
from src.auxiliary_functions.auxiliary_functions import fd


def fetch_data_set(name: str, samples_per_class_synthetic: int = 100, noise_synthetic: float = 0.1):
    """
    Loads the data sets.
    Args:

    Args:
        name: str
            Name of the data set.
            data set name:  # samples                       /   # features  /   # classes
            - 'abalone':    4067                            /   10          /   16
            - 'banknote':   1372                            /   4           /   2
            - 'cancer':     569                             /   30          /   2
            - 'digits':     1797                            /   64          /   10
            - 'htru2':      17898                           /   8           /   2
            - 'iris':       150                             /   4           /   3
            - 'madelon':    2600                            /   500         /   2
            - 'seeds':      210                             /   7           /   3
            - 'sonar':      208                             /   60          /   2
            - 'spam':       4601                            /   57          /   2
            - 'synthetic': 2 x samples_per_class_synthetic  /   3           /   2
            - 'voice':      126                             /   310         /   2
            - 'wine':       178                             /   13          /   3
        samples_per_class_synthetic: int, Optional
            (Default is 100.)
        noise_synthetic: int, Optional
            (Default is 0.1.)

    Returns:
        X: np.ndarray
            Data.
        y: np.ndarray
            Labels.
    """
    if name == "abalone":
        X, y = download_abalone()
    elif name == 'banknote':
        X, y = download_banknote()
    elif name == 'cancer':
        X, y = download_cancer()
    elif name == 'digits':
        X, y = download_digits()
    elif name == 'htru2':
        X, y = download_htru2()
    elif name == 'iris':
        X, y = download_iris()
    elif name == 'madelon':
        X, y = download_madelon()
    elif name == 'sonar':
        X, y = download_sonar()
    elif name == 'spam':
        X, y = download_spam()
    elif name == 'synthetic':
        X, y = create_synthetic_data(samples_per_class_synthetic, noise_synthetic)
    elif name == 'seeds':
        X, y = download_seeds()
    elif name == 'voice':
        X, y = download_voice()
    elif name == 'wine':
        X, y = download_wine()
    else:
        X, y = None, None
        print("No valid data set was selected.")
    return fd(X), fd(y)


def download_abalone():
    """
    Downloads the 'abalone' data set, turns the 'Sex' category to three numerical features: 'Male', 'Female', and
    'Infant', and then delets all classes except the ones with {5, 6, ..., 20} 'Rings', ultimately culminating in a data
    set of 4067 samples with 10 features 'Male', 'Female', 'Infant', 'Length', 'Diameter', 'Height', 'Whole weight',
    'Shucked weight', 'Viscera weight', and 'Shell weight' and the label 'Rings'.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    Data set information:
        Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell
        through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-
        consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further
        information, such as weather patterns and location (hence food availability) may be required to solve the
        problem. From the original data examples with missing values were removed (the majority having the predicted
        value missing), and the ranges of the continuous values have been scaled for use with an ANN (by dividing by
        200).

    Attribute information:
        Given is the attribute name, attribute type, the measurement unit and a brief description. The number of rings
        is the value to predict: either as a continuous value or as a classification problem.

        Name / Data Type / Measurement Unit / Description
        -----------------------------
        Sex / nominal / -- / M, F, and I (infant)
        Length / continuous / mm / Longest shell measurement
        Diameter / continuous / mm / perpendicular to length
        Height / continuous / mm / with meat in shell
        Whole weight / continuous / grams / whole abalone
        Shucked weight / continuous / grams / weight of meat
        Viscera weight / continuous / grams / gut weight (after bleeding)
        Shell weight / continuous / grams / after being dried
        Rings / integer / -- / +1.5 gives the age in years

    Class distribution:
        Class	Examples
        -----	--------
        1	1
        2	1
        3	15
        4	57
        5	115
        6	259
        7	391
        8	568
        9	689
        10	634
        11	487
        12	267
        13	203
        14	126
        15	103
        16	67
        17	58
        18	42
        19	32
        20	26
        21	14
        22	6
        23	9
        24	2
        25	1
        26	1
        27	2
        29	1
        -----	----
        Total	4177

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """

    dataset_path = keras.utils.get_file("abalone", "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                    'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

    dataset = pd.read_csv(dataset_path, names=column_names)

    cleanup_nums = {"Sex": {"M": 1, "F": 2, "I": 3}}
    dataset = dataset.replace(cleanup_nums)

    dataset['Sex'] = dataset['Sex'].map({1: 'Male', 2: 'Female', 3: 'Infant'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    dataset = dataset[['Male', 'Female', 'Infant', 'Length', 'Diameter', 'Height', 'Whole weight',
                       'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']]

    dataset = dataset.to_numpy()
    X = dataset[:, :-1]
    y = fd(dataset[:, -1])

    smaller = (y <= 20).flatten()
    X = X[smaller, :]
    y = y[smaller, :]
    larger = (y >= 5).flatten()
    X = X[larger, :]
    y = y[larger, :]

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 4067, "Wrong number of samples."
    assert X.shape[1] == 10, "Wrong number of features."
    return X, y


def download_banknote():
    """
    Downloads the 'banknote' data set, a data set of 1372 samples with 4 features 'Variance of wavelet transformed
    image', 'Skewness of wavelet transformed image ','Curtosis of wavelet transformed image', and 'Entropy of image'.
    The labels indicate whether a banknote is fake or not.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    dataset_path = keras.utils.get_file("banknote", "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    column_names = ['Variance of wavelet transformed image', 'Skewness of wavelet transformed image ',
                    'Curtosis of wavelet transformed image', 'Entropy of image', 'Class']

    dataset = pd.read_csv(dataset_path, names=column_names)

    dataset = dataset.to_numpy()
    X = dataset[:, :-1]
    y = fd(dataset[:, -1])

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 1372, "Wrong number of samples."
    assert X.shape[1] == 4, "Wrong number of features."

    return X, y


def download_cancer():
    """Downloads the 'cancer' data set. It consists of 569 samples of 30 features, which are used to predict whether a
    tumor is benign or malignant.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
        """
    X, y = load_breast_cancer(return_X_y=True)
    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 569, "Wrong number of samples."
    assert X.shape[1] == 30, "Wrong number of features."

    return X, y


def download_digits():
    """
    Downloads the 'digits' data set, a data set of 1797 samples with 64 features and 10 classes. The goal is to determine
    the hand-written number corresponding to each sample.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 1797, "Wrong number of samples."
    assert X.shape[1] == 64, "Wrong number of features."

    return X, y


def download_htru2():
    """
    Downloads the 'htru2' data set, a data set of 17898 samples with 8 features and 2 classes. Candidates must be
    classified in to pulsar and non-pulsar classes to aid discovery.
    htru2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe
    Survey (South) [1].

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        [1] M. J. Keith et al., 'The High Time Resolution Universe Pulsar Survey - I. System Configuration and Initial
        Discoveries',2010, Monthly Notices of the Royal Astronomical Society, vol. 409, pp. 619-627.
        DOI: 10.1111/j.1365-2966.2010.17325.x
        [2] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    try:
        dataset = pd.read_csv("data_sets/HTRU_2/HTRU_2.csv", header=None, engine='python')
    except:
        dataset = pd.read_csv("../data_sets/HTRU_2/HTRU_2.csv", header=None, engine='python')
    dataset = dataset.to_numpy()
    X = dataset[:, :-1]
    y = fd(dataset[:, -1])
    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 17898, "Wrong number of samples."
    assert X.shape[1] == 8, "Wrong number of features."

    return X, y


def download_iris():
    """
    Downloads the 'iris' data set, a data set of 150 samples with 4 features 'sepal length in cm', 'sepal width in cm',
    'petal length in cm', and 'petal width in cm'. The goal is to determine to which of the three classes each sample
    belongs.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 150, "Wrong number of samples."
    assert X.shape[1] == 4, "Wrong number of features."

    return X, y


def download_madelon():
    """
        Downloads the training and validation samples of the 'madelon' data set, a binary classification data set
        totalling 2600 samples with 500 features.

        References:
            [1] Isabelle Guyon, Steve R. Gunn, Asa Ben-Hur, Gideon Dror, 2004. Result analysis of the NIPS 2003 feature
            selection challenge. In: NIPS. [Web Link].
            [2] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
            Irvine, CA: University of California, School of Information and Computer Science.

        """
    path_X_train = keras.utils.get_file("madelon_train_data", "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data")

    X_train = pd.read_csv(path_X_train, sep=" ", header=None).to_numpy()[:, :-1]

    path_y_train = keras.utils.get_file("madelon_train_labels", "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels")

    y_train = pd.read_csv(path_y_train, sep=" ", header=None).to_numpy()

    path_X_valid = keras.utils.get_file("madelon_valid_data", "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data")

    X_valid = pd.read_csv(path_X_valid, sep=" ", header=None).to_numpy()[:, :-1]

    path_y_valid = keras.utils.get_file("madelon_valid_labels", "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels")

    y_valid = pd.read_csv(path_y_valid, sep=" ", header=None).to_numpy()

    X = np.vstack((X_train, X_valid))
    y = np.vstack((y_train, y_valid))

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 2600, "Wrong number of samples."
    assert X.shape[1] == 500, "Wrong number of features."

    return X, y

def download_seeds():
    """
    Downloads the 'seeds' data set, called 'seeds', consisting of 7 input variables and 210 observations.

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.

    """
    dataset_path = keras.utils.get_file("seeds", "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt")

    dataset = pd.read_csv(dataset_path, sep='\s+', header=None, engine='python')

    dataset = dataset.to_numpy()
    X = dataset[:, :-1]
    y = fd(dataset[:, -1])

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 210, "Wrong number of samples."
    assert X.shape[1] == 7, "Wrong number of features."

    return X, y


def download_sonar():
    """
    Downloads the 'sonar' data set, which consists of 208 samples of 60 features. The goal is to classify whether an
    object is a rock or a mine.

    References:
        The data set was contributed to the benchmark collection by Terry Sejnowski, now at the Salk Institute and the
        University of California at San Deigo. The data set was developed in collaboration with R. Paul Gorman of
        Allied-Signal Aerospace Technology Center

        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.

    """
    dataset_path = keras.utils.get_file("sonar", "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

    dataset = pd.read_csv(dataset_path, header=None)
    dataset = dataset.replace(['M', 'R'], [0, 1])

    cleanup_nums = {"Sex": {"M": 1, "F": 2, "I": 3}}
    dataset = dataset.replace(cleanup_nums)

    dataset = dataset.to_numpy()
    X = dataset[:, :-1]
    y = fd(dataset[:, -1])

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 208, "Wrong number of samples."
    assert X.shape[1] == 60, "Wrong number of features."

    return X, y


def download_spam():
    """
        Downloads the 'spam' spambase data set, a binary classification data set totalling 4601 samples with 57
        features.

        References:
            Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
            Irvine, CA: University of California, School of Information and Computer Science.

        """
    dataset_path = keras.utils.get_file("spam", "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data")

    dataset = pd.read_csv(dataset_path, header=None).to_numpy()

    X = dataset[:, :-1]
    y = dataset[:, -1]

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 4601, "Wrong number of samples."
    assert X.shape[1] == 57, "Wrong number of features."

    return X, y


def create_synthetic_data(samples_per_class: int = 100, noise: float = 0.1):
    """
    Creates a 'synthetic' data set of samples_per_class samples per class with noise Gaussian noise.

    Creates a synthetic data set consisting of two classes:
        Samples of class 1 satisfy x_1^2 + 0.01 * x_2^2 + x_3^2  - 1 = 0.
        Samples of class 2 satisfy x_1^2 + x_3^2 - 1.4 = 0.

    Args:
        samples_per_class: int, Optional
            (Default is 100.)
        noise: int, Optional
            (Default is 0.1.)

    Returns:
        X: np.ndarray
        y: np.ndarray
    """
    # class 1
    x_1 = np.random.random((samples_per_class, 1)) * 0.99
    x_2 = np.random.random((samples_per_class, 1))
    x_3 = np.sqrt(np.ones((samples_per_class, 1)) - x_1 ** 2 - 0.01 * x_2 ** 2)

    X_1 = np.hstack((x_1, x_2, x_3))
    y_1 = fd(np.zeros((samples_per_class, 1)))

    assert (np.abs(
        (X_1[:, 0] ** 2 + 0.01 * X_1[:, 1] ** 2 + X_1[:, 2] ** 2 - np.ones((samples_per_class, 1)))) <= 10e-10).all()

    # class 2
    x_1 = np.random.random((samples_per_class, 1))
    x_2 = np.random.random((samples_per_class, 1))
    x_3 = np.sqrt(1.4 * np.ones((samples_per_class, 1)) - x_1 ** 2)

    X_2 = np.hstack((x_1, x_2, x_3))
    y_2 = fd(np.ones((samples_per_class, 1)))

    assert (np.abs((X_2[:, 0] ** 2 + X_2[:, 2] ** 2 - 1.4 * np.ones((samples_per_class, 1)))) <= 10e-10).all()

    X = np.vstack((X_1, X_2))
    y = np.vstack((y_1, y_2))

    # add noise
    noise_matrix = np.random.normal(0, 0.1, X.shape)
    X = X + noise_matrix

    # embedd in R10
    # X = tmp_X.dot(np.random.normal(0, 0.5, (4, 10)))
    # correlation_matrix = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    #                                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    #                                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    #                                ])
    correlation_matrix_tmp = np.random.normal(0, noise, (10, 10))
    correlation_matrix = np.triu(correlation_matrix_tmp.dot(correlation_matrix_tmp.T) + np.identity(10))[:3, :10]
    # print(correlation_matrix)
    X = X.dot(correlation_matrix)
    assert X.shape[1] == 10, "Wrong number of features."

    return X, y


def download_voice():
    """
    Downloads the lsvt 'voice' rehabilitation data set, a data set of 126 samples with 310 features. Aim: assess whether
    voice rehabilitation treatment lead to phonations considered 'acceptable' or 'unacceptable' (binary class
    classification problem).

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    try:
        X = pd.read_excel("data_sets/lsvt/LSVT_voice_rehabilitation.xlsx", sheet_name="Data").to_numpy()
        y = pd.read_excel("data_sets/lsvt/LSVT_voice_rehabilitation.xlsx", sheet_name="Binary response").to_numpy()
    except:
        X = pd.read_excel("../data_sets/lsvt/LSVT_voice_rehabilitation.xlsx", sheet_name="Data").to_numpy()
        y = pd.read_excel("../data_sets/lsvt/LSVT_voice_rehabilitation.xlsx", sheet_name="Binary response").to_numpy()

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 126, "Wrong number of samples."
    assert X.shape[1] == 310, "Wrong number of features."

    return X, y


def download_wine():
    """
    Downloads the 'wine' data set, a data set of 178 samples with 13 features. The goal is to determine the type of
    wine.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """

    wine = datasets.load_wine()
    X = wine.data
    y = wine.target

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 178, "Wrong number of samples."
    assert X.shape[1] == 13, "Wrong number of features."

    return X, y
