from sklearn import datasets
import numpy as np
import pandas as pd
from src.auxiliary_functions.auxiliary_functions import fd


def fetch_data_set(name: str):
    """Loads the data sets.

    Args:
        name: str
            Name of the data set.
            data set name:  # samples                       /   # features  /   # classes
            - 'bank':       1372                            /   4           /   2
            - 'credit':     30000                           /   22          /   2
            - 'digits':     1797                            /   64          /   10
            - 'htru':       17898                           /   8           /   2
            - 'seeds':      210                             /   7           /   3
            - 'sepsis':     110204                          /   3           /   2
            - 'skin':       245057                          /   3           /   2
            - 'sonar':      208                             /   60          /   2
            - 'spam':       4601                            /   57          /   2
            - 'synthetic':  2000000                         /   3           /   2

    Returns:
        X: np.ndarray
            Data.
        y: np.ndarray
            Labels.
    """
    if name == 'bank':
        X, y = download_bank()
    elif name == 'credit':
        X, y = download_credit()
    elif name == 'digits':
        X, y = download_digits()
    elif name == 'htru':
        X, y = download_htru()
    elif name == 'sepsis':
        X, y = download_sepsis()
    elif name == 'skin':
        X, y = download_skin()
    elif name == 'sonar':
        X, y = download_sonar()
    elif name == 'spam':
        X, y = download_spam()
    elif name == 'synthetic':
        X, y = create_synthetic_data()
    elif name == 'seeds':
        X, y = download_seeds()
    else:
        X, y = None, None
        print("No valid data set was selected.")
    return fd(X), fd(y)


def download_bank():
    """Downloads the 'bank' data set, a data set of 1372 samples with 4 features.
    The labels indicate whether a banknote is fake or not.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        [1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    column_names = ['Variance of wavelet transformed image', 'Skewness of wavelet transformed image ',
                    'Curtosis of wavelet transformed image', 'Entropy of image', 'Class']
    dataset = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        names=column_names).to_numpy()

    X = dataset[:, :-1]
    y = fd(dataset[:, -1])

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 1372, "Wrong number of samples."
    assert X.shape[1] == 4, "Wrong number of features."

    return X, y


def download_credit():
    """Downloads the default of 'credit' card data set, a data set of 30000 samples with 22 features. Aim: determine
    whether customers are going to default.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        [1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
        [2] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of
        probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
    """
    dataset = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        header=None).to_numpy()

    dataset = dataset[2:, 2:].astype(float)
    X = dataset[:, :-1]
    y = dataset[:, -1]

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 30000, "Wrong number of samples."
    assert X.shape[1] == 22, "Wrong number of features."

    return X, y


def download_digits():
    """Downloads the 'digits' data set, a data set of 1797 samples with 64 features and 10 classes. The goal is to
    determine the hand-written number corresponding to each sample.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        [1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 1797, "Wrong number of samples."
    assert X.shape[1] == 64, "Wrong number of features."

    return X, y


def download_htru():
    """Downloads the 'htru' data set, a data set of 17898 samples with 8 features and 2 classes. Candidates must be
    classified into pulsar and non-pulsar classes to aid discovery.

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
        dataset = pd.read_csv("data_sets/HTRU_2/HTRU_2.csv", header=None, engine='python').to_numpy()
    except:
        dataset = pd.read_csv("../data_sets/HTRU_2/HTRU_2.csv", header=None, engine='python').to_numpy()
    X = dataset[:, :-1]
    y = fd(dataset[:, -1])
    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 17898, "Wrong number of samples."
    assert X.shape[1] == 8, "Wrong number of features."

    return X, y


def download_iris():
    """Downloads the 'iris' data set, a data set of 150 samples with 4 features and 3 classes.

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        [1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 150, "Wrong number of samples."
    assert X.shape[1] == 4, "Wrong number of features."


def download_seeds():
    """Downloads the 'seeds' data set, consisting of 7 input variables, 3 classes, and 210 observations.

    References:
        [1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
                          sep='\s+', header=None, engine='python').to_numpy()
    X = dataset[:, :-1]
    y = fd(dataset[:, -1])

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 210, "Wrong number of samples."
    assert X.shape[1] == 7, "Wrong number of features."

    return X, y


def download_sepsis():
    """Downloads the primary cohort from the 'sepsis' dataset from Norway: 3 features for 110,204 patient admissions

    Returns:
        X: np.array
            Data.
        y: np.array
            Labels.

    References:
        [1] Davide Chicco, Giuseppe Jurman, â€œSurvival prediction of patients with sepsis from age, sex, and septic
        episode number aloneâ€. Scientific Reports 10, 17156 (2020).
        [2] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    try:
        dataset = pd.read_csv("data_sets/sepsis/1.csv", engine='python')
    except:
        dataset = pd.read_csv("../data_sets/sepsis/1.csv", engine='python')
    dataset = dataset.to_numpy().astype(float)

    X = dataset[:, :-1]
    y = fd(dataset[:, -1])
    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 110204, "Wrong number of samples."
    assert X.shape[1] == 3, "Wrong number of features."

    return X, y


def download_skin():
    """Downloads the 'skin' data set, a binary classification data set totalling 245057 samples with 3
    features.

    References:
        [1] Rajen Bhatt, Abhinav Dhall, 'Skin Segmentation Dataset', UCI Machine Learning Repository
        Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    dataset = pd.read_csv("data_sets/skin/Skin_NonSkin.txt", sep='\s+', header=None).to_numpy()

    X = dataset[:, :-1]
    y = dataset[:, -1]
    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 245057, "Wrong number of samples."
    assert X.shape[1] == 3, "Wrong number of features."

    return X, y


def download_sonar():
    """Downloads the 'sonar' data set, which consists of 208 samples of 60 features. The goal is to classify whether an
    object is a rock or a mine.

    References:
        [1] The data set was contributed to the benchmark collection by Terry Sejnowski, now at the Salk Institute and
        the University of California at San Deigo. The data set was developed in collaboration with R. Paul Gorman of
        Allied-Signal Aerospace Technology Center
        [2] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    dataset = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        header=None)
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
    """Downloads the 'spam' spambase data set, a binary classification data set totalling 4601 samples with 57 features.

    References:
        [1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
        Irvine, CA: University of California, School of Information and Computer Science.
    """
    dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
                          header=None).to_numpy()

    X = dataset[:, :-1]
    y = dataset[:, -1]

    assert X.shape[0] == y.shape[0], "Number of data points does not coincide with the number of labels."
    assert X.shape[0] == 4601, "Wrong number of samples."
    assert X.shape[1] == 57, "Wrong number of features."

    return X, y


def create_synthetic_data(samples_per_class: int = 1000000, noise: float = 0.05):
    """Creates a 'synthetic' data set of samples_per_class samples per class with Gaussian noise.

    Creates a synthetic data set consisting of two classes:
        Samples of class 1 satisfy x_1^2 + 0.01 * x_2 + x_3**2 - 1 = 0.
        Samples of class 2 satisfy x_1^2 + x_3^2 - 1.3 = 0.

    Args:
        samples_per_class: int, Optional
            (Default is 1000000.)
        noise: int, Optional
            (Default is 0.05.)

    Returns:
        X: np.ndarray
        y: np.ndarray
    """
    # class 1
    x_1 = np.random.random((samples_per_class, 1)) * 0.99
    x_2 = np.random.random((samples_per_class, 1))
    x_3 = np.sqrt(np.ones((samples_per_class, 1)) - x_1 ** 2 - 0.01 * x_2)

    X_1 = np.hstack((x_1, x_2, x_3))
    y_1 = fd(np.zeros((samples_per_class, 1)))

    # class 2
    x_1 = np.random.random((samples_per_class, 1))
    x_2 = np.random.random((samples_per_class, 1))
    x_3 = np.sqrt(1.3 * np.ones((samples_per_class, 1)) - x_1 ** 2)

    X_2 = np.hstack((x_1, x_2, x_3))
    y_2 = fd(np.ones((samples_per_class, 1)))

    X = np.vstack((X_1, X_2))
    y = np.vstack((y_1, y_2))

    # add noise
    noise_matrix = np.random.normal(0, noise, X.shape)
    X = X + noise_matrix
    return X, y
