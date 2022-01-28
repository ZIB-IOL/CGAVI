import os
import pandas as pd


def save_data_frame(data_frame: pd.DataFrame, algorithm_name: str, dataset_name: str, experiment: str = "performance",
                    test: bool = False, number: int = -1):
    """
    Saves the data_frame in the data_frames directory.

    Args:
        data_frame: data_frame
        algorithm_name: str
        dataset_name: str
        experiment: str, Optional
            (Default is "performance".)
        test: bool, Optional
            (Default is False.)
        number: int, Optional
            Number of run.
    """
    directory = "data_frames"
    if not os.path.exists(os.path.join("./", directory)):
        os.makedirs(os.path.join("./", directory))
    directory = directory + "/" + experiment
    if not os.path.exists(os.path.join("./", directory)):
        os.makedirs(os.path.join("./", directory))
    directory = directory + "/" + dataset_name
    if not os.path.exists(os.path.join("./", directory)):
        os.makedirs(os.path.join("./", directory))
    directory = directory + "/" + algorithm_name
    if not os.path.exists(os.path.join("./", directory)):
        os.makedirs(os.path.join("./", directory))
    if test:
        directory = directory + "/" + "test"
        if not os.path.exists(os.path.join("./", directory)):
            os.makedirs(os.path.join("./", directory))
    else:
        directory = directory + "/" + "train"
        if not os.path.exists(os.path.join("./", directory)):
            os.makedirs(os.path.join("./", directory))
    if number == -1:
        i = 1
        filename = directory + "/" + "run--{}.pk1".format(i)
        while os.path.exists(filename):
            i += 1
            filename = directory + "/" + "run--{}.pk1".format(i)
    else:
        filename = directory + "/" + "run--{}.pk1".format(number)
    data_frame.to_pickle(filename, protocol=3)


def load_data_frame(algorithm_name: str, dataset_name: str, experiment: str = "performance",
                    test: bool = False, number: int = 1):
    """
    Loads the data_frame.

    Args:
        algorithm_name: str
        dataset_name: str
        experiment: str, Optional
            (Default is "performance".)
        test: bool, Optional
            (Default is False.)
        number: int, Optional
            (Default is 1.)

    Returns:
        data_frame: data_frame
    """
    directory = "data_frames" + "/" + experiment + "/" + dataset_name + "/" + algorithm_name
    if test:
        directory = directory + "/" + "test"

    else:
        directory = directory + "/" + "train"
    if not os.path.exists(os.path.join("./", directory)):
        print("Error. Directory doesn't exist.")

    filename = directory + "/" + "run--{}.pk1".format(number)
    if not os.path.exists(os.path.join("./", filename)):
        print("Error. File doesn't exist.")
    return pd.read_pickle(filename)
