import unittest

from global_ import gpu_memory_


from src.auxiliary_functions.auxiliary_functions import HiddenPrints
from src.experiment_setups.experiment_setups import ExperimentSetups
from src.gpu.memory_allocation import set_gpu_memory
import cupy as cp
import numpy as np


class TestExperimentSetups(unittest.TestCase):

    def setUp(self):
        set_gpu_memory(gpu_memory_)

    def test_evaluate(self):
        """Tests whether ExperimentSetups() behaves as intended."""

        Cs = [1, 1.5]
        hyperparameters_vca = {"algorithm": "vca", "psi": [0.5], "C": Cs}
        hyperparameters_avi = {"algorithm": "avi", "psi": [0.5], "tau": [0], "C": Cs}
        hyperparameters_l1_cgavi = {"algorithm": "l1_cgavi", "psi": [0.25], "eps_factor": [0.7], "lmbda": [5],
                                       "tau": [50], "C": Cs, "tol": 0.01}
        hyperparameters_l2_cgavi = {"algorithm": "l2_cgavi", "psi": [0.25], "eps_factor": [0.7], "lmbda": [5],
                                       "tau": [50], "C": Cs, "tol": 0.01}
        hyperparameters_agdavi = {"algorithm": "agdavi", "psi": [0.25], "lmbda": [5],
                                    "tau": [50], "C": Cs, "tol": 0.01}
        hyperparameters_svm = {"algorithm": "svm", "C": Cs, "degree": [5]}
        cp.random.seed(19)
        np.random.seed(19)
        for name in ['banknote']:
            for hyperparameters in [hyperparameters_l1_cgavi, hyperparameters_l2_cgavi, hyperparameters_agdavi, hyperparameters_avi,
                                    hyperparameters_vca, hyperparameters_svm]:
                for i in range(0, 1):
                    pc = ExperimentSetups(name, hyperparameters)
                    with HiddenPrints():
                        pc.cross_validation(saving=False)
