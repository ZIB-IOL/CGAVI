import unittest
import numpy as np
from global_ import gpu_memory_
from src.data_sets.data_set_creation import fetch_data_set
from src.gpu.memory_allocation import set_gpu_memory


class TestDataSetLoading(unittest.TestCase):

    def setUp(self):
        set_gpu_memory(gpu_memory_)

    def test_load_data_set(self):
        """Tests whether fetch_data_set() behaves as intended."""
        names = ['abalone', 'banknote', 'cancer', 'digits', 'htru2', 'iris', 'madelon', 'seeds', 'sonar',
                 'spam', 'synthetic', 'voice', 'wine']
        for name in names:
            X, y = fetch_data_set(name)
            self.assertTrue(isinstance(X, np.ndarray)), "Data set should be an array."
            self.assertTrue(isinstance(y, np.ndarray)), "Labels should be an array."
