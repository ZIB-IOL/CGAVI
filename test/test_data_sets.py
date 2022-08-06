import unittest
import numpy as np
from src.data_sets.data_set_creation import fetch_data_set


class TestDataSetLoading(unittest.TestCase):

    def test_load_data_set(self):
        """Tests whether fetch_data_set() behaves as intended."""
        names = ['bank', 'credit', 'digits', 'htru', 'seeds', 'sepsis', 'skin', 'sonar', 'spam', 'synthetic']
        for name in names:

            X, y = fetch_data_set(name)
            self.assertTrue(isinstance(X, np.ndarray)), "Data set should be an array."
            self.assertTrue(isinstance(y, np.ndarray)), "Labels should be an array."
