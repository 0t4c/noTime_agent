import unittest
import numpy as np
from itertools import permutations
from typing import Iterable, Generator

from agent_code.noTime_agent.callbacks import get_explode_in
import agent_code.noTime_agent.features as f

def self_cross_product(original: Iterable, times: int=2) -> Generator[np.array, None, None]:
    m = len(original)
    for i in range(m**times):
        yield np.array(tuple(original[i//m**j % m] for j in range(times)))

class FeatureIndexMappingTestCase(unittest.TestCase):
    num_states = f.N_STATES
    # test whether the get_index_from_feature function is the inverse of the get_features_from_index function
    def test_equal(self):
        for i in range(self.num_states):
            features = f.get_features_from_index(i)
            i_prime = f.get_index_from_features(features)
            self.assertEqual(i, i_prime)
    # test whether the get_features_from_index function is surjective
    def test_surjective(self):
        features = [f.get_features_from_index(i) for i in range(self.num_states)]
        self.assertEqual(np.unique(features, axis=1).shape[0], self.num_states)

    # test the explode mapping Functions
    def test_explosion(self):
        for summands in self_cross_product([0, 1], 4):
            value = sum(2**i for i, s in enumerate(summands) if s)
            for i, s in enumerate(summands):
                self.assertEqual(s, get_explode_in(value, i))

if __name__ == '__main__':
    unittest.main()
