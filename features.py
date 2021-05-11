"""
feature : content
-------------------------------------------------------
    0   : int - content of the upper neighbouring tile
    1   : int - content of the bottom neighbouring tile
    2   : int - content of the right neighbouring tile
    3   : int - content of the left neighbouring tile
    4   : int - content of the current tile
    5   : int - mode content
"""

import numpy as np

CONTENT_FIRST_NEIGHBOUR = 0
CONTENT_UP, CONTENT_RIGHT, CONTENT_DOWN, CONTENT_LEFT, = 0, 1, 2, 3
CURRENT_FIELD = 4
MODE = 5

N_FEATURE_VALUES = [*[4]*4, 5, 3]
N_FEATURES = len(N_FEATURE_VALUES)
N_STATES = np.prod(N_FEATURE_VALUES)

def get_index_from_features(features: np.array) -> int:
    """
        Convert the feature vector to a single integer.
        note: feature = [u,r,d,l,c,m] , Q_values = [Q_up, Q_right, Q_down, Q_left, Q_stay, Q_bomb]
        aim feature: [u,r,d,l,c,m] -> index: n = l + 5*u + 5*5*d + 5*5*5*r + 5*5*5*5*c + 5*5*5*5*5*m
    """
    return int(sum(features[i]*np.prod([1] + N_FEATURE_VALUES[:i])
                   for i in range(N_FEATURES)))

def get_features_from_index(n: int) -> np.array:
    """
        Converts an integer to the feature vector
        note: feature = [u,r,d,l,c,m] , Q_values = [Q_up, Q_right, Q_down, Q_left, Q_stay, Q_bomb]
        aim n -> [u,r,d,l,c,m] = n % 5, (n//5) % 5, (n//5**2) % 5, (n//5**3) % 5, (n//5**4) % 5, (n//5**4)% 2
    """
    return np.array(list(
        (n//np.prod([1] + N_FEATURE_VALUES[:i])) % N_FEATURE_VALUES[i]
        for i in range(N_FEATURES)))
