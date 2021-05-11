import random
import numpy as np

from . import features as f
from . import callbacks as c
from agents import Agent

class Policy:
    parameters: dict

    def __init__(self, name: str):
        self.parameters = {"name": name}

    def act(self, agent: Agent, feature_index: int) -> str:
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def print_current_parameter(self):
        raise NotImplementedError()

# ------------------------------Greedy-policy----------------------------------
class Greedy(Policy):
    def __init__(self):
        super().__init__("greedy")

    def act(self, agent: Agent, feature_index: int) -> str:
        agent.logger.debug("Querying model for action.")
        # get the Q values via the index and get the action with the highest Q value
        action_index = np.argmax(agent.Q_table[feature_index, :])  # Pi(s) = argmax_a Q(s,a)
        # ARGMAX POLICY
        chosen_action = c.ACTIONS[action_index]
        return chosen_action

    def update(self):
        pass

    def print_current_parameter(self):
        pass


# ------------------------Epsilon-greedy-policy-----------------------------
class EpsilonGreedy(Policy):
    def __init__(self, initial_random_prob: float, random_prob_min: float,
                 random_prob_decay_factor: float):
        #
        super().__init__("epsilon_greedy")
        self.parameters["initial_random_prob"] = initial_random_prob
        self.parameters["random_prob"] = initial_random_prob
        self.parameters["random_prob_min"] = random_prob_min
        self.parameters["random_prob_decay_factor"] = random_prob_decay_factor

    def act(self, agent: Agent, feature_index: int) -> str:
        # With a probability of self.random_prob do a random action (random exploitation)
        if random.random() < self.parameters["random_prob"]:
            #self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(c.ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        # With a probability 1 - self.random_prob do the following:
        # trained action
        agent.logger.debug("Querying model for action.")
        # get the Q values via the index and get the action with the highest Q value
        action_index = np.argmax(agent.Q_table[feature_index, :])  # Pi(s) = argmax_a Q(s,a)
        # ARGMAX POLICY
        chosen_action = c.ACTIONS[action_index]
        return chosen_action

    def update(self):
        if self.parameters["random_prob"] > self.parameters["random_prob_min"]:
            self.parameters["random_prob"] *= 1-self.parameters["random_prob_decay_factor"]

    def print_current_parameter(self):
        print("{:>12}: {:6.2f}".format("epsilon", self.parameters["random_prob"]))

"""
#--------------------Epsilon-Probalistic-policy-----------------------------
def probalistic_epsilon_greedy(self, feature_index: int)->str:
        # OUR OWN PROBALISTIC POLICY (TO AVOID LOOPING)
        Q_values = self.Q_table[feature_index, :]
        if any(Q_values > 0):
            Q_value_probabilities = Q_values * (Q_values > 0)
        else:
            Q_value_probabilities = Q_values - np.min(Q_values) * 1.1
        Q_value_probabilities /= sum(Q_value_probabilities)
        Q_value_probabilities *= Q_value_probabilities**5
        Q_value_probabilities /= sum(Q_value_probabilities)
        return np.random.choice(c.ACTIONS, p = Q_value_probabilities)
"""

#-------------------------Softmax-policy------------------------------------
class Softmax(Policy):
    def __init__(self, initial_rho: float, rho_min: float,
                 rho_decay_factor: float):
        super().__init__("softmax")
        self.parameters["initial_rho"] = initial_rho
        self.parameters["rho"] = initial_rho
        self.parameters["rho_min"] = rho_min
        self.parameters["rho_decay_factor"] = rho_decay_factor

    def act(self, agent: Agent, feature_index: int) -> str:
        softmax_probabilities = np.exp(agent.Q_table[feature_index, :]
                                       / self.parameters["rho"])
        eps = 1e-12
        if np.sum(softmax_probabilities) < eps:  # avoid NaN ValueError
            for x in range(20):
                tmp_rho = self.parameters["rho"] * np.exp(x)
                p = np.exp(agent.Q_table[feature_index, :] / tmp_rho)
                if np.sum(p) > eps:
                    softmax_probabilities = p
                    print(f"used rho = {tmp_rho} to avoid zero division")
                    break
            else:
                softmax_probabilities = np.exp(agent.Q_table[feature_index, :])
                print("used 1 instead of bad rho", self.parameters["rho"])
            print(softmax_probabilities)
        sp_is_inf = np.isinf(softmax_probabilities)
        if any(sp_is_inf):
            print(softmax_probabilities)
            softmax_probabilities[sp_is_inf] = \
                np.max([np.sum(softmax_probabilities[~sp_is_inf]), 1])
        softmax_probabilities /= np.sum(softmax_probabilities)
        chosen_action = np.random.choice(c.ACTIONS, p=softmax_probabilities)
        return chosen_action

    def update(self):
        if self.parameters["rho"] >self. parameters["rho_min"]:
            self.parameters["rho"] *= 1-self.parameters["rho_decay_factor"]

    def print_current_parameter(self):
        print("{:>12}: {:6.2f}".format("rho", self.parameters["rho"]))

#------------------------Modified-Softmax-policy--------------------------------
class ModSoftmax(Softmax):
    def __init__(self, initial_rho: float, rho_min: float,
                 rho_decay_factor: float):
        super().__init__(initial_rho, rho_min, rho_decay_factor)
        self.parameters["name"] = "modified softmax"

    def act(self, agent: Agent, feature_index: int) -> str:
        features = f.get_features_from_index(feature_index)
        softmax_probabilities = np.exp(agent.Q_table[feature_index, :]
                                       / self.parameters["rho"])
        mask = features[:5] == np.array([2, 2, 2, 2, 4])
        softmax_probabilities[:5][mask] = 0
        if features[4] == 4 or features[4] == 0:
            # do not place a bomb in these cases
            softmax_probabilities[5] = 0
        if np.sum(softmax_probabilities) == 0:
            s = Softmax(self.parameters["rho"], self.parameters["rho_min"],
                        self.parameters["rho_decay_factor"])
            return s.act(agent, feature_index)
        eps = 1e-15
        if np.sum(softmax_probabilities) < eps:  # avoid NaN ValueError
            for x in range(20):
                tmp_rho = self.parameters["rho"] * np.exp(x)
                p = np.exp(agent.Q_table[feature_index, :] / tmp_rho)
                if np.sum(p) > eps:
                    softmax_probabilities = p
                    print(f"used rho = {tmp_rho} to avoid zero division")
                    break
            else:
                softmax_probabilities = np.exp(agent.Q_table[feature_index, :])
                print("used 1 instead of bad rho", self.parameters["rho"])
        sp_is_inf = np.isinf(softmax_probabilities)
        if any(sp_is_inf):
            print(softmax_probabilities)
            softmax_probabilities[sp_is_inf] = \
                np.max([np.sum(softmax_probabilities[~sp_is_inf]), 1])
        softmax_probabilities /= np.sum(softmax_probabilities)
        chosen_action = np.random.choice(c.ACTIONS, p=softmax_probabilities)
        return chosen_action
