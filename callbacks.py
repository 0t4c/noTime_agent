import os
import numpy as np
from typing import List, Tuple, Generator, Dict, Optional

from . import features as f
from . import policies as p
import settings as s

BombsType = List[Tuple[Tuple[int, int], int]]
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.coin_counter = CoinCounter()

    self.use_pretrained = True

    if not os.path.isfile("Q_table.npy") \
       or self.train and not self.use_pretrained:
        self.logger.info("Setting up model from scratch.")
        # step 0: initialize arbitrary Q^(0)
        n_states = f.N_STATES
        n_actions = 6
        self.Q_table = np.random.rand(n_states, n_actions)
        # Problem: "make sure that Q^(0)(s_fin,a) = 0 for all terminal states s_fin"
        # didnt know to implement this in our case where the feature vector dont say
        # something about terminal ...
    else:
        print("load model")
        self.logger.info("Loading model from saved state.")
        self.Q_table = np.load("Q_table.npy")


    # choose between "greedy_policy", "epsilon_greedy_policy", "probalistic_epsilon_greedy", "softmax_policy"
    self.policy = p.Greedy() #epsilon_greedy_policy
    self.own_bomb = None  # (x, y)
    self.own_bomb_countdown = 0

    """
    softmax_parameters = {
        # parameter for softmax policy and annealing of the temperature
        "initial_rho":  0.1, # starting value
        "rho_decay_factor":  0.0,
        "rho_min":  0.1
    }
    self.policy = p.ModSoftmax(**softmax_parameters)
    """

class CoinCounterState:
    count: int
    scores: Dict[str, int]

    def __init__(self, count: int = 9, scores: Dict[str, int] = {}):
        self.count = count
        self.scores = scores

    def copy(self):
        return CoinCounterState(self.count, self.scores.copy())


class CoinCounter:
    current: CoinCounterState
    # last: CoinCounterState

    def __init__(self):
        self.current = CoinCounterState(scores={})  # why need the default?
        # self.last = CoinCounterState(scores={})

    def update(self, game_state: dict):
        if game_state["step"] == 1:  # reset counter for each new round
            self.__init__()
        # self.last = self.current.copy()
        agents = game_state["others"] + [game_state["self"]]
        for name, score, other_can_place_bomb, pos in agents:
            if name in self.current.scores:
                before_last_score = self.current.scores[name]
            else:
                before_last_score = 0
            self.current.scores[name] = score
            self.current.count -= (score - before_last_score) % 5


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # update the number of collectable coins
    self.coin_counter.update(game_state)
    # get from state the current feature
    # print("calling state_to_features from act")

    if self.own_bomb_countdown <= 0:
        if (game_state["self"][3], 3) in game_state["bombs"]:
            self.own_bomb, self.own_bomb_countdown = game_state["self"][3], 4
        else:
            self.own_bomb = None
    if self.own_bomb_countdown > 0:
        self.own_bomb_countdown -= 1

    current_features = state_to_features(game_state, self.coin_counter.current,
                                         self.own_bomb)

    # get from feature the index
    feature_index = f.get_index_from_features(current_features)

    #print("noTime", current_features)

    if self.train:
        if game_state["round"] % self.test_period == 0:
            action = self.test_policy.act(self, feature_index)
        elif game_state["round"] % self.test_period == 1:
            action = self.validation_policy.act(self, feature_index)
        else:
            action = self.train_policy.act(self, feature_index)
    else:
        action = self.policy.act(self, feature_index)

    return action


def state_to_features(game_state: dict, coin_counter: CoinCounterState,
                      own_bomb: Optional[Tuple[int, int]]) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # TODO:
    # problem: training with opponents leads to bad policy
    # todo: find reason fix it
    # possible reasons:
    # - feature is not right if there are OPPONENTS
    # - reward assignment is not good if there are opponents
    # ideas:
    # - detect substancial changes in Q-table

    """
    feature : content
    -------------------------------------------------------
        0   : int - content of the upper neighbouring tile
        1   : int - content of the bottom neighbouring tile
        2   : int - content of the right neighbouring tile
        3   : int - content of the left neighbouring tile
        4   : int - content of the current tile
        5   : int - mode content


    FEATURE 0,1,2,3 - say something about neighboring fields
    neigbouring fields:
        0 free
        1 favourable (coin/crate)
        2 not free (crate/wall/opponent/save death)
        3 potential danger


        maybe:
        6 possible death

        temporarily:
        5 opponent
        -1 bombs
        2 wall

    FEATURE 4: current field
        0 if no safe death and (can not place bomb or if placing a bomb at current tile would lead to safe death or placing a bomb dont effect a tile) -> learn him to do not drop a bomb here
        1 can place a bomb here which would affect at least 1 tile and no safe death
        2 placing a bomb would at least affect 3 tiles and no safe death
        3 (placing a bomb would at least affect 6 tiles or placing bomb would lead to a kill) and no safe death
        4 if bomb is on the tile or safe death


    FEATURE 5: mode
        0 coin selecting mode --> 1 in FEATURE 0-3 = moving towards coins; 1-3 in FEATURE 4 = ?
        1 bomb location searching/placing mode --> --> 1 in FEATURE 0-3 = moving towards better place for bombing

        # later
        2 opponent killing
    """

    others = game_state["others"]

    features = np.zeros(f.N_FEATURES)

    coins_position = game_state["coins"]

    # if there are coins the mode is: coin selecting
    # if there are no coins the mode is: searching for a better bombing position
    features[f.MODE] = 0 if len(coins_position) > 0 else 1
    if coin_counter.count == 0 and len(others) > 0:
        features[f.MODE] = 2
        #print("TERMINATOR MODE")

    own_x, own_y, field_map, bombs, can_place_bomb = state_variables(game_state)

    # assign the bomb temporalily to -1
    for pos, countdown in bombs:
        x, y = pos
        field_map[x, y] = -1


    # Is KAMIKAZE reasonable ?

    # IDEAS: implement that he goes into safe death of own bomb

    # kamikaze feature [2,2,2,2,2,2]

    own_name, own_score = game_state["self"][:2]
    own_bomb_countdown = ([countdown for pos, countdown in bombs
                           if pos == own_bomb]+[42])[0]
    score_available_for_opponent = coin_counter.count + 5 * (len(others)-1)
    best_opponent_score_living = max((score
                                      for name, score, can_place_bomb, pos
                                      in others), default=0)
    best_opponent_score = max((score
                               for name, score in coin_counter.scores.items()
                               if name != own_name), default=0)

    # if there are no coins and we would win the round we will commit suicide to
    # ensure that the others dont get points by killing us
    # caution: must ensure to not be killed by an opponent’s bomb
    if coin_counter.count == 0:
        if own_score > score_available_for_opponent + best_opponent_score_living \
           and own_score > best_opponent_score:
            own_danger_zone = list(bomb_spread(own_x, own_y))
            if can_place_bomb:
                if all(pos not in own_danger_zone
                       for pos, _ in bombs if pos != own_bomb):
                    print("Kamikaze for the win!")
                    return np.array([2]*f.N_FEATURES)
            else:
                min_countdown = min((countdown for pos, countdown in bombs
                                     if pos in own_danger_zone
                                     and pos != own_bomb), default=42)
                if own_bomb_countdown < min_countdown \
                   and (own_bomb in own_danger_zone
                        or own_bomb == (own_x, own_y)):
                    print("Kamikaze for the win!")
                    return np.array([2]*f.N_FEATURES)



    neighbouring_fields = [(own_x + x, own_y + y) for x, y in DIRECTIONS]

    features[f.CONTENT_FIRST_NEIGHBOUR:f.CONTENT_FIRST_NEIGHBOUR+4] = np.array([
        field_map[x, y] for x, y in neighbouring_fields
    ])

    current_dist = min_coin_distance(own_x,own_y,coins_position)

    for i, pos in enumerate(neighbouring_fields):
        x, y = pos
        if features[f.CONTENT_FIRST_NEIGHBOUR+i] != 0:
            continue
        elif safe_death(x, y, bombs, field_map):
            features[f.CONTENT_FIRST_NEIGHBOUR+i] = 4
        elif features[f.MODE] == 0:
            if min_coin_distance(x, y, coins_position) < current_dist:
                features[f.CONTENT_FIRST_NEIGHBOUR+i] = 1

    neighbouring_field_score = np.zeros((5,))
    neighbouring_field_score[f.CURRENT_FIELD] = attackable_crates_oppenents(
        own_x, own_y, field_map, bombs, can_place_bomb)

    # only if the agent is in bombing placement mode
    if features[f.MODE] == 1:
        for i, pos in enumerate(neighbouring_fields):
            x, y = pos
            if features[f.CONTENT_FIRST_NEIGHBOUR+i] == 0:
                neighbouring_field_score[i] = attackable_crates_oppenents(x, y, field_map, bombs, can_place_bomb)
        if np.max(neighbouring_field_score) > 0:
            m = np.max(neighbouring_field_score)
            while np.max(neighbouring_field_score) > 0 and m > 0 and \
                1 not in features[f.CONTENT_FIRST_NEIGHBOUR:f.CONTENT_FIRST_NEIGHBOUR+5]:
                if neighbouring_field_score[f.CURRENT_FIELD] == m:
                    if not bomb_safe_death(own_x, own_y, bombs, field_map, 4):
                        features[f.CURRENT_FIELD] = 1
                        break
                for i, score in enumerate(neighbouring_field_score[:4]):
                    if score == m:
                        (x, y) = neighbouring_fields[i]
                        if not bomb_safe_death(x, y, bombs, field_map, 5):
                            features[f.CONTENT_FIRST_NEIGHBOUR+i] = 1
                m -= 1

        min_crate_distances = np.ones((5,)) * 42
        min_crate_distances[f.CURRENT_FIELD] = min_crate_distance(own_x, own_y, field_map)
        # if the neighbouring fields and the current fields dont indicate a better pos for bombing
        if not 1 in features[f.CONTENT_FIRST_NEIGHBOUR:f.CONTENT_FIRST_NEIGHBOUR+5]:
            for i, pos in enumerate(neighbouring_fields):
                x, y = pos
                if features[f.CONTENT_FIRST_NEIGHBOUR+i] == 0:
                    min_crate_distances[i] = min_crate_distance(x, y, field_map)
            if not all(min_crate_distances == 42) and np.min(min_crate_distances) != min_crate_distances[f.CURRENT_FIELD]:
                features[f.CONTENT_FIRST_NEIGHBOUR:f.CONTENT_FIRST_NEIGHBOUR+5][min_crate_distances == np.min(min_crate_distances)] = 1

    elif features[f.MODE] == 2:
        min_opponent_distances = np.ones((5,)) * 42
        min_opponent_distances[f.CURRENT_FIELD] = \
            min_opponent_distance(own_x, own_y, others)
        for i, pos in enumerate(neighbouring_fields):
            x, y = pos
            if features[f.CONTENT_FIRST_NEIGHBOUR+i] == 0:
                min_opponent_distances[f.CONTENT_FIRST_NEIGHBOUR+i] = \
                    min_opponent_distance(x, y, others)
            if not all(min_opponent_distances == 42) and \
               np.min(min_opponent_distances) != min_opponent_distances[f.CURRENT_FIELD]:
                features[f.CONTENT_FIRST_NEIGHBOUR:f.CONTENT_FIRST_NEIGHBOUR+5][min_opponent_distances == np.min(min_opponent_distances)] = 1
            # TODO: only attack opponent if they are not subjected to safe
            # death by another opponent

    # FEATURE 4
    if field_map[own_x, own_y] == -1 or safe_death(own_x, own_y, bombs, field_map):
        features[f.CURRENT_FIELD] = 4
    else:
        features[f.CURRENT_FIELD] = field_strategy(
            own_x, own_y, neighbouring_field_score[f.CURRENT_FIELD])
    # if placing the bomb would lead to safe death (e.g placing bomb in the corner)
    if features[f.CURRENT_FIELD] in [1, 2, 3]:
        # attention safe_death
        if bomb_safe_death(own_x, own_y, bombs, field_map, lookout_steps = 4):
            features[f.CURRENT_FIELD] = 0


    # Correct FEATURE 0,1,2,3: set all bombs to value 2 (wall)
    features[features == -1] = 2
    # opponents/crates/save death fields are equivalent to walls
    features[:4][(features > 2)[:4]] = 2


    # features for fighting against opponents
    # Check whether planting a bomb on current field would lead to save death of opponent (resulted by our bomb)
    if features[f.CURRENT_FIELD] in [1, 2]:
        for name, score, other_can_place_bomb, pos in game_state["others"]:
            x, y = pos
            temp_field_map = field_map.copy()
            temp_field_map[x, y] = 0
            temp_field_map[own_x, own_y] = -1
            if safe_death(x, y, [((own_x, own_y),4)], temp_field_map,
                          lookout_steps = 4):
                features[f.CURRENT_FIELD] = 3

    # check whether an opponent can block the last way out of safe death
    for name, score, other_can_place_bomb, pos in game_state["others"]:
        x, y = pos
        temp_field_map = field_map.copy()
        # consider movement of the opponent
        for delta_x, delta_y in DIRECTIONS:
            x_prime, y_prime = x + delta_x, y + delta_y
            temp_field_map[x, y] = 0
            if temp_field_map[x_prime, y_prime] == 0:
                temp_field_map[x_prime, y_prime] = 5
                for i, own_pos_prime in enumerate(neighbouring_fields):
                    if features[i] not in [0, 1]:
                        continue
                    own_x_prime, own_y_prime = own_pos_prime
                    if safe_death(own_x_prime, own_y_prime, bombs, temp_field_map):
                        print("possible danger")
                        features[i] = 3
                temp_field_map[x_prime, y_prime] = 0
            temp_field_map[x, y] = 5

    return features

# further functions

def state_variables(game_state: dict) -> Tuple[int, int, np.array,
                                               BombsType, bool]:
    own_x, own_y = game_state["self"][3]
    field_map = game_state["field"].copy()
    bombs = game_state["bombs"]
    can_place_bomb = game_state["self"][2]

    field_map[field_map == 1] = 3  # crates
    field_map[field_map == -1] = 2  # walls
    for name, score, other_can_place_bomb, pos in game_state["others"]:
        x, y = pos
        field_map[x, y] = 5

    return own_x, own_y, field_map, bombs, can_place_bomb

def attackable_crates_oppenents(own_x: int, own_y: int, field_map: np.array,
    bombs: List[Tuple[Tuple[int, int], int]], can_place_bomb: bool,
    count_crates: bool = True, count_opponents: bool = True) -> int:
    # TODO: TAKE INTO ACCOUNT THAT IN THE NEXT STEP HE CAN ACTUALLY DROP A BOMB IF THE OWN BOB EXPLODES IN THE NEXT STEP
    if not can_place_bomb:
        return 0
    # TODO: TAKE INTO ACCOUNT THAT THE TARGETED CRATES/OPPONENTS MIGHT BE DESTROYED BY ANOTHER BOMB
    targets = []
    if count_crates:
        targets.append(3)
    if count_opponents:
        targets.append(5)
    return sum(field_map[x, y] in targets
        for x, y in bomb_spread(own_x, own_y))

def field_strategy(own_x: int, own_y: int, num_attackable: int) -> int:  #give the feature 4
    if num_attackable > 0:
        if num_attackable < 3:
            return 1
        elif num_attackable < 6:
            return 2
        else:
            return 3
    return 0

def bomb_spread(bomb_x: int, bomb_y: int) -> Generator[Tuple[int, int], None, None]:
    for x, y in DIRECTIONS:
        for i in range(1,4):
            x_prime, y_prime = bomb_x+x*i, bomb_y+y*i
            if x_prime % 2 ==0 and y_prime % 2 == 0 or not in_arena(x_prime, y_prime):
                break
            yield x_prime, y_prime

# check whether the coordinate (x,y) is in the arena
def in_arena(x: int, y: int) -> bool:
    return x < s.COLS-1 and y < s.ROWS-1 and x > 0 and y > 0

def get_explode_in(value: int, countdown: int) -> bool:
    return (value // 2**countdown) % 2 == 1

def set_explode_in(value: int, countdown: int) -> int:
    return value + (0 if get_explode_in(value, countdown) % 2 else 2**countdown)

def safe_death(x: int, y: int, bombs: List[Tuple[Tuple[int, int], int]],
               field_map: np.array, lookout_steps: int = 3) -> bool:
    explosion_map = np.zeros_like(field_map)

    for pos, countdown in bombs:
        bomb_x, bomb_y = pos
        for explode_x, explode_y in bomb_spread(bomb_x, bomb_y):
            explosion_map[explode_x, explode_y] = \
                set_explode_in(explosion_map[explode_x, explode_y], countdown)

    start_points = set(((x, y),))
    for step in range(lookout_steps + 1):
        possible_escapes = []
        for x_candidate, y_candidate in start_points:
            if not get_explode_in(explosion_map[x_candidate, y_candidate], step):
                possible_escapes.append((x_candidate, y_candidate))
        start_points = set()
        for x_candidate, y_candidate in possible_escapes:
            for delta_x, delta_y in DIRECTIONS + [(0, 0)]:
                x_prime, y_prime = x_candidate + delta_x, y_candidate + delta_y
                if in_arena(x_prime, y_prime) and field_map[x_prime, y_prime] in [0, 1]:
                    start_points.add((x_prime, y_prime))
        if len(start_points) == 0:
            return True
    return False

def bomb_safe_death(x: int, y: int, bombs: List[Tuple[Tuple[int, int], int]],
                    field_map: np.array, lookout_steps: int) -> bool:
    """
    returns true if placing a bomb on tile (x,y) would lead to safe death
    """
    temp_field_map = field_map.copy()
    temp_field_map[x, y] = -1
    hypothetical_bombs = bombs + [((x, y), lookout_steps)]
    return safe_death(x, y, hypothetical_bombs, temp_field_map, lookout_steps)

def min_crate_distance(own_x: int, own_y: int, field_map: np.array) -> int:
    # computes the distance (number of steps to get there) to the next nearest
    # coin from the position (own_x, own_y)
    crates = np.argwhere(field_map == 3) # get crate positions
    if np.sum(field_map == 3) == 0:
        return 30
    return min(
        abs(crate_x - own_x) + abs(crate_y - own_y)
        + (2 if (crate_x == own_x and crate_x % 2 == 0 and crate_y != own_y
        or crate_y == own_y and crate_y % 2 == 0 and crate_x != own_x) else 0)
        for crate_x, crate_y in crates)

def min_coin_distance(own_x: int, own_y: int, coins: List[Tuple[int]]) -> int:
    # computes the distance (number of steps to get there) to the next nearest
    # coin from the position (own_x, own_y)
    if len(coins) == 0:
        return 30
    return min(
        abs(coin_x - own_x) + abs(coin_y - own_y)
        + (2 if (coin_x == own_x and coin_x % 2 == 0 and coin_y != own_y
           or coin_y == own_y and coin_y % 2 == 0 and coin_x != own_x) else 0)
        for coin_x, coin_y in coins)

def min_opponent_distance(own_x: int, own_y: int, opponents: List[Tuple[str,
                          int, bool, Tuple[int, int]]]) -> int:
     # computes the distance (number of steps to get there) to the next nearest
     # coin from the position (own_x, own_y)

     opponent_positions = [opponent[3] for opponent in opponents]
     return min_coin_distance(own_x, own_y, coins=opponent_positions)
