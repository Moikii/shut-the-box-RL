import numpy as np

# from __future__ import annotations
from collections import defaultdict
from baseline_policies import (
    choose_random,
    get_possible_combinations,
    set_to_action_format,
)
from action_conversions import action_to_index, index_to_action


class KlappbrettAgent:
    def __init__(
        self,
        board_size: int,
        number_of_dice: int,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.q_values = defaultdict(
            lambda: board_size
            * (board_size + 1)
            / 2
            * np.concatenate(
                [np.zeros([2**self.board_size, 1]), np.zeros([2**self.board_size, 1])],
                axis=1,
            )
        )

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.board_size = board_size
        self.number_of_dice = number_of_dice
        self.past_actions = list()
        self.past_obs = list()

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            action = choose_random(obs)
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            action = self._choose_random_from_best_possible(obs)
        self.past_actions.append(action)
        self.past_obs.append(obs)
        self.q_values[obs][action_to_index(action), 1] += 1
        # print(f"obs get: {obs}")
        # print(f"index get: {action_to_index(action)}")
        # print(f"usage get: {self.q_values[obs][action_to_index(action), 1]}")
        return action

    def _get_possible_combinations_index(self, obs):
        possible_combinations = get_possible_combinations(obs)
        possible_combinations = [
            set_to_action_format(i, self.board_size) for i in possible_combinations
        ]

        return [action_to_index(i) for i in possible_combinations]

    def _choose_random_from_best_possible(self, obs):
        possible_combinations_index = self._get_possible_combinations_index(obs)

        if len(possible_combinations_index) == 0:
            return index_to_action(self.board_size, 0)
        else:
            maximum = max(
                [self.q_values[obs][i, 0] for i in possible_combinations_index]
            )
            index = np.random.choice(
                [
                    j
                    for j in possible_combinations_index
                    if self.q_values[obs][j, 0] == maximum
                ]
            )
            return index_to_action(self.board_size, index)

    def update(
        self,
        obs: tuple[tuple[int], tuple[int]],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[tuple[int], tuple[int]],
    ):
        index = action_to_index(action)
        possible_combinations_index = self._get_possible_combinations_index(next_obs)
        if len(possible_combinations_index) != 0:
            future_q_value = (not terminated) * max(
                [self.q_values[next_obs][i, 0] for i in possible_combinations_index]
            )
        else:
            future_q_value = self.q_values[obs][index, 0]
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[obs][index, 0]
        )

        self.q_values[obs][index, 0] = (
            self.q_values[obs][index, 0] + self.lr * temporal_difference
        )

    def update_2(
        self,
        obs: tuple[tuple[int], tuple[int]],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[tuple[int], tuple[int]],
    ):
        if terminated:
            for o, a in zip(self.past_obs, self.past_actions):
                index = action_to_index(a)
                usages = self.q_values[o][index, 1]
                # print(usages)
                # print(self.q_values[o][index, 0])
                self.q_values[o][index, 0] = (
                    float(self.q_values[o][index, 0] * max(1, usages - 1) + reward)
                    / usages
                )
                # print(self.q_values[o][index, 0])

            self.past_actions = list()
            self.past_obs = list()

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
