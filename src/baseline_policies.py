import numpy as np
from typing import List, Set, Dict, AnyStr


def get_possible_combinations(obs):
    available_numbers = set(obs[0]).difference(set([0]))
    possible_combinations = dict()
    for i in range(1, obs[1] + 1):
        if i in available_numbers:
            possible_combinations[i] = [set([i])]
        else:
            possible_combinations[i] = []
        for j in range(1, i):
            for c in possible_combinations[i - j]:
                new_combination = c.union(set([j]))
                if (
                    sum(new_combination) == i
                    and new_combination not in possible_combinations[i]
                    and new_combination.issubset(available_numbers)
                ):
                    possible_combinations[i].append(new_combination)
    return possible_combinations[obs[1]]


def choose_random(obs):
    possible_combinations = get_possible_combinations(obs)
    if len(possible_combinations) != 0:
        return set_to_action_format(
            np.random.choice(possible_combinations), len(obs[0])
        )
    else:
        return [0] * len(obs[0])


def choose_from_lexographical_ordering(obs, largest_first, index):
    possible_combinations = get_possible_combinations(obs)
    if len(possible_combinations) != 0:
        return set_to_action_format(
            sorted(
                possible_combinations,
                key=lambda x: ",".join(map(str, sorted(x, reverse=largest_first))),
                reverse=largest_first,
            )[min(index, len(possible_combinations) - 1)],
            len(obs[0]),
        )
    else:
        return [0] * len(obs[0])


def set_to_action_format(to_flip, board_size):
    return [i if i in to_flip else 0 for i in range(1, board_size + 1)]
