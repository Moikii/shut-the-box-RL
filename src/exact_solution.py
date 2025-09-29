from typing import List
from itertools import combinations
from tqdm import tqdm


expected_return = {}
expected_return[str([])] = 45

max_points = 45

prob_two_dices = [
    1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36
]


def current_points(board_state: List[int]):
    return max_points - sum(board_state)


def possible_flips(board_state: List[int], dice: int):
    combinations_list = []
    for r in range(1, len(board_state) + 1):
        for combination in combinations(board_state, r):
            if sum(combination) == dice:
                combinations_list.append(combination)
    return combinations_list


def calc_expected_return(board_state: List[int]):
    if board_state == ():
        return max_points

    returns = []
    for roll in range(2, 13):
        best_flip = current_points(board_state)
        for flip in possible_flips(board_state, roll):
            temp_board = [b for b in board_state if b not in flip]
            best_flip = max([best_flip, expected_return[str(temp_board)]])

        returns.append(best_flip)

    return sum([prob_two_dices[i] * returns[i] for i in range(11)])


for r in tqdm(range(1, 10)):
    for combination in combinations(range(1, 10), r):
        c = calc_expected_return(list(combination))
        expected_return[str(list(combination))] = c

print(expected_return[str(list(range(1, 10)))])
print(len(expected_return))
