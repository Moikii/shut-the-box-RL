import numpy as np
from typing import List, Set, Dict, AnyStr


def choose_random(observation: Dict[str, List]) -> List[bool]:
    """
    Choose a random combination of numbers from the given observation.

    Args:
        observation (Dict[str, List): A dictionary containing lists of the available numbers in binary representation, rolled dice and possible combinations.

    Returns:
        List[bool]: A binary representation of the chosen numbers, where 1 indicates the number is chosen and 0 indicates it is not.

    Notes:
        - If the observation does not contain any possible combinations, an empty set is chosen.
        - The chosen numbers are represented as a binary list, where the index of the number corresponds to its position
            in the available numbers list.

    Example:
        >>> observation = {
        ...     "available_numbers": [1, 1, 1, 0, 1],
                "rolled_dice": [2,3]
        ...     "possible_combinations": [{5}, {2, 3}],
        ... }
        >>> choose_random(observation)
        [0, 1, 1, 0, 0]
    """
    available_numbers_len = len(observation["board_state"])
    if len(observation["possible_combinations"]) != 0:
        to_flip = np.random.choice(observation["possible_combinations"])
    else:
        to_flip = set()
    return flips_to_binary(available_numbers_len, to_flip)


def choose_largest_number(observation: Dict[str, List]) -> List[bool]:
    """
    Choose the combination of numbers with the smallest possible values from the given observation.

    Args:
        observation (Dict[str, List): A dictionary containing lists of the available numbers in binary representation, rolled dice and possible combinations.

    Returns:
        List[bool]: A binary representation of the chosen numbers, where 1 indicates the number is chosen and 0 indicates it is not.

    Notes:
        - If the observation does not contain any possible combinations, an empty set is chosen.
        - The chosen numbers are represented as a binary list, where the index of the number corresponds to its position
            in the available numbers list.

    Example:
        >>> observation = {
        ...     "available_numbers": [1, 1, 1, 0, 1],
                "rolled_dice": [2,3]
        ...     "possible_combinations": [{5}, {2, 3}],
        ... }
        >>> choose_random(observation)
        [0, 0, 0, 0, 1]
    """
    available_numbers_len = len(observation["board_state"])
    if len(observation["possible_combinations"]) != 0:
        to_flip = sorted(
            observation["possible_combinations"],
            key=lambda x: ",".join(map(str, sorted(x, reverse=True))),
            reverse=True,
        )[0]
    else:
        to_flip = set()
    return flips_to_binary(available_numbers_len, to_flip)


def choose_smallest_number(observation: Dict[str, List]) -> List[bool]:
    """
    Choose the combination of numbers with the smallest possible values from the given observation.

    Args:
        observation (Dict[str, List): A dictionary containing lists of the available numbers in binary representation, rolled dice and possible combinations.

    Returns:
        List[bool]: A binary representation of the chosen numbers, where 1 indicates the number is chosen and 0 indicates it is not.

    Notes:
        - If the observation does not contain any possible combinations, an empty set is chosen.
        - The chosen numbers are represented as a binary list, where the index of the number corresponds to its position
            in the available numbers list.

    Example:
        >>> observation = {
        ...     "available_numbers": [1, 1, 1, 0, 1],
                "rolled_dice": [2,3]
        ...     "possible_combinations": [{5}, {2, 3}],
        ... }
        >>> choose_random(observation)
        [0, 1, 1, 0, 0]
    """
    available_numbers_len = len(observation["board_state"])
    if len(observation["possible_combinations"]) != 0:
        to_flip = sorted(
            observation["possible_combinations"],
            key=lambda x: ",".join(map(str, sorted(x, reverse=False))),
            reverse=False,
        )[0]
    else:
        to_flip = set()
    return flips_to_binary(available_numbers_len, to_flip)


def flips_to_binary(n: int, to_flip: set[int]) -> List[bool]:
    """
    A function that maps a set of integers to a binary representation in a list of boolean values.

    Parameters:
        n (int): The total number of elements in the resulting binary list.
        to_flip (set[int]): A set of integers to be mapped to a binary list
    Returns:
        List[bool]: A list of boolean values representing the binary mapping of the input set.

    Example:
        >>> n = 5
        >>> to_flip = {2, 3}
        >>> to_flip(observation)
        [0, 1, 1, 0, 0]
    """
    to_flip_binary = np.zeros(n, dtype=int)
    for n in to_flip:
        to_flip_binary[n - 1] = 1
    return to_flip_binary
