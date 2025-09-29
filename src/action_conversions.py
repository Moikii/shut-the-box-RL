def index_to_action(board_size, index):
    binary = [int(x) for x in bin(index)[2:]]
    binary = [0] * (board_size - len(binary)) + binary
    binary.reverse()
    return [i[0] * i[1] for i in zip(range(1, board_size + 1), binary)]


def action_to_index(action):
    return sum([2 ** (i - 1) if i != 0 else 0 for i in action])
