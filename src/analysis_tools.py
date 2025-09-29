import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from action_conversions import index_to_action, action_to_index
import pygame


def get_ranked_actions(agent, obs):
    action_values = agent.q_values[obs]
    sorted_indices = np.argsort(action_values[:, 0]).tolist()
    sorted_indices.reverse()
    return [
        (set(index_to_action(agent.board_size, i)).difference(set([0])))
        for i in sorted_indices
        if agent.q_values[obs][i][0] != 0
    ]


def is_ordered_subset(rankings):
    while rankings[0]:
        first = rankings[0].pop(0)
        for i in range(1, len(rankings)):
            if (len(rankings[i]) != 0) and (first == rankings[i][0]):
                rankings[i].pop(0)
    print(rankings)
    return sum([len(ranking) for ranking in rankings]) == 0


def calculate_policy(env, agent):
    roll_dict = dict()
    for state, roll in agent.q_values.keys():
        if roll not in roll_dict.keys():
            roll_dict[roll] = list()
        roll_dict[roll].append(get_ranked_actions(agent, (state, roll)))

    for roll in roll_dict.keys():
        roll_dict[roll] = sorted(roll_dict[roll], key=lambda x: len(x), reverse=True)
        same_order = is_ordered_subset(roll_dict[roll])
        if same_order:
            print(f"{roll}: {roll_dict[roll][0]}")


def plot_training(env_wrapped):
    rolling_length = 500
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(env_wrapped.return_queue).flatten(),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env_wrapped.length_queue).flatten(),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    plt.tight_layout()
    plt.show()


def plot_policy(
    agent, board_size=9, number_of_dice=2, number_of_sides=6, window_width=1024, fps=1
):
    board_state = np.ones(board_size)
    window_height = window_width * 3 / 4
    pygame.init()
    pygame.display.init()
    window = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()

    canvas = pygame.Surface((window_width, window_height))
    canvas.fill((0, 100, 0))
    pix_number_size_width = window_width / board_size
    pix_number_size_height = window_height / 3

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:  # and event.button == 1:
                # update
                ## if mouse is pressed get position of cursor ##
                pos = pygame.mouse.get_pos()
                ## check if cursor is on button ##
            elif event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                pygame.quit()

        for i, number in enumerate(board_state):
            if number != 0:
                color = (222, 184, 135)
            else:
                color = (139, 69, 19)

            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    (i * pix_number_size_width, 0),
                    (pix_number_size_width, pix_number_size_height),
                ),
            )
            font = pygame.font.Font(pygame.font.get_default_font(), 36)
            text = font.render(f"{i+1}", True, (0, 0, 0))
            text_rect = text.get_rect(
                center=(
                    i * pix_number_size_width + pix_number_size_width / 2,
                    window_height / 4,
                )
            )
            canvas.blit(text, dest=text_rect)

        for x in range(1, board_size):
            pygame.draw.line(
                canvas,
                0,
                (pix_number_size_width * x, 0),
                (pix_number_size_width * x, pix_number_size_height),
                width=3,
            )

        pygame.draw.line(canvas, 0, (0, 0), (0, window_height), width=3)
        pygame.draw.line(canvas, 0, (0, 0), (window_width, 0), width=3)
        pygame.draw.line(
            canvas,
            0,
            (0, window_height),
            (window_width, window_height),
            width=3,
        )
        pygame.draw.line(
            canvas,
            0,
            (window_width, 0),
            (window_width, window_height),
            width=3,
        )

        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()


def display_ranked_actions(agent, obs, canvas, window_width, window_height, window):
    font = pygame.font.Font(pygame.font.get_default_font(), 36)
    text = font.render(f"{get_ranked_actions(agent, obs)}", True, (0, 0, 0))

    text_rect = text.get_rect(
        center=(
            window_width / 2,
            window_height / 2,
        )
    )
    canvas.blit(text, dest=text_rect)

    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.update()


def render_best_flip(
    action, window_width, window_height, env, canvas, window, clock, fps
):
    pix_number_size_width = window_width / env.board_size
    pix_number_size_height = window_height / 3

    if max(action) == 0:
        font = pygame.font.Font(pygame.font.get_default_font(), 72)
        text = font.render("Game Over!", True, (200, 0, 0))
        text_rect = text.get_rect(
            center=(
                window_width / 2,
                window_height * 3 / 4,
            )
        )
        canvas.blit(text, dest=text_rect)
    else:
        for i, number in enumerate(action):
            if number != 0:
                pygame.draw.line(
                    canvas,
                    (200, 0, 0),
                    (i * pix_number_size_width, 0),
                    (i * pix_number_size_width, pix_number_size_height),
                    width=8,
                )
                pygame.draw.line(
                    canvas,
                    (200, 0, 0),
                    (i * pix_number_size_width, 0),
                    ((i + 1) * pix_number_size_width, 0),
                    width=8,
                )
                pygame.draw.line(
                    canvas,
                    (200, 0, 0),
                    ((i + 1) * pix_number_size_width, 0),
                    ((i + 1) * pix_number_size_width, pix_number_size_height),
                    width=8,
                )
                pygame.draw.line(
                    canvas,
                    (200, 0, 0),
                    (i * pix_number_size_width, pix_number_size_height),
                    ((i + 1) * pix_number_size_width, pix_number_size_height),
                    width=8,
                )

    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.update()
    clock.tick(fps)


def close(window):
    if window is not None:
        pygame.display.quit()
        pygame.quit()
