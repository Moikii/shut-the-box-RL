import gymnasium as gym
import numpy as np
import pygame
import pygame.draw
from gymnasium import spaces
from typing import Dict, List, Any, Tuple


class Klappbrett(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self, render_mode: str = None, numbers: int = 9, dice: int = 2, sides: int = 6
    ) -> None:
        """
        Initializes an instance of the Klappbrett class with the given parameters.

        Args:
            render_mode (str, optional): The render mode for the environment. Defaults to None.
            numbers (int, optional): The number of numbers in the game. Defaults to 9.
            dice (int, optional): The number of dice in the game. Defaults to 2.
            sides (int, optional): The number of sides on each dice. Defaults to 6.

        Returns:
            None
        """
        self.numbers = numbers
        self.dice = dice
        self.sides = sides
        self.number_of_rolls = 0
        self.window_width = 1024
        self.window_height = self.window_width * 0.75
        self.current_flip = None
        self.action_space = spaces.MultiBinary(self.numbers)
        self.observation_space = spaces.Dict(
            {
                "board_state": spaces.MultiBinary(self.numbers),
                "dice_state": spaces.MultiDiscrete(
                    np.ones(self.dice) * self.sides, start=np.ones(self.dice)
                ),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the observation state of the environment.
        - "board_state" (numpy array): The current state of the game board.
        - "dice_state" (numpy array): The current state of the dice.
        - "possible_combinations" (list): Possible combinations based on the current dice and board state.
        """
        return {
            "board_state": self._board_state,
            "dice_state": self._dice_state,
            "possible_combinations": self.get_possible_combinations(
                self._dice_state, self._board_state
            ),
        }

    def _get_info(self) -> Dict[str, int]:
        """
        Returns a dictionary containing the current score and the number of rolls.

        Returns:
            dict: A dictionary with two keys:
                - "current_score" (int): The sum all non-flipped numbers
                - "number_of_rolls" (int): The number of rolls performed in the current episode
        """
        return {
            "current_score": self._board_state.dot(
                np.array(range(1, self.numbers + 1))
            ).sum(),
            "number_of_rolls": self.number_of_rolls,
        }

    def reset(self, seed: int = None) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): The seed value for random number generation. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and information about the environment.
                - observation (dict): The initial observation of the environment.
                - info (dict): Information about the environment, including the current score and number of rolls.
        """
        super().reset(seed=seed)

        self._board_state = np.ones(self.numbers, dtype=int)
        self._dice_state = self.np_random.integers(
            1, self.sides, size=self.dice, dtype=int, endpoint=True
        )
        self.number_of_rolls = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: np.array) -> Tuple:
        self.current_flip = action
        if self.render_mode == "human":
            self._render_frame()

        if (
            (sum(action) == 0)
            | ((np.bitwise_and(self._board_state, action) != action).any())
            | (action.dot(np.arange(1, self.numbers + 1)) != self._dice_state.sum())
        ):
            terminated = True
        else:
            terminated = False
            self.number_of_rolls += 1
            self._board_state = np.bitwise_xor(self._board_state, action)
            if self.render_mode == "human":
                self._render_frame()
            self._dice_state = self.np_random.integers(
                1, self.sides, size=self.dice, dtype=int, endpoint=True
            )

        reward = (
            -self._board_state.dot(np.arange(1, self.numbers + 1)) if terminated else 0
        )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def get_possible_combinations(self, rolled_dice, available_numbers):
        n = rolled_dice.sum()
        available_numbers = set(
            [i + 1 for i, n in enumerate(available_numbers) if n == 1]
        )

        possible_combinations = dict()
        for i in range(1, n + 1):
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
        return possible_combinations[n]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 100, 0))
        pix_number_size_width = self.window_width / self.numbers
        pix_number_size_height = self.window_height / 3

        for i, number in enumerate(self._board_state):
            if number == 1:
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
                    self.window_height / 4,
                )
            )
            canvas.blit(text, dest=text_rect)

        # Finally, add some gridlines
        for x in range(1, self.numbers):
            pygame.draw.line(
                canvas,
                0,
                (pix_number_size_width * x, 0),
                (pix_number_size_width * x, pix_number_size_height),
                width=3,
            )

        pygame.draw.line(canvas, 0, (0, 0), (0, self.window_height), width=3)
        pygame.draw.line(canvas, 0, (0, 0), (self.window_width, 0), width=3)
        pygame.draw.line(
            canvas,
            0,
            (0, self.window_height),
            (self.window_width, self.window_height),
            width=3,
        )
        pygame.draw.line(
            canvas,
            0,
            (self.window_width, 0),
            (self.window_width, self.window_height),
            width=3,
        )

        if self.current_flip is not None:
            for i, number in enumerate(self.current_flip):
                if number == 1:
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

        font = pygame.font.Font(pygame.font.get_default_font(), 36)
        text = font.render(f"{self._dice_state}", True, (0, 0, 0))
        if self.current_flip is not None and sum(self.current_flip) == 0:
            text = font.render("Game Over!", True, (0, 0, 0))

        text_rect = text.get_rect(
            center=(
                self.window_width / 2,
                self.window_height / 2,
            )
        )
        canvas.blit(text, dest=text_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
