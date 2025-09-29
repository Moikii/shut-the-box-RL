import gymnasium as gym
import numpy as np
import pygame
import pygame.draw
from gymnasium import spaces
from typing import Dict, List, Any, Tuple


class Klappbrett(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        render_mode: str | None = str(),
        board_size: int = 9,
        number_of_dice: int = 2,
        number_of_sides: int = 6,
    ) -> None:
        self.board_size = board_size
        self.number_of_dice = number_of_dice
        self.number_of_sides = number_of_sides

        self.number_of_rolls = 0
        self.window_width = 1024
        self.window_height = self.window_width * 0.75

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.canvas = None

    def _get_obs(self):
        return (tuple(self._board_state), self._dice_state)

    def _get_info(self):
        return {
            "current_score": self._get_reward(),
            "number_of_rolls": self.number_of_rolls,
        }

    def _get_reward(self):  #
        return (self.board_size * (self.board_size + 1)) / 2 - sum(
            i[1] if i[0] != 0 else 0
            for i in zip(self._board_state, range(1, self.board_size + 1))
        )

    def reset(self, seed: int = None, options=None):
        super().reset(seed=seed)

        self._board_state = [i for i in range(1, self.board_size + 1)]
        self._dice_state = self.np_random.integers(
            1, self.number_of_sides + 1, size=self.number_of_dice, dtype=int
        ).sum()

        self.number_of_rolls = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_board()
            self._render_dice()
        return observation, info

    def step(self, action):
        if self.render_mode == "human":
            self._render_current_flip(action)

        if sum(
            i[1] if i[0] != 0 else 0 for i in zip(action, range(1, self.board_size + 1))
        ) != self._dice_state | (
            not set(action).difference(set([0])).issubset(set(self._board_state))
        ):
            terminated = True
        else:
            terminated = False
            self.number_of_rolls += 1
            self._board_state = [i[0] - i[1] for i in zip(self._board_state, action)]
            self._dice_state = self.np_random.integers(
                1, self.number_of_sides + 1, size=self.number_of_dice, dtype=int
            ).sum()

            if self.render_mode == "human":
                self._render_board()
                self._render_dice()

        reward = self._get_reward() if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _render_board(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.window_width, self.window_height))
        self.canvas.fill((0, 100, 0))
        pix_number_size_width = self.window_width / self.board_size
        pix_number_size_height = self.window_height / 3

        for i, number in enumerate(self._board_state):
            if number != 0:
                color = (222, 184, 135)
            else:
                color = (139, 69, 19)

            pygame.draw.rect(
                self.canvas,
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
            self.canvas.blit(text, dest=text_rect)

        for x in range(1, self.board_size):
            pygame.draw.line(
                self.canvas,
                0,
                (pix_number_size_width * x, 0),
                (pix_number_size_width * x, pix_number_size_height),
                width=3,
            )

        pygame.draw.line(self.canvas, 0, (0, 0), (0, self.window_height), width=3)
        pygame.draw.line(self.canvas, 0, (0, 0), (self.window_width, 0), width=3)
        pygame.draw.line(
            self.canvas,
            0,
            (0, self.window_height),
            (self.window_width, self.window_height),
            width=3,
        )
        pygame.draw.line(
            self.canvas,
            0,
            (self.window_width, 0),
            (self.window_width, self.window_height),
            width=3,
        )

        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def _render_dice(self):
        font = pygame.font.Font(pygame.font.get_default_font(), 36)
        text = font.render(f"{self._dice_state}", True, (0, 0, 0))

        text_rect = text.get_rect(
            center=(
                self.window_width / 2,
                self.window_height / 2,
            )
        )
        self.canvas.blit(text, dest=text_rect)

        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def _render_current_flip(self, action):
        pix_number_size_width = self.window_width / self.board_size
        pix_number_size_height = self.window_height / 3

        if max(action) == 0:
            font = pygame.font.Font(pygame.font.get_default_font(), 72)
            text = font.render("Game Over!", True, (200, 0, 0))
            text_rect = text.get_rect(
                center=(
                    self.window_width / 2,
                    self.window_height * 3 / 4,
                )
            )
            self.canvas.blit(text, dest=text_rect)
        else:
            for i, number in enumerate(action):
                if number != 0:
                    pygame.draw.line(
                        self.canvas,
                        (200, 0, 0),
                        (i * pix_number_size_width, 0),
                        (i * pix_number_size_width, pix_number_size_height),
                        width=8,
                    )
                    pygame.draw.line(
                        self.canvas,
                        (200, 0, 0),
                        (i * pix_number_size_width, 0),
                        ((i + 1) * pix_number_size_width, 0),
                        width=8,
                    )
                    pygame.draw.line(
                        self.canvas,
                        (200, 0, 0),
                        ((i + 1) * pix_number_size_width, 0),
                        ((i + 1) * pix_number_size_width, pix_number_size_height),
                        width=8,
                    )
                    pygame.draw.line(
                        self.canvas,
                        (200, 0, 0),
                        (i * pix_number_size_width, pix_number_size_height),
                        ((i + 1) * pix_number_size_width, pix_number_size_height),
                        width=8,
                    )

        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
