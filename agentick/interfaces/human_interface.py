"""Pygame-based human play interface."""

import pygame


class HumanInterface:
    """Pygame-based human play mode with keyboard mapping."""

    def __init__(self, env):
        self.env = env
        pygame.init()

        # Get pixel observation to determine window size
        pixel_obs = env.get_pixel_observation()
        self.window_size = (pixel_obs.shape[1], pixel_obs.shape[0])
        self.screen = pygame.Surface(self.window_size)

    def get_human_action(self):
        """Get action from keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return 1  # MOVE_UP
                elif event.key == pygame.K_DOWN:
                    return 2  # MOVE_DOWN
                elif event.key == pygame.K_LEFT:
                    return 3  # MOVE_LEFT
                elif event.key == pygame.K_RIGHT:
                    return 4  # MOVE_RIGHT
                elif event.key == pygame.K_SPACE:
                    return 5  # PICKUP
                elif event.key == pygame.K_d:
                    return 6  # DROP
                elif event.key == pygame.K_e:
                    return 7  # USE
                elif event.key == pygame.K_r:
                    return -1  # Reset
        return 0  # NOOP
