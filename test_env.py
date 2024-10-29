import gymnasium as gym
import pygame
from megagrid import MegaGrid

def main():
    env = gym.make('MegaGrid-v0', render_mode="human")
    obs, info = env.reset()

    print("MegaGrid Environment created successfully.")
    print("Controls:")
    print("Arrow keys: Move (Left/Right to turn, Up to move forward)")
    print("Space: Interact (pick up keys or unlock doors)")
    print("R: Reset environment")
    print("Q: Quit game")

    running = True
    while running:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
                elif event.key == pygame.K_SPACE:
                    action = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_q:
                    running = False

                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    if done:
                        print("Goal reached!" if terminated else "Episode truncated!")
                        obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
