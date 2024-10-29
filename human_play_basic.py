import gymnasium as gym
from minigrid.envs import EmptyEnv
import pygame
import time

# Register the MiniGrid environment
gym.envs.registration.register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='minigrid.envs:EmptyEnv',
    kwargs={'size': 8}
)

# Create the environment
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")

# Initialize Pygame for keyboard input
pygame.init()

print("Environment created successfully.")
print("Controls:")
print("Arrow keys: Move (Left/Right to turn, Up to move forward)")
print("R: Reset environment")
print("Q: Quit game")

obs, info = env.reset()
done = False

while not done:
    env.render()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 0  # Turn left
            elif event.key == pygame.K_RIGHT:
                action = 1  # Turn right
            elif event.key == pygame.K_UP:
                action = 2  # Move forward
            elif event.key == pygame.K_r:
                obs, info = env.reset()
                continue
            elif event.key == pygame.K_q:
                done = True
                break
            else:
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                print("Episode ended. Resetting...")
                time.sleep(1)
                obs, info = env.reset()

    time.sleep(0.1)  # Small delay to prevent the game from running too fast

env.close()
pygame.quit()
print("Game ended.")
