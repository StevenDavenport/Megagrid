import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
from megagrid import MegaGrid

def visualize_observation(obs):
    # Display the observation
    plt.figure(figsize=(10, 10))
    plt.imshow(obs['image'])
    plt.title(obs['description'])
    plt.show()

def main():
    env = gym.make('MegaGrid-v0', render_mode="human")
    obs, info = env.reset()
    
    print("MegaGrid Environment created successfully.")
    print("Controls:")
    print("Arrow keys: Move (Left/Right/Up/Down)")
    print("Space: Interact (pick up keys or unlock doors)")
    print("Q: Quit game")
    
    running = True
    while running:
        env.render()
        
        # Visualize current observation
        visualize_observation(obs)
        
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
                elif event.key == pygame.K_q:
                    running = False
                
                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    #print(f"Direction: {obs['direction']}, Position: {obs['agent_pos']}")
                    #print(obs['description'])

    env.close()

if __name__ == "__main__":
    main()
