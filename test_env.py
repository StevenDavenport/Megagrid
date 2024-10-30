import gymnasium as gym
import pygame
import numpy as np
import cv2
from megagrid import MegaGrid

def update_observation_window(obs):
    # Convert the observation image to BGR format for OpenCV
    obs_img = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
    
    # Scale up the image (21x21 → 420x420)
    obs_img = cv2.resize(obs_img, (420, 420), interpolation=cv2.INTER_NEAREST)
    
    # Add black background for text
    obs_display = np.zeros((480, 420, 3), dtype=np.uint8)
    obs_display[0:420, :] = obs_img
    
    # Add the description text with larger font
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = obs['description'][:80]  # Show more text
    # Split text into two lines if it's too long
    if len(text) > 40:
        line1 = text[:40]
        line2 = text[40:]
        cv2.putText(obs_display, line1, (10, 445), font, 0.7, (255, 255, 255), 1)
        cv2.putText(obs_display, line2, (10, 470), font, 0.7, (255, 255, 255), 1)
    else:
        cv2.putText(obs_display, text, (10, 460), font, 0.7, (255, 255, 255), 1)
    
    # Show the observation window
    cv2.imshow('Agent Observation', obs_display)
    cv2.waitKey(1)  # Update the window

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
        
        # Update observation window using OpenCV
        update_observation_window(obs)
        
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

    cv2.destroyAllWindows()
    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()
