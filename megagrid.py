import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Door, Key, WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLORS, COLOR_NAMES
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_triangle
import random
import numpy as np
import pygame
from skimage.draw import rectangle_perimeter
import math

# Add pink if it's not already defined
if 'pink' not in COLORS:
    COLORS['pink'] = (255, 192, 203)  # RGB value for pink

# Add white to COLORS if it's not already there
if 'white' not in COLORS:
    COLORS['white'] = (255, 255, 255)

VALID_COLORS = ['red', 'green', 'blue', 'purple']

class Star(WorldObj):
    def __init__(self):
        super().__init__('ball', 'yellow')  # Changed to use 'yellow' instead of 'white'

class Agent:
    def __init__(self, dir=0, color='yellow'):  # Changed default color to 'yellow'
        self.dir = dir
        self.color = color
        self.key = None
        self.stars = 0

    @property
    def current_color(self):
        # Return the color of the held key, or yellow if no key
        return self.key if self.key else 'yellow'  # Changed default to 'yellow'

class MegaGrid(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, width=100, height=100, max_steps=1000, render_mode="human"):
        self.width = width
        self.height = height
        self.max_steps = max_steps

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(21, 21, 3), dtype=np.uint8
                ),
                "description": gym.spaces.Text(max_length=200)  # Increased length for detailed descriptions
            }
        )

        self.action_space = gym.spaces.Discrete(5)  # 0: left, 1: right, 2: forward, 3: down, 4: interact

        self.mission_space = MissionSpace(mission_func=lambda: "Collect stars and unlock doors")

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 10  # Size of each cell in pixels

        self.step_count = 0

        self.last_action_description = ""

    @classmethod
    def make(cls, render_mode="human"):
        return cls(render_mode=render_mode)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height-1, Wall())
        for j in range(height):
            self.grid.set(0, j, Wall())
            self.grid.set(width-1, j, Wall())
        
        # Generate rooms
        self.rooms = []
        self._split_room(1, 1, width-2, height-2)
        
        # Add doors between rooms and keys in each room
        self._add_doors_and_keys()
        
        # Debug: Print the number of rooms and doors
        print(f"Number of rooms: {len(self.rooms)}")
        door_count = sum(self._count_doors(room) for room in self.rooms)
        print(f"Total number of doors: {door_count}")
        
        # Place the agent in a random room
        self.agent_pos = self.place_agent()
        self.agent_dir = random.randint(0, 3)
        self.agent = Agent(self.agent_dir, 'red')

        # Place stars in random empty positions
        num_stars = len(self.rooms) // 2
        for _ in range(num_stars):
            self.place_obj(Star())

    def _split_room(self, x, y, w, h):
        room_area = w * h
        min_room_size = 20  # Increased from 9
        max_room_size = 200  # Increased from 36

        if room_area <= max_room_size or w < 4 or h < 4:  # Changed from 3 to 4
            if room_area >= min_room_size:
                self.rooms.append((x, y, w, h))
            return

        # Decide split direction
        if w > h:
            split_vertical = True
        elif h > w:
            split_vertical = False
        else:
            split_vertical = random.choice([True, False])

        if split_vertical:
            split = random.randint(4, w - 4)  # Changed from 3 to 4
            self._split_room(x, y, split, h)
            self._split_room(x + split, y, w - split, h)
            # Add wall
            for i in range(y, y + h):
                self.grid.set(x + split - 1, i, Wall())
        else:
            split = random.randint(4, h - 4)  # Changed from 3 to 4
            self._split_room(x, y, w, split)
            self._split_room(x, y + split, w, h - split)
            # Add wall
            for i in range(x, x + w):
                self.grid.set(i, y + split - 1, Wall())

    def _add_doors_and_keys(self):
        for room in self.rooms:
            # Add doors to the room
            door_colors = set()
            for _ in range(2):  # Try to add up to 2 doors
                door_color = self._add_door_to_room(room)
                if door_color:
                    door_colors.add(door_color)
            
            # Ensure at least one key in the room
            if not door_colors:
                # If no doors were added, add a random color key
                key_color = random.choice(VALID_COLORS)
            else:
                # Add keys for each door color
                for key_color in door_colors:
                    self._add_key_to_room(room, key_color)
            
            # Add an extra random key to ensure the player can always progress
            extra_key_color = random.choice(VALID_COLORS)
            self._add_key_to_room(room, extra_key_color)

        # After adding all doors and keys, ensure connectivity
        self._ensure_connectivity()

    def _ensure_connectivity(self):
        # Create a set of all door colors in the environment
        all_door_colors = set()
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door):
                    all_door_colors.add(cell.color)
        
        # Ensure each room has keys for all door colors
        for room in self.rooms:
            room_keys = set()
            x, y, w, h = room
            for i in range(x, x + w):
                for j in range(y, y + h):
                    cell = self.grid.get(i, j)
                    if isinstance(cell, Key):
                        room_keys.add(cell.color)
            
            # Add missing keys
            for color in all_door_colors - room_keys:
                self._add_key_to_room(room, color)

    def _add_door_to_room(self, room):
        x, y, w, h = room
        possible_doors = [
            (x-1, y+h//2, 0, (x-2, y+h//2)),  # Left wall
            (x+w, y+h//2, 0, (x+w+1, y+h//2)),  # Right wall
            (x+w//2, y-1, 1, (x+w//2, y-2)),  # Top wall
            (x+w//2, y+h, 1, (x+w//2, y+h+1))  # Bottom wall
        ]
        random.shuffle(possible_doors)
        
        for door_x, door_y, is_vertical, adjacent_pos in possible_doors:
            # Ensure the door is not on a boundary wall and the adjacent cell is empty
            if (1 <= door_x < self.width-1 and 1 <= door_y < self.height-1 and
                isinstance(self.grid.get(door_x, door_y), Wall) and
                self.grid.get(*adjacent_pos) is None):
                # Choose a random color for the door
                door_color = random.choice(VALID_COLORS)
                door = Door(door_color, is_open=False)
                self.grid.set(door_x, door_y, door)
                return door_color
        print(f"Failed to add door to room {room}")  # Debug print
        return None

    def _add_key_to_room(self, room, key_color):
        x, y, w, h = room
        attempts = 0
        max_attempts = 100  # Prevent infinite loop
        
        def is_adjacent_to_door(pos_x, pos_y):
            # Check all adjacent cells for doors
            adjacent_positions = [
                (pos_x + 1, pos_y),
                (pos_x - 1, pos_y),
                (pos_x, pos_y + 1),
                (pos_x, pos_y - 1)
            ]
            for adj_x, adj_y in adjacent_positions:
                if (0 <= adj_x < self.width and 
                    0 <= adj_y < self.height and 
                    isinstance(self.grid.get(adj_x, adj_y), Door)):
                    return True
            return False

        while attempts < max_attempts:
            key_x = random.randint(x, x + w - 1)
            key_y = random.randint(y, y + h - 1)
            
            # Check if position is empty and not adjacent to a door
            if (self.grid.get(key_x, key_y) is None and 
                not is_adjacent_to_door(key_x, key_y)):
                self.grid.set(key_x, key_y, Key(key_color))
                print(f"Key placed at ({key_x}, {key_y}) with color {key_color}")  # Debug print
                return
            attempts += 1
        print(f"Failed to place key in room {room}")  # Debug print

    def _count_doors(self, room):
        x, y, w, h = room
        doors = 0
        if x > 0 and isinstance(self.grid.get(x-1, y+h//2), Door):
            doors += 1
        if x+w < self.width and isinstance(self.grid.get(x+w, y+h//2), Door):
            doors += 1
        if y > 0 and isinstance(self.grid.get(x+w//2, y-1), Door):
            doors += 1
        if y+h < self.height and isinstance(self.grid.get(x+w//2, y+h), Door):
            doors += 1
        return doors

    def _can_move_to(self, pos):
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return False
        cell = self.grid.get(*pos)
        return cell is None or (isinstance(cell, Door) and cell.is_open)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._gen_grid(self.width, self.height)
        self.step_count = 0
        self.agent.stars = 0
        self.agent.key = None

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False

        # Clear the last action description
        self.last_action_description = ""

        if action == 4:  # Interact
            self._interact()
        else:
            # Movement actions with detailed descriptions
            next_pos = list(self.agent_pos)
            direction = ""
            if action == 0:  # Left
                next_pos[0] -= 1
                self.agent_dir = 2
                direction = "left"
            elif action == 1:  # Right
                next_pos[0] += 1
                self.agent_dir = 0
                direction = "right"
            elif action == 2:  # Forward/Up
                next_pos[1] -= 1
                self.agent_dir = 3
                direction = "up"
            elif action == 3:  # Down
                next_pos[1] += 1
                self.agent_dir = 1
                direction = "down"

            next_pos = tuple(next_pos)
            if self._can_move_to(next_pos):
                self.agent_pos = next_pos
            else:
                cell = self.grid.get(*next_pos) if 0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height else None

        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, False, info

    def _get_obs(self):
        # Create a partial observation of the grid around the agent
        obs_size = 21
        obs = np.zeros((obs_size, obs_size, 3), dtype=np.uint8)
        
        # Calculate the center of the observation window
        center = obs_size // 2
        
        # Get the FOV cells (visible cells not blocked by walls)
        fov_cells = set(self._get_fov())
        
        for i in range(obs_size):
            for j in range(obs_size):
                # Calculate world coordinates (always aligned to north)
                world_x = self.agent_pos[0] + (i - center)
                world_y = self.agent_pos[1] + (j - center)
                
                if 0 <= world_x < self.width and 0 <= world_y < self.height:
                    cell = self.grid.get(world_x, world_y)
                    
                    if (world_x, world_y) in fov_cells:
                        # Cell is visible
                        if cell is None:
                            obs[j, i] = (0, 0, 0)  # Empty cell
                        elif isinstance(cell, Wall):
                            obs[j, i] = (128, 128, 128)  # Wall
                        elif isinstance(cell, Door):
                            # Make doors more visible with brighter colors
                            door_color = COLORS[cell.color]
                            if cell.is_open:
                                # Make open doors slightly darker than closed ones
                                obs[j, i] = tuple(max(0, c - 50) for c in door_color)
                            else:
                                obs[j, i] = door_color
                        elif isinstance(cell, Key):
                            obs[j, i] = COLORS[cell.color]  # Key color
                        elif isinstance(cell, Star):
                            obs[j, i] = COLORS['yellow']  # Star color
                        
                        # Add agent position to observation
                        if (world_x, world_y) == self.agent_pos:
                            obs[j, i] = COLORS[self.agent.current_color]  # Agent color
                    else:
                        # Cell is not visible - show as wall color if beyond a wall
                        obs[j, i] = (128, 128, 128)  # Wall color for non-visible cells
                else:
                    # Out of bounds is now treated the same as areas beyond walls
                    obs[j, i] = (128, 128, 128)  # Wall color

        return {
            "image": obs,
            "description": self.last_action_description
        }

    def _get_info(self):
        return {
            "steps": self.step_count,
            "agent_pos": self.agent_pos,
            "stars": self.agent.stars
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))
        canvas.fill((0, 0, 0))  # Fill the background with black

        # Draw the grid cells
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                rect = pygame.Rect(
                    i * self.cell_size, 
                    j * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                # Draw base cell
                if cell is None:
                    pygame.draw.rect(canvas, (40, 40, 40), rect)
                elif isinstance(cell, Wall):
                    pygame.draw.rect(canvas, COLORS['grey'], rect)
                elif isinstance(cell, Door):
                    door_color = COLORS[cell.color]
                    pygame.draw.rect(canvas, door_color, rect)
                    pygame.draw.rect(canvas, (0, 0, 0), rect, 1)  # Black outline
                elif isinstance(cell, Key):
                    key_color = COLORS[cell.color]
                    center = (
                        int(i * self.cell_size + self.cell_size / 2),
                        int(j * self.cell_size + self.cell_size / 2)
                    )
                    radius = int(self.cell_size * 0.4)
                    pygame.draw.circle(canvas, key_color, center, radius)
                    pygame.draw.circle(canvas, (0, 0, 0), center, radius, 1)
                elif isinstance(cell, Star):
                    star_color = COLORS['yellow']
                    center = (
                        int(i * self.cell_size + self.cell_size / 2),
                        int(j * self.cell_size + self.cell_size / 2)
                    )
                    radius = int(self.cell_size * 0.3)
                    pygame.draw.circle(canvas, star_color, center, radius)
                    for angle in range(0, 360, 45):
                        end_point = (
                            center[0] + int(radius * 1.5 * math.cos(math.radians(angle))),
                            center[1] + int(radius * 1.5 * math.sin(math.radians(angle)))
                        )
                        pygame.draw.line(canvas, star_color, center, end_point, 2)

                pygame.draw.rect(canvas, (100, 100, 100), rect, 1)  # Grid cell outline

        # Highlight agent's FOV with a semi-transparent overlay
        fov = self._get_fov()
        fov_surface = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size), pygame.SRCALPHA)
        for x, y in fov:
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(fov_surface, (255, 255, 255, 96), rect)  # Increased opacity from 64 to 96
        canvas.blit(fov_surface, (0, 0))

        # Draw grid lines (avoiding walls)
        for i in range(self.width + 1):
            for j in range(self.height):
                if i == 0 or i == self.width or not isinstance(self.grid.get(i-1, j), Wall) or not isinstance(self.grid.get(i, j), Wall):
                    pygame.draw.line(canvas, (64, 64, 64), 
                                     (i * self.cell_size, j * self.cell_size), 
                                     (i * self.cell_size, (j+1) * self.cell_size))

        for j in range(self.height + 1):
            for i in range(self.width):
                if j == 0 or j == self.height or not isinstance(self.grid.get(i, j-1), Wall) or not isinstance(self.grid.get(i, j), Wall):
                    pygame.draw.line(canvas, (64, 64, 64), 
                                     (i * self.cell_size, j * self.cell_size), 
                                     ((i+1) * self.cell_size, j * self.cell_size))

        # Draw the agent
        agent_color = COLORS[self.agent.current_color]
        agent_pos = (
            int((self.agent_pos[0] + 0.5) * self.cell_size),
            int((self.agent_pos[1] + 0.5) * self.cell_size)
        )
        agent_size = int(self.cell_size * 0.8)

        # Define a default triangle (pointing right)
        points = [
            (agent_pos[0] + agent_size // 2, agent_pos[1]),
            (agent_pos[0] - agent_size // 2, agent_pos[1] - agent_size // 2),
            (agent_pos[0] - agent_size // 2, agent_pos[1] + agent_size // 2)
        ]

        # Rotate the triangle based on agent direction
        if self.agent_dir == 0:  # Facing right (default)
            pass  # No rotation needed
        elif self.agent_dir == 1:  # Facing down
            points = [(x - agent_pos[0], y - agent_pos[1]) for x, y in points]
            points = [(y + agent_pos[0], x + agent_pos[1]) for x, y in points]
        elif self.agent_dir == 2:  # Facing left
            points = [(2 * agent_pos[0] - x, y) for x, y in points]
        elif self.agent_dir == 3:  # Facing up
            points = [(x - agent_pos[0], y - agent_pos[1]) for x, y in points]
            points = [(y + agent_pos[0], -x + agent_pos[1]) for x, y in points]

        pygame.draw.polygon(canvas, agent_color, points)

        # Draw the agent's inventory (single key)
        if self.agent.key:
            key_rect = pygame.Rect(
                0,  # Only one position needed
                self.height * self.cell_size, 
                self.cell_size, 
                self.cell_size
            )
            pygame.draw.rect(canvas, COLORS[self.agent.key], key_rect)
            pygame.draw.rect(canvas, (0, 0, 0), key_rect, 1)  # Black outline

        # Draw the agent's star count
        font = pygame.font.Font(None, 24)
        star_text = font.render(f"Stars: {self.agent.stars}", True, COLORS['white'])
        canvas.blit(star_text, (10, self.height * self.cell_size + 10))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return pygame.surfarray.array3d(canvas).transpose((1, 0, 2))

    def _find_empty_position(self):
        while True:
            pos = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))
            if self.grid.get(*pos) is None:
                return pos

    def place_obj(self, obj):
        pos = self._find_empty_position()
        self.grid.set(*pos, obj)
        return pos

    def place_agent(self):
        return self._find_empty_position()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _is_door(self, pos):
        x, y = pos
        return isinstance(self.grid.get(x, y), Door)

    def update_agent_dir(self, new_dir):
        self.agent_dir = new_dir % 4  # This ensures the direction is always 0, 1, 2, or 3

    def _interact(self):
        front_pos = self._get_next_pos()
        front_cell = self.grid.get(*front_pos)

        if isinstance(front_cell, Key):
            if self.agent.key is not None:
                self.last_action_description = (
                    f"Agent picked up a {front_cell.color} key"
                )
                # Swap keys with detailed description
                old_key_color = self.agent.key
                self.agent.key = front_cell.color
                self.grid.set(*front_pos, Key(old_key_color))
            else:
                # Pick up key with detailed description
                self.agent.key = front_cell.color
                self.grid.set(*front_pos, None)
                
        elif isinstance(front_cell, Door):
            self.last_action_description = (
                    f"Agent used a {self.agent.key} key on a {front_cell.color} door"
                )
            if self.agent.key == front_cell.color:
                # Successful door interaction
                self.grid.set(*front_pos, None)
                
                self.agent.key = None
        elif isinstance(front_cell, Star):
            self.last_action_description = (
                    f"Agent picked up a star"
                )
            self.agent.stars += 1
            self.grid.set(*front_pos, None)

    def _get_fov(self):
        """Get the cells in the agent's field of view (centered on agent), blocked by walls and closed doors."""
        fov = set()
        x, y = self.agent_pos
        radius = 10  # For 21x21 grid (10 cells in each direction)
        
        # Always add the agent's position
        fov.add((x, y))
        
        # Helper function to check if a cell blocks vision
        def blocks_vision(pos):
            if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
                return True
            cell = self.grid.get(*pos)
            # Both walls and closed doors block vision
            return (isinstance(cell, Wall) or 
                    (isinstance(cell, Door) and not cell.is_open))

        # Cast rays in all directions
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i == 0 and j == 0:
                    continue
                
                target_x, target_y = x + i, y + j
                
                # Use Bresenham's line algorithm
                dx, dy = abs(target_x - x), abs(target_y - y)
                x0, y0 = x, y
                
                if dx > dy:
                    steps = dx
                else:
                    steps = dy
                    
                if steps == 0:
                    continue
                    
                x_inc = (target_x - x) / float(steps)
                y_inc = (target_y - y) / float(steps)
                
                curr_x, curr_y = x0, y0
                vision_blocked = False
                
                for _ in range(int(steps) + 1):
                    cell_x, cell_y = int(round(curr_x)), int(round(curr_y))
                    
                    # Add this cell to FOV before checking if it blocks vision
                    if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                        fov.add((cell_x, cell_y))
                    
                    # If we hit a wall, stop this ray
                    if blocks_vision((cell_x, cell_y)):
                        vision_blocked = True
                        break
                    
                    if vision_blocked:
                        break
                        
                    curr_x += x_inc
                    curr_y += y_inc

        return list(fov)

    def _get_next_pos(self):
        """Get the position in front of the agent based on their direction"""
        x, y = self.agent_pos
        if self.agent_dir == 0:  # Right
            return (x + 1, y)
        elif self.agent_dir == 1:  # Down
            return (x, y + 1)
        elif self.agent_dir == 2:  # Left
            return (x - 1, y)
        else:  # Up
            return (x, y - 1)

# Register the environment
gym.envs.registration.register(
    id='MegaGrid-v0',
    entry_point=lambda render_mode: MegaGrid(render_mode=render_mode),
)

