# MegaGrid Environment

## Description
MegaGrid is a customizable grid-world environment built on the Gymnasium framework. It features procedurally generated rooms connected by colored doors that can be unlocked with matching keys. The agent must navigate through rooms, collect keys, unlock doors, and gather stars while receiving text descriptions of its actions.

## Features
- Procedurally generated rooms with varying sizes
- Color-coded doors and keys system
- Field of view (FOV) system that limits visibility
- Text descriptions of agent actions
- Star collection objectives
- Real-time observation window showing agent's perspective

## Installation
1. Clone the repository
2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage
Run the environment:
```python
python test_env.py
```

### Controls
- Arrow Keys: Move (Left/Right/Up/Down)
- Space: Interact (pick up keys/unlock doors)
- Q: Quit game

## Environment Details

### Observation Space
- Image: 21x21x3 RGB array showing agent's field of view
- Description: Text description of agent's last action

### Action Space
Five discrete actions:
- 0: Left
- 1: Right
- 2: Forward/Up
- 3: Down
- 4: Interact

### Objects
- Doors: Color-coded barriers between rooms
- Keys: Matching colored keys to unlock doors
- Stars: Collectible objectives
- Walls: Define room boundaries

### Agent Properties
- Direction: Facing direction (0-3)
- Color: Current color (based on held key)
- Key: Currently held key (if any)
- Stars: Number of stars collected

## Visualization
The environment provides two windows:
1. Main game window (Pygame): Shows the entire environment
2. Observation window (OpenCV): Shows agent's current view and action descriptions

## Technical Details
- Room generation uses recursive splitting algorithm
- FOV system uses raycasting to determine visible cells
- Ensures all rooms are accessible through key-door placement
- Maintains room connectivity through strategic door placement

## Dependencies
- gymnasium>=0.29.1
- pygame>=2.5.2
- numpy>=1.26.0
- opencv-python>=4.8.1
- scikit-image>=0.22.0
- minigrid>=2.3.1

## License
[MIT License](https://opensource.org/licenses/MIT)
```

</rewritten_file>