import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from gymnasium.utils import seeding
import numpy as np
import copy
from tool_envs import ToolBase
from typing import Optional, List

class FrozenLakeTool(ToolBase):
    name = "frozen_lake_tool"
    
    def __init__(self, **kwargs):
        super().__init__(
            name=self.name,
            description="A tool that simulates the Frozen Lake environment.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "integer", "description": "The action to take (0: none, 1: left, 2: down, 3: right, 4: up)"}
                },
                "required": ["action"]
            }
        )
        self.env = GymFrozenLakeEnv(desc=generate_random_map(size=8, p=0.8, seed=44), is_slippery=True, render_mode="ansi")
        self.env.reset()
        self.reward = 0

    def execute(self, *args, **kwargs) -> str:
        """
        Execute the tool functionality by taking an action in the Frozen Lake environment.
        
        Args:
            action: The action to take (0: none, 1: left, 2: down, 3: right, 4: up)
            
        Returns:
            observation: The current state of the environment
            reward: The reward received after taking the action
        """
        action = kwargs.get("action", 0)
        # if action == 0:
        #     return self.env.render(), self.reward
        
        player_pos, reward, done, _, _ = self.env.step(action)
        self.reward = reward
        print(player_pos, reward, done)
        if done:
            self.reset()
        
        return self.env.render(), self.reward

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.env.reset()
        self.reward = 0

# Helper function to generate a random map for the Frozen Lake environment
def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    # generate random start and end points

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])

        while True:
            start_r = np_random.integers(0, size)
            start_c = np_random.integers(0, size)
            goal_r = np_random.integers(0, size)
            goal_c = np_random.integers(0, size)
            
            # Ensure start and goal are different positions
            if (start_r, start_c) != (goal_r, goal_c):
                break
            
        board[start_r][start_c] = "S"
        board[goal_r][goal_c] = "G"
        
        valid = is_valid(board, size)
    return ["".join(x) for x in board]

# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    # find the start point
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0]))
    # dfs to check if there is a path from start to goal
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False

# Example usage
# action:[int]
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3
if __name__ == "__main__":
    tool = FrozenLakeTool()
    observation, reward = tool.execute(action=1)
    print(f"Observation:\n{observation}")
    print(f"Reward: {reward}")
    observation, reward = tool.execute(action=2)
    print(f"Observation:\n{observation}")
    print(f"Reward: {reward}")