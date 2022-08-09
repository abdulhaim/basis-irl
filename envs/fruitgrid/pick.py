from envs.fruitgrid.minigrid import *
from envs.fruitgrid.register import register
import random

from gym import spaces
from envs.fruitgrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        return full_grid

class PickEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
            self,
            multitask=True,
            task_id="red",
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0):
        self.multitask = multitask
        self.task_id = task_id
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.n_colors = 3
        self.reward_freq = {"red": 0.8, "orange": 0.2, "green": 0.0}

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            see_through_walls=True
        )

    def reset(self):
        return super(PickEnv, self).reset()

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        self.objects = []
        self.colors = list(range(self.n_colors))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        colors = ["red", "orange", "green"]
        for j in range(len(colors)):
            for i in range(3):
                self.objects.append(Ball(color=colors[j]))
                self.place_obj(self.objects[-1], max_tries=100)

        self.place_agent()

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.multitask:
            if self.carrying:
                obj = self.carrying
                if obj.color == self.task_id:
                    reward += 1
                self.place_obj(obj, max_tries=100)
        else:
            if self.carrying:
                obj = self.carrying
                reward += self.reward_freq[obj.color]
                self.place_obj(obj, max_tries=100)

        self.carrying = None
        return obs, reward, done, info

    def get_expert_reward(self, action):
        reward = 0 
        if self.multitask:
            if self.carrying:
                obj = self.carrying
                if obj.color == self.task_id:
                    reward += 1
        else:
            if self.carrying:
                obj = self.carrying
                reward += self.reward_freq[obj.color]

        return reward

class PickEnv8x8(PickEnv):
    def __init__(self, multitask, task_id):
        super().__init__(multitask, task_id, size=8, agent_start_pos=None)


def create_envs(config):
    tasks = ["red", "orange", "green"]
    envs = []

    for i in range(config.env_num_tasks):
        env = FullyObsWrapper(PickEnv8x8(multitask=config.multiple_tasks, task_id=tasks[i]))
        env.seed(config.seed)
        env.action_space.seed(config.seed)
        env.max_episode_steps = env.max_steps
        state = env.reset()
        env.observation_shape = env.reset().shape
        envs.append(env)

    return envs

