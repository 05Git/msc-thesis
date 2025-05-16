import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
import numpy as np

class VecEnvMDTransferActionWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(venv=venv)
        self.valid_moves = venv.action_space.nvec[0]
        self.valid_attacks = venv.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 7]) # 8 moves, 6 attacks, plus 1 no-op each
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        for action in actions:
            action[0] = action[0] if action[0] < self.valid_moves else self.no_move_idx
            action[1] = action[1] if action[1] < self.valid_attacks else self.no_attack_idx
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

class VecEnvDiscreteTransferActionWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, no_op_idx: int = 0):
        super().__init__(venv=venv)
        self.valid_actions = venv.action_space.n
        self.action_space = gym.spaces.Discrete(8 + 6 + 1) # 8 moves, 6 attacks, plus 1 no-op each
        self.no_op_idx = no_op_idx

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        for action in actions:
            action = action if action < self.valid_actions else self.no_op_idx
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

class MDTransferActionWrapper(gym.Wrapper):
    def __init__(self, env, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.valid_moves = env.action_space.nvec[0]
        self.valid_attacks = env.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 7]) # 8 moves, 6 attacks, plus 1 no-op each
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        '''
        Checks if a chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        move_idx, attack_idx = action[0]
        move = move_idx if move_idx < self.valid_moves else self.no_move_idx
        attack = attack_idx if attack_idx < self.valid_attacks else self.no_attack_idx
        step_result = self.env.step([[move, attack]])

        # Unpack the step result depending on the API.
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result

        return obs, reward, terminated, truncated, info

class DiscreteTransferActionWrapper(gym.Wrapper):
    def __init__(self, env, no_op_idx: int = 0):
        super().__init__(env)
        self.valid_actions = env.action_space.n
        self.action_space = gym.spaces.Discrete(8 + 6 + 1) # 8 moves, 6 attacks, plus 1 no-op
        self.no_op_idx = no_op_idx

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        '''
        Checks if the chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        action = action if action < self.valid_actions else self.no_op_idx
        return self.env.step(action)