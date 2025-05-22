import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
import numpy as np

class VecEnvMDTransferActionWrapper(VecEnvWrapper):
    '''
    VectorEnv Multi-Discrete Transfer Wrapper
    Necessary for wrapping SubprocVecEnvs during distributed training
    '''
    def __init__(self, venv: VecEnv, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(venv=venv)
        self.valid_moves = venv.action_space.nvec[0]
        self.valid_attacks = venv.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 19]) # 19 attacks is the largest possible action space (MvC)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
        if hasattr(venv, "image_space_keys"):
            self.image_space_keys = venv.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs["frame"]

    def step_async(self, actions: np.ndarray) -> None:
        '''
        Checks if chosen actions are within the underlying venv's valid move list.
        Replaces any that do not with no-op.
        '''
        actions = np.array([
            (move if move < self.valid_moves else self.no_move_idx,
            attack if attack < self.valid_attacks else self.no_attack_idx)
            for move, attack in actions
        ])
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs["frame"], reward, done, info

class VecEnvDiscreteTransferActionWrapper(VecEnvWrapper):
    '''
    VectorEnv Discrete Transfer Wrapper
    Necessary for wrapping SubprocVecEnvs during distributed training
    '''
    def __init__(self, venv: VecEnv, no_op_idx: int = 0):
        super().__init__(venv=venv)
        self.valid_actions = venv.action_space.n
        self.action_space = gym.spaces.Discrete(27) # 27 actions is the largest possible action space (MvC)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
        if hasattr(venv, "image_space_keys"):
            self.image_space_keys = venv.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs["frame"]

    def step_async(self, actions: np.ndarray) -> None:
        '''
        Checks if chosen actions are within the underlying venv's valid move list.
        Replaces any that do not with no-op.
        '''
        actions = np.array([
            action if action < self.valid_actions else self.no_op_idx
            for action in actions
        ])
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs["frame"], reward, done, info

class MDTransferActionWrapper(gym.Wrapper):
    def __init__(self, env, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.valid_moves = env.action_space.nvec[0]
        self.valid_attacks = env.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 19]) # 19 actions is the largest possible action space (MvC)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
        if hasattr(env, "image_space_keys"):
            self.image_space_keys = env.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs["frame"]

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
            trunc = False
        else:
            obs, reward, done, trunc, info = step_result
        return obs["frame"], reward, done, trunc, info

class DiscreteTransferActionWrapper(gym.Wrapper):
    def __init__(self, env, no_op_idx: int = 0):
        super().__init__(env)
        self.valid_actions = env.action_space.n
        self.action_space = gym.spaces.Discrete(27) # 19 actions is the largest possible action space (MvC)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
        if hasattr(env, "image_space_keys"):
            self.image_space_keys = env.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["frame"], info

    def step(self, action):
        '''
        Checks if the chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        action = action if action < self.valid_actions else self.no_op_idx
        step_result = self.env.step(action)
        # Unpack the step result depending on the API.
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            trunc = False
        else:
            obs, reward, done, trunc, info = step_result
        return obs["frame"], reward, done, trunc, info

class MDPlayWrapper(gym.Wrapper):
    def __init__(self, env, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.valid_moves = env.action_space.nvec[0]
        self.valid_attacks = env.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 19]) # 19 actions is the largest possible action space (MvC)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        if hasattr(env, "image_space_keys"):
            self.image_space_keys = env.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs["frame"]

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
            trunc = False
        else:
            obs, reward, done, trunc, info = step_result
        return obs["frame"], reward, done, trunc, info

class DiscretePlayWrapper(gym.Wrapper):
    def __init__(self, env, no_op_idx: int = 0):
        super().__init__(env)
        self.valid_actions = env.action_space.n
        self.action_space = gym.spaces.Discrete(27) # 19 actions is the largest possible action space (MvC)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        if hasattr(env, "image_space_keys"):
            self.image_space_keys = env.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["frame"], info

    def step(self, action):
        '''
        Checks if the chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        action = action if action < self.valid_actions else self.no_op_idx
        step_result = self.env.step(action)
        # Unpack the step result depending on the API.
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            trunc = False
        else:
            obs, reward, done, trunc, info = step_result
        return obs["frame"], reward, done, trunc, info