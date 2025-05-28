import gymnasium as gym
from ray.rllib.env.vector_env import VectorEnv, VectorEnvWrapper
import numpy as np

class VecEnvMDTransferWrapper(VectorEnvWrapper):
    '''
    VectorEnv Multi-Discrete Transfer Wrapper
    Necessary for wrapping SubprocVecEnvs during distributed training
    '''
    def __init__(self, venv: VectorEnv, stack_frames: int, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            num_envs=venv.num_envs
        )
        self.venv = venv
        self.valid_moves = venv.action_space["agent_0"].nvec[0]
        self.valid_attacks = venv.action_space["agent_0"].nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 11]) # 11 attacks is the largest possible action space (kof)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        if hasattr(venv, "image_space_keys"):
            self.image_space_keys = venv.unwrapped.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def vector_reset(self):
        obs = self.venv.vector_reset()
        return obs["frame"]
    
    def reset_at(self, index):
        obs = self.venv.reset_at(index)
        return obs["frame"]

    def vector_step(self, actions):
        obs, reward, dones, info = self.venv.vector_step(actions)
        return obs["frame"], reward, dones, info

class VecEnvDiscreteTransferWrapper(VectorEnvWrapper):
    '''
    VectorEnv Discrete Transfer Wrapper
    Necessary for wrapping SubprocVecEnvs during distributed training
    '''
    def __init__(self, venv: VectorEnv, stack_frames: int, no_op_idx: int = 0):
        super().__init__(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            num_envs=venv.num_envs
        )
        self.venv = venv
        self.valid_actions = venv.action_space["agent_0"].n
        self.action_space = gym.spaces.Discrete(19) # 19 actions is the largest possible action space (kof)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        if hasattr(venv, "image_space_keys"):
            self.image_space_keys = venv.unwrapped.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def vector_reset(self):
        obs = self.venv.vector_reset()
        return obs["frame"]
    
    def reset_at(self, index):
        obs = self.venv.reset_at(index)
        return obs["frame"]

    def vector_step(self, actions):
        obs, reward, dones, info = self.venv.vector_step(actions)
        return obs["frame"], reward, dones, info

class MDTransferWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames: int, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.valid_moves = env.action_space.nvec[0]
        self.valid_attacks = env.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 11]) # 11 actions is the largest possible action space (kof)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        if hasattr(env, "image_space_keys"):
            self.image_space_keys = env.unwrapped.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["frame"], info

    def step(self, action):
        '''
        Checks if a chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        move_idx, attack_idx = action
        move = move_idx if move_idx < self.valid_moves else self.no_move_idx
        attack = attack_idx if attack_idx < self.valid_attacks else self.no_attack_idx
        step_result = self.env.step([move, attack])
        # Unpack the step result depending on the API.
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            trunc = False
        else:
            obs, reward, done, trunc, info = step_result
        return obs["frame"], reward, done, trunc, info

class DiscreteTransferWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames: int, no_op_idx: int = 0):
        super().__init__(env)
        self.valid_actions = env.action_space["agent_0"].n
        self.action_space = gym.spaces.Discrete(19) # 19 actions is the largest possible action space (kof)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        if hasattr(env, "image_space_keys"):
            self.image_space_keys = env.unwrapped.image_space_keys
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
