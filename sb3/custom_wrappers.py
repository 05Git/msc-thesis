import gymnasium as gym
from gymnasium.spaces import Dict
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
import numpy as np
from diambra.arena import Roles

class VecEnvMDTransferWrapper(VecEnvWrapper):
    '''
    VectorEnv Multi-Discrete Transfer Wrapper
    Necessary for wrapping SubprocVecEnvs during distributed training
    '''
    def __init__(self, venv: VecEnv, stack_frames: int, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(venv=venv)
        self.venv = venv
        if type(venv.action_space) == Dict:
            self.valid_moves = venv.action_space["agent_0"].nvec[0]
            self.valid_attacks = venv.action_space["agent_0"].nvec[1]
        else:
            self.valid_moves = venv.action_space.nvec[0]
            self.valid_attacks = venv.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 11]) # 11 attacks is the largest possible action space (kof)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        if hasattr(venv, "image_space_keys"):
            self.image_space_keys = venv.unwrapped.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs[self.image_space_keys[0]]

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
        obs, reward, dones, info = self.venv.step_wait()
        return obs[self.image_space_keys[0]], reward, dones, info

class VecEnvDiscreteTransferWrapper(VecEnvWrapper):
    '''
    VectorEnv Discrete Transfer Wrapper
    Necessary for wrapping SubprocVecEnvs during distributed training
    '''
    def __init__(self, venv: VecEnv, stack_frames: int, no_op_idx: int = 0):
        super().__init__(venv=venv)
        self.venv = venv
        if type(venv.action_space) == Dict:
            self.valid_actions = venv.action_space["agent_0"].n
        else:
            self.valid_actions = venv.action_space.n
        self.action_space = gym.spaces.Discrete(19) # 19 actions is the largest possible action space (kof)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        if hasattr(venv, "image_space_keys"):
            self.image_space_keys = venv.unwrapped.image_space_keys
        else:
            self.image_space_keys = ["frame"]

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs[self.image_space_keys[0]]

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
        return obs[self.image_space_keys[0]], reward, done, info

class MDTransferWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames: int, characters: list[str], no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.env = env
        self.valid_moves = env.action_space.nvec[0]
        self.valid_attacks = env.action_space.nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 11]) # 11 actions is the largest possible action space (kof)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        self.characters = characters
        self.char_queue = list(np.random.permutation(self.characters))
        try:
            self.image_space_keys = list(getattr(env.unwrapped, "image_space_keys", ["frame"]))
        except Exception:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        # Making this work with tuples of characters for KOF -_-
        next_characters = str(self.char_queue.pop(0)).split()
        next_characters = tuple(s.translate({ord(i): None for i in "[]'"}) for s in next_characters)
        if len(next_characters) == 2: # Some characters in MK3 have two names instead of just one, ugh
            next_characters = " ".join(next_characters)
        episode_settings = {
            "characters" : next_characters
        }
        if not len(self.char_queue) > 0:
            self.char_queue = list(np.random.permutation(self.characters))
        obs, info = self.env.reset(options=episode_settings, **kwargs)
        return obs[self.image_space_keys[0]], info

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
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        return obs[self.image_space_keys[0]], reward, terminated, truncated, info

class DiscreteTransferWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames: int, characters: list[str], no_op_idx: int = 0):
        super().__init__(env)
        self.env = env
        self.valid_actions = env.action_space.n
        self.action_space = gym.spaces.Discrete(19) # 19 actions is the largest possible action space (kof)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        self.characters = characters
        self.char_queue = list(np.random.permutation(self.characters))
        try:
            self.image_space_keys = list(getattr(env.unwrapped, "image_space_keys", ["frame"]))
        except Exception:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        episode_settings = {
            "characters" : str(self.char_queue.pop(0))
        }
        if not len(self.char_queue) > 0:
            self.char_queue = list(np.random.permutation(self.characters))
        obs, info = self.env.reset(options=episode_settings, **kwargs)
        return obs[self.image_space_keys[0]], info

    def step(self, action):
        '''
        Checks if the chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        action = action if action < self.valid_actions else self.no_op_idx
        step_result = self.env.step(action)
        # Unpack the step result depending on the API.
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        return obs[self.image_space_keys[0]], reward, terminated, truncated, info

class MultiAgentMDTransferWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames: int, defense_training: bool = False, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.env = env
        self.valid_moves = env.action_space["agent_0"].nvec[0]
        self.valid_attacks = env.action_space["agent_0"].nvec[1]
        self.action_space = gym.spaces.MultiDiscrete([9, 11]) # 11 actions is the largest possible action space (kof)
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        try:
            self.image_space_keys = list(getattr(env.unwrapped, "image_space_keys", ["frame"]))
        except Exception:
            self.image_space_keys = ["frame"]
        self.defense_training = defense_training
        self.health_keys = ["P1_health"]
        self.last_health_value = None
        self.roles = [Roles.P1, Roles.P2]

    def reset(self, **kwargs):
        episode_roles = tuple(np.random.permutation(self.roles))
        if episode_roles[0] == Roles.P1: # Assumes policy is agent_0 (always seems to be)
            self.health_keys[0] = "P1_health"
        else:
            self.health_keys[0] = "P2_health"
        episode_settings = {
            "role": episode_roles
        }
        obs, info = self.env.reset(options=episode_settings, **kwargs)
        if self.defense_training:
            self.last_health_value = obs[self.health_keys[0]][0]
        return obs[self.image_space_keys[0]], info

    def step(self, action):
        '''
        Checks if a chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        p1_move_idx, p1_attack_idx = action[:2]
        p2_move_idx, p2_attack_idx = action[2:] if len(action) > 2 else [0,0]
        p1_move = p1_move_idx if p1_move_idx < self.valid_moves else self.no_move_idx
        p1_attack = p1_attack_idx if p1_attack_idx < self.valid_attacks else self.no_attack_idx
        p2_move = p2_move_idx if p2_move_idx < self.valid_moves else self.no_move_idx
        p2_attack = p2_attack_idx if p2_attack_idx < self.valid_attacks else self.no_attack_idx
        if self.defense_training:
            p2_move = np.random.randint(0, self.valid_moves)
            p2_attack = np.random.randint(0, self.valid_attacks)
        step_result = self.env.step({"agent_0": [p1_move, p1_attack], "agent_1": [p2_move, p2_attack]})
        # Unpack the step result depending on the API.
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        if self.defense_training:
            new_health_value = obs[self.health_keys[0]][0]
            reward = self.last_health_value - new_health_value # Focus on difference between own health values to encourage defense
            self.last_health_value = new_health_value
        return obs[self.image_space_keys[0]], reward, terminated, truncated, info
    
class MultiAgentDiscreteTransferWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames: int, no_op_idx: int = 0):
        super().__init__(env)
        self.env = env
        self.valid_actions = env.action_space["agent_0"].n
        self.action_space = gym.spaces.Discrete(19) # 19 actions is the largest possible action space (kof)
        self.no_op_idx = no_op_idx
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
        try:
            self.image_space_keys = list(getattr(env.unwrapped, "image_space_keys", ["frame"]))
        except Exception:
            self.image_space_keys = ["frame"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[self.image_space_keys[0]], info

    def step(self, action):
        '''
        Checks if the chosen action is within the underlying env's valid move list.
        If not, replaces it with no-op.
        '''
        try:
            p1_action, p2_action = action
        except:
            p1_action, p2_action = [action, 0]
        p1_action = p1_action if p1_action < self.valid_actions else self.no_op_idx
        p2_action = p2_action if p2_action < self.valid_actions else self.no_op_idx
        step_result = self.env.step({"agent_0": p1_action, "agent_1": p2_action})
        # Unpack the step result depending on the API.
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        return obs[self.image_space_keys[0]], reward, terminated, truncated, info