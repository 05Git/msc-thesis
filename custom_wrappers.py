import gymnasium as gym
import numpy as np
from diambra.arena import Roles
from typing import Optional, Union


class PixelObsWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames: int = 4):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack_frames), np.uint8)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["frame"], info
    
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        return obs["frame"], reward, terminated, truncated, info


class ActionWrapper1P(gym.Wrapper):
    def __init__(
        self,
        env,
        action_space: str = "multi_discrete",
        no_op: Union[int, list[int]] = 0,
        max_actions: Union[int, list[int]] = [9, 11]
    ):
        super().__init__(env)
        self.env = env
        if action_space == "multi_discrete":
            self.valid_actions = env.action_space.nvec
            self.action_space = gym.spaces.MultiDiscrete(max_actions)
        elif action_space == "discrete":
            self.valid_actions = env.action_space.n
            self.action_space = gym.spaces.Discrete(max_actions)
        else:
            raise Exception(f"Invalid action_space input argument: '{action_space}'\nValid arguments: 'discrete', 'multi_discrete'")
        self.no_op = no_op if type(no_op) == list else [no_op for _ in range(len(self.valid_actions))]
        assert len(self.no_op) == len(self.valid_actions)
        for no_op_act, valid_act in zip(self.no_op, self.valid_actions):
            assert no_op_act < valid_act
    
    def step(self, action):
        for idx in range(len(action)):
            action[idx] = action[idx] if action[idx] < self.valid_actions[idx] else self.no_op[idx]
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated, truncated, info


class ActionWrapper2P(gym.Wrapper):
    def __init__(
        self,
        env,
        action_space: str = "multi_discrete",
        no_op: Union[int, list[int]] = 0,
        max_actions: Union[int, list[int]] = [9, 11],
        opp_type: str = "no_op"
    ):
        super().__init__(env)
        self.env = env
        assert env.action_space["agent_0"] == env.action_space["agent_1"]
        if action_space == "multi_discrete":
            self.valid_actions = env.action_space["agent_0"].nvec
            self.action_space = gym.spaces.MultiDiscrete(max_actions)
        elif action_space == "discrete":
            self.valid_actions = env.action_space["agent_0"].n
            self.action_space = gym.spaces.Discrete(max_actions)
        else:
            raise Exception(f"Invalid action_space input argument: '{action_space}'\nValid arguments: 'discrete', 'multi_discrete'")
        self.no_op = no_op if type(no_op) == list else [no_op for _ in range(len(self.valid_actions))]
        assert len(self.no_op) == len(self.valid_actions)
        for no_op_act, valid_act in zip(self.no_op, self.valid_actions):
            assert no_op_act < valid_act
        assert opp_type in ["no_op", "random"]
        self.opp_type = opp_type
    
    def step(self, action):
        p1_actions = action[:len(self.valid_actions)]
        if self.opp_type == "no_op":
            non_agent_action = self.no_op
        elif self.opp_type == "random":
            non_agent_action = [np.random.randint(0, x) for x in self.valid_actions]
        p2_actions = action[len(self.valid_actions):] if len(action) > len(self.valid_actions) else non_agent_action
        for idx in range(len(p1_actions)):
            p1_actions[idx] = p1_actions[idx] if p1_actions[idx] < self.valid_actions[idx] else self.no_op[idx]
        for idx in range(len(p2_actions)):
            p2_actions[idx] = p2_actions[idx] if p2_actions[idx] < self.valid_actions[idx] else self.no_op[idx]
        step_result = self.env.step({"agent_0": p1_actions, "agent_1": p2_actions})
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated, truncated, info


class InterleavingWrapper(gym.Wrapper):
    def __init__(self, env, character_list: list[str]):
        super().__init__(env)
        self.character_list = character_list
        self.character_queue = list(np.random.permutation(self.character_list))

    def reset(self, **kwargs):
        next_character = str(self.character_queue.pop())
        if not len(self.character_queue) > 0:
            self.character_queue = list(np.random.permutation(self.character_list))
        episode_settings = kwargs.get("options", {})
        if episode_settings is None:
            episode_settings = {}
        episode_settings.update({"characters": next_character})
        kwargs["options"] = episode_settings
        obs, info = self.env.reset(**kwargs)
        return obs, info
        

class DefTrainWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.health_key = "P1_health"
        self.last_health_value = None
        self.roles = [Roles.P1, Roles.P2]

    def reset(self, **kwargs):
        episode_roles = tuple(np.random.permutation(self.roles))
        if episode_roles[0] == Roles.P1: # Assumes policy is agent_0 (always seems to be)
            self.health_key = "P1_health"
        else:
            self.health_key = "P2_health"
        episode_settings = kwargs.get("options", {})
        if episode_settings is None:
            episode_settings = {}
        episode_settings.update({"role": episode_roles})
        kwargs["options"] = episode_settings
        obs, info = self.env.reset(**kwargs)
        self.last_health_value = obs[self.health_key][0]
        return obs, info
    
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        new_health_value = obs[self.health_key][0]
        reward = new_health_value - self.last_health_value
        self.last_health_value = new_health_value
        return obs, reward, terminated, truncated, info


class AttTrainWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.health_key = "P2_health"
        self.last_health_value = None
        self.roles = [Roles.P1, Roles.P2]

    def reset(self, **kwargs):
        episode_roles = tuple(np.random.permutation(self.roles))
        if episode_roles[0] == Roles.P1: # Assumes policy is agent_0 (always seems to be)
            self.health_key = "P2_health"
        else:
            self.health_key = "P1_health"
        episode_settings = kwargs.get("options", {})
        if episode_settings is None:
            episode_settings = {}
        episode_settings.update({"role": episode_roles})
        kwargs["options"] = episode_settings
        obs, info = self.env.reset(**kwargs)
        self.last_health_value = obs[self.health_key][0]
        return obs, info
    
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        new_health_value = obs[self.health_key][0]
        reward = self.last_health_value - new_health_value
        self.last_health_value = new_health_value
        return obs, reward, terminated, truncated, info
