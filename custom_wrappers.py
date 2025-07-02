import gymnasium as gym
import numpy as np

from diambra.arena import Roles
from stable_baselines3 import PPO
from typing import Optional, Union


class TeacherInputWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        teachers: list[PPO],
        timesteps: int,
        deterministic: bool = True,
        teacher_action_space: str = "multi_discrete",
        use_teacher_actions: bool = False,
        initial_epsilon: float = 1.,
    ):
        super().__init__(env)
        self.env = env
        self.teachers = teachers
        self.deterministic = deterministic
        if teacher_action_space == "multi_discrete":
            action_space_shape = env.action_space.nvec
        elif teacher_action_space == "discrete":
            action_space_shape = env.action_space.n
        else:
            raise Exception(f"Invalid action_space input argument: '{teacher_action_space}'\nValid arguments: 'discrete', 'multi_discrete'")
        self.observation_space = gym.spaces.Dict({
            "image": env.observation_space,
            "teacher_actions": gym.spaces.Box(
                low=np.array([np.zeros_like(action_space_shape)] * len(teachers)).reshape(len(teachers), len(action_space_shape)),
                high=np.array([action_space_shape] * len(teachers)).reshape(len(teachers), len(action_space_shape)),
                dtype=np.uint8,
            )
        })
        assert initial_epsilon >= 0 and initial_epsilon <= 1
        self.use_teacher_actions = use_teacher_actions
        self.teacher_actions = None
        self.timesteps = timesteps
        self.current_step = 0
        self.progress = 1
        self.initial_epsilon = initial_epsilon

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def step(self, action):
        self.current_step += 1
        if self.progress > 0:
            self.progress -= self.current_step / self.timesteps
        action = self.action(action) if self.use_teacher_actions else action
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, observation):
        teacher_actions = []
        for teacher in self.teachers:
            action, _ = teacher.predict(observation, deterministic=self.deterministic)
            teacher_actions.append(action)
        teacher_actions = np.array(teacher_actions)
        self.teacher_actions = teacher_actions
        return {"image": observation, "teacher_actions": teacher_actions}
    
    def action(self, action):
        epsilon = self.progress * self.initial_epsilon
        teacher_action_idx = np.random.choice(range(len(self.teacher_actions)))
        p = np.random.rand()
        if p > epsilon:
            return action
        else:
            return self.teacher_actions[teacher_action_idx]


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
    

class AddLastActions(gym.Wrapper):
    def __init__(
        self,
        env,
        action_space: str = "multi_discrete",
        action_history_len: int = 1,
        use_similarity_penalty: bool = False,
        similarity_penalty_alpha: float = 1e-3,
    ):
        super().__init__(env)
        self.env = env
        if action_space == "multi_discrete":
            action_space_shape = env.action_space.nvec
        elif action_space == "discrete":
            action_space_shape = env.action_space.n
        else:
            raise Exception(f"Invalid action_space input argument: '{action_space}'\nValid arguments: 'discrete', 'multi_discrete'")
        if type(env.observation_space) == gym.spaces.Box:
            self.observation_space = gym.spaces.Dict({
                "image": env.observation_space,
                "last_actions": gym.spaces.Box(
                    low=np.array([np.zeros_like(action_space_shape, dtype=np.uint8)] * action_history_len),
                    high=np.array(np.array([action_space_shape]) * action_history_len),
                    dtype=np.uint8,
                )
            })
        elif type(env.observation_space) == gym.space.Dict:
            self.observation_space = env.observation_space
            self.observation_space.update({
                "last_actions": gym.spaces.Box(
                    low=np.array([np.zeros_like(action_space_shape, dtype=np.uint8)] * action_history_len),
                    high=np.array(np.array([action_space_shape]) * action_history_len),
                    dtype=np.uint8,
                )
            })
        else:
            raise Exception("Base env observation space must be Box or Dict")
        self.last_actions = np.array([np.zeros_like(action_space_shape, dtype=np.uint8)] * action_history_len)
        self.use_similarity_penalty = use_similarity_penalty
        self.similarity_penalty_alpha = similarity_penalty_alpha
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        self.last_actions.append(action)
        self.last_actions.pop(0)
        if self.use_similarity_penalty:
            reward = self.penalty(reward)
        return self.observation(obs), reward, terminated, truncated, info
    
    def observation(self, observation):
        if type(observation) == dict:
            observation.update({"last_actions": self.last_actions})
        else:
            observation = {"image": observation, "last_actions": self.last_actions}
        return observation
    
    def penalty(self, reward):
        penalty = 0
        actionsT = self.last_actions.T
        for prev_actions in actionsT:
            unique_actions = np.array(list(set(prev_actions)))
            penalty -= (actionsT.shape[0] - unique_actions.shape[0]) * self.similarity_penalty_alpha
        penalty /= actionsT.shape[0]
        return reward - penalty


class NoOpWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env,
        no_attack: int = 0,
        action_space_type: str = "multi_discrete",
    ):
        super().__init__(env)
        self.last_attack = None
        self.no_attack = no_attack
        assert action_space_type in ["discrete", "multi_discrete"]
        self.action_space_type = action_space_type
    
    def action(self, action):
        attack = action[1] if self.action_space_type == "multi_discrete" else action
        if attack == self.last_attack:
            attack = self.no_attack
            if self.action_space_type == "multi_discrete":
                action[1] = attack
            else:
                action = attack
        self.last_attack = attack
        return action

class ActionWrapper1P(gym.Wrapper):
    def __init__(
        self,
        env,
        action_space: str = "multi_discrete",
        no_op: Union[int, list[int]] = 0,
        max_actions: Union[int, list[int]] = [9,11]
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
        for max_act, valid_act in zip(max_actions, self.valid_actions):
            assert valid_act <= max_act
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
        max_actions: Union[int, list[int]] = [9,11],
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
        for max_act, valid_act in zip(max_actions, self.valid_actions):
            assert valid_act <= max_act
        self.no_op = no_op if type(no_op) == list else [no_op for _ in range(len(self.valid_actions))]
        assert len(self.no_op) == len(self.valid_actions)
        for no_op_act, valid_act in zip(self.no_op, self.valid_actions):
            assert no_op_act < valid_act
        assert opp_type in ["no_op", "random", "jump"]
        self.opp_type = opp_type
        self.act_counter = 20
        self.last_move = np.random.choice([1,2,3])
    
    def step(self, action):
        p1_actions = action[:len(self.valid_actions)]
        if self.opp_type == "no_op":
            non_agent_action = self.no_op
        elif self.opp_type == "random":
            non_agent_action = [np.random.randint(0, x) for x in self.valid_actions]
        elif self.opp_type == "jump":
            #TODO: implement discrete version
            self.act_counter = (self.act_counter - 1) if self.act_counter > 0 else 20
            move = self.last_move if self.act_counter != 20 else np.random.choice([1,2,3])
            self.last_move = move
            attack = np.random.randint(1, 6) if self.act_counter == 0 else self.no_op[1]
            non_agent_action = [move, attack]
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
    def __init__(self, env, character_list: list[str], one_p_env: bool):
        super().__init__(env)
        self.character_list = character_list
        self.character_queue = list(np.random.permutation(self.character_list))
        self.one_p_env = one_p_env

    def reset(self, **kwargs):
        next_character = str(self.character_queue.pop())
        if not len(self.character_queue) > 0:
            self.character_queue = list(np.random.permutation(self.character_list))
        next_character = next_character.translate({ord(i): None for i in "'[]"})
        next_character = tuple(next_character.split())
        if len(next_character) == 2 and self.one_p_env:
            next_character = " ".join(next_character)
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
