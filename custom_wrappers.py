"""
custom_wrappers.py: Custom wrappers to use for training and evaluating policies.
"""
import gymnasium as gym
import numpy as np

from diambra.arena import Roles
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from typing import Optional, Union
from collections import OrderedDict


class JumpBonus(gym.Wrapper):
    """
    Add a small bonus for jumping.
    Requires multi_discrete action space.
    """
    def __init__(self, env: gym.Env, jump_bonus: float = 1e-4):
        super().__init__(env)
        self.jump_bonus = jump_bonus
    
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
        if action[1] in [2,3,4]:
            reward += self.jump_bonus
        return obs, reward, terminated, truncated, info
    

class TwoPTrainWrapper(gym.ActionWrapper):
    """
    SB3 policies don't work with dict action spaces, so need to trick it into thinking
    a 2 player env's action space is a 1 player action space.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = env.action_space["agent_0"]
    
    def action(self, action):
        return action


class TeacherInputWrapper(gym.Wrapper):
    """
    Add teacher's suggested actions as observation data.

    :param env: Environment for student to interact with.
    :param teachers: (dict) Dictionary of teachers organised by ID.
    :param timesteps: (int) Number of timesteps to anneal epsilon over, if using teacher actions.
    :param deterministic: (bool) Whether teachers should use a deterministic or stochastic policy.
    :param teacher_action_space: (str) Whether the teachers use a discrete or multi-discrete action_space.
    :param use_teacher_action: (bool) Whether to use the teacher's recommended actions or not.
    :param initial_epsilon: (float) Initial teacher-action epsilon value.
    """
    def __init__(
        self,
        env: gym.Env,
        teachers: dict[str: OnPolicyAlgorithm],
        timesteps: int = 100_000,
        deterministic: bool = True,
        teacher_action_space: str = "multi_discrete",
        use_teacher_actions: bool = False,
        initial_epsilon: float = 1.,
    ):
        super().__init__(env)
        self.teachers = teachers
        self.deterministic = deterministic
        if teacher_action_space == "multi_discrete":
            action_space_shape = env.action_space.nvec
        elif teacher_action_space == "discrete":
            action_space_shape = [env.action_space.n]
        else:
            raise Exception(f"Invalid action_space input argument: '{teacher_action_space}'\nValid arguments: 'discrete', 'multi_discrete'")
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Dict({
                "image": env.observation_space,
                **{id: gym.spaces.Box(
                        low=np.zeros_like(action_space_shape),
                        high=np.array(action_space_shape),
                        dtype=np.uint8
                    ) for id in teachers.keys()
                }
            })
        elif isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
            for id in teachers.keys():
                self.observation_space[id] = gym.spaces.Box(
                    low=np.zeros_like(action_space_shape),
                    high=np.array(action_space_shape),
                    dtype=np.uint8
                )
        else:
            raise Exception("Base env observation space must be Box or Dict")
        
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
            # Anneal epsilon if it's greater than 0
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
        # Add teachers' recommended actions to vector observations
        teacher_actions = {}
        teacher_observation = observation["image"] if isinstance(observation, dict) else observation
        for id, teacher in self.teachers.items():
            action, _ = teacher.predict(teacher_observation, deterministic=self.deterministic)
            teacher_actions[id] = action
        self.teacher_actions = teacher_actions
        if type(observation) == dict:
            observation.update(teacher_actions)
        else:
            observation = {"image": observation, **teacher_actions}
        return observation
    
    def action(self, action):
        # Replace the student's action with a random teacher's action with probability (epsilon)
        epsilon = self.progress * self.initial_epsilon
        teacher_id = np.random.choice(list(self.teachers.keys()))
        p = np.random.rand()
        if p > epsilon:
            return action
        else:
            return self.teacher_actions[teacher_id]


class PixelObsWrapper(gym.ObservationWrapper):
    """
    Filters out non-image observations from a Dict space.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, env.observation_space["frame"].shape, np.uint8)
    
    def observation(self, observation):
        return observation["frame"]
    

class AddLastActions(gym.Wrapper):
    """
    Adds the last N actions to the observation history.
    Can also be used to penalise repetitive actions.
    """
    def __init__(
        self,
        env: gym.Env,
        action_space: str = "multi_discrete",
        action_history_len: int = 1,
        use_similarity_penalty: bool = False,
        similarity_penalty_alpha: float = 1e-3,
    ):
        super().__init__(env)
        if action_space == "multi_discrete":
            action_space_shape = env.action_space.nvec
        elif action_space == "discrete":
            action_space_shape = [env.action_space.n]
        else:
            raise Exception(f"Invalid action_space input argument: '{action_space}'\nValid arguments: 'discrete', 'multi_discrete'")
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Dict({
                "image": env.observation_space,
                "last_actions": gym.spaces.Box(
                    low=np.array([np.zeros_like(action_space_shape)] * action_history_len),
                    high=np.array(np.array([action_space_shape] * action_history_len)),
                    dtype=np.uint8,
                )
            })
        elif isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
            self.observation_space["last_actions"] = gym.spaces.Box(
                low=np.array([np.zeros_like(action_space_shape)] * action_history_len),
                high=np.array(np.array([action_space_shape] * action_history_len)),
                dtype=np.uint8,
            )
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
        self.last_actions = self.last_actions[1:]
        self.last_actions = np.concatenate((self.last_actions, [action]))
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
        actionsT = self.last_actions.T # Transpose last_actions vector to align action types with each other
        for prev_actions in actionsT:
            unique_actions = np.array(list(set(prev_actions)))
            penalty -= (actionsT.shape[0] - unique_actions.shape[0]) * self.similarity_penalty_alpha
        penalty /= actionsT.shape[0] # Normalise penalty by number of performable actions
        return reward - penalty


class NoOpWrapper(gym.ActionWrapper):
    """
    Replaces an action with 0 if it's the same as the previous frame's action.
    Necessary for evaluating deterministic policies that have collapsed to spamming one move repeatedly.
    """
    def __init__(
        self,
        env: gym.Env,
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


class ActionMaskWrapper(gym.ActionWrapper):
    """
    Mask any invalid actions in a base action space with no-op.
    Allows for transfer between games where one game has a larger action space.
    Action space must be chosen before training.
    """
    def __init__(
        self,
        env: gym.Env,
        max_actions: Union[int, list[int]],
        num_players: int = 1,
        action_space: str = "multi_discrete",
        no_op: Union[int, list[int]] = 0,
    ):
        super().__init__(env)
        assert num_players in [1,2]
        self.num_players = num_players

        assert action_space in ["discrete", "multi_discrete"]
        self.max_actions = max_actions
        self.action_space_shape = len(max_actions) if type(max_actions) == list else max_actions

        self.no_op = no_op if type(no_op) == list else [no_op for _ in range(self.action_space_shape)]
        assert len(self.no_op) == self.action_space_shape

        if num_players == 2:
            assert env.action_space["agent_0"] == env.action_space["agent_1"]
            if action_space == "multi_discrete":
                self.valid_actions = {
                    "agent_0": env.action_space["agent_0"].nvec,
                    "agent_1": env.action_space["agent_1"].nvec,
                }
                self.action_space = gym.spaces.Dict({
                    "agent_0": gym.spaces.MultiDiscrete(max_actions),
                    "agent_1": gym.spaces.MultiDiscrete(max_actions),
                })
            else:
                self.valid_actions = {
                    "agent_0": env.action_space["agent_0"].n,
                    "agent_1": env.action_space["agent_1"].n,
                }
                self.action_space = gym.spaces.Dict({
                    "agent_0": gym.spaces.Discrete(max_actions),
                    "agent_1": gym.spaces.Discrete(max_actions),
                })

            for max_act, valid_act in zip(max_actions, self.valid_actions["agent_0"]):
                assert valid_act <= max_act

            for no_op_act, valid_act in zip(self.no_op, self.valid_actions["agent_0"]):
                assert no_op_act < valid_act
        else:        
            if action_space == "multi_discrete":
                self.valid_actions = env.action_space.nvec
                self.action_space = gym.spaces.MultiDiscrete(max_actions)
            elif action_space == "discrete":
                self.valid_actions = env.action_space.n
                self.action_space = gym.spaces.Discrete(max_actions)

            for max_act, valid_act in zip(max_actions, self.valid_actions):
                assert valid_act <= max_act

            for no_op_act, valid_act in zip(self.no_op, self.valid_actions):
                assert no_op_act < valid_act

    def action(self, action):
        if self.num_players == 1:
            for idx in range(self.action_space_shape):
                action[idx] = action[idx] if action[idx] < self.valid_actions[idx] else self.no_op[idx]
        else:
            for idx in range(self.action_space_shape):
                action["agent_0"][idx] = action["agent_0"][idx] if action["agent_0"][idx] < self.valid_actions[idx] else self.no_op[idx]
                action["agent_1"][idx] = action["agent_1"][idx] if action["agent_1"][idx] < self.valid_actions[idx] else self.no_op[idx]

        return action


class OpponentController(gym.ActionWrapper):
    """
    Controls an enemy agent in a 2 player env to perform specific behaviours.
    """
    def __init__(self, env: gym.Env, opp_type: str):
        super().__init__(env)
        assert env.action_space["agent_0"] == env.action_space["agent_1"]
        assert opp_type in ["no_op", "rand", "jump"]
        self.opp_type = opp_type
        self.act_counter = 20
        self.last_move = np.random.choice([2,3,4])
        if type(env.action_space["agent_0"]) == gym.spaces.Discrete:
            self.action_space_shape = env.action_space["agent_0"].n
        else:
            self.action_space_shape = env.action_space["agent_0"].shape[0]
    
    def action(self, action):
        agent_action = action["agent_0"] if type(action) == OrderedDict else action
        if self.opp_type == "no_op":
            non_agent_action = [0 for _ in range(self.action_space_shape)]
        elif self.opp_type == "rand":
            non_agent_action = self.env.action_space.sample()["agent_1"]
        elif self.opp_type == "jump":
            self.act_counter = (self.act_counter - 1) if self.act_counter > 0 else 20
            move = self.last_move if self.act_counter != 20 else np.random.choice([2,3,4])
            self.last_move = move
            if type(self.env.action_space["agent_1"]) == gym.spaces.MultiDiscrete:
                attack = np.random.randint(1, self.env.action_space["agent_1"].nvec[1]) if self.act_counter == 0 else 0
                non_agent_action = [move, attack]
            elif type(self.env.action_space["agent_1"]) == gym.spaces.Discrete:
                non_agent_action = [np.random.randint(9, self.env.action_space["agent_1"].n) if self.act_counter == 0 else move]

        return {"agent_0": agent_action, "agent_1": non_agent_action}


class InterleavingWrapper(gym.Wrapper):
    """
    Randomly chooses a character from a given list at the start of each new episode.
    """
    def __init__(self, env: gym.Env, character_list: list[str], one_p_env: bool):
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
    """
    Reward shaping: Agent is encouraged to maintain its health as long as possible.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.health_key = "P1_health"
        self.last_health_value = None
        self.roles = [Roles.P1, Roles.P2]
        self.timer_key = "timer" if "timer" in env.observation_space.spaces.keys() else None
        self.timer_max = env.observation_space["timer"].high if "timer" in env.observation_space.spaces.keys() else None
        self.timer_min = env.observation_space["timer"].low if "timer" in env.observation_space.spaces.keys() else None

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
        if self.timer_key is not None:
            reward += ((self.timer_max - obs[self.timer_key][0]) / (self.timer_max - self.timer_min)) * 1e-2
        return obs, reward.squeeze(), terminated, truncated, info


class AttTrainWrapper(gym.Wrapper):
    """
    Reward shaping: agent is encouraged to reduce opponent's health as fast as possible.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.health_key = "P2_health"
        self.last_health_value = None
        self.roles = [Roles.P1, Roles.P2]
        self.timer_key = "timer" if "timer" in env.observation_space.spaces.keys() else None
        self.timer_max = env.observation_space["timer"].high if "timer" in env.observation_space.spaces.keys() else None
        self.timer_min = env.observation_space["timer"].low if "timer" in env.observation_space.spaces.keys() else None

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
        if self.timer_key is not None:
            reward -= ((self.timer_max - obs[self.timer_key][0]) / (self.timer_max - self.timer_min)) * 1e-2
        return obs, reward.squeeze(), terminated, truncated, info
    