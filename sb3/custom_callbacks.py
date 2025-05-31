from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import warnings
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


'''
Extended version of evaluate_policy() which tracks custom metrics
Original code from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py
'''
def evaluate_policy_with_arcade_metrics(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    #############################################################################
    # Custom Metrics:                                                           #
    # stages_completed: Number of arcade stages completed in each env           #
    # arcade_runs_completed: Number of full arcade runs completed across envs   #
    #############################################################################
    stages_completed = np.zeros(n_envs, dtype=np.float64)
    arcade_runs_completed = np.zeros(n_envs, dtype=np.float64)

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())
                
                ###################################################################
                # Increment for each stage completed in an episode
                if info["stage_done"]:
                    stages_completed[i] += 1
                    # Increment evey time the final stage is successfully completed
                    if done:
                        arcade_runs_completed[i] += 1
                
                ###################################################################

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    ####################################################################
    
    # Normalize by episode count targets before calculating mean and std
    stages_completed /= episode_count_targets
    mean_stages = np.mean(stages_completed)
    std_stages = np.std(stages_completed)

    arcade_runs_completed /= episode_count_targets
    mean_arcade_runs = np.mean(arcade_runs_completed)
    std_arcade_runs = np.std(arcade_runs_completed)

    ####################################################################

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    
    # Return custom metrics alongside reward info
    if return_episode_rewards:
        return episode_rewards, episode_lengths, stages_completed, arcade_runs_completed
    return (mean_reward, std_reward), (mean_stages, std_stages), (mean_arcade_runs, std_arcade_runs)


class DiambraEvalCallback(BaseCallback):
    """
    Code extended from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L30
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.total_stages_completed = 0
        self.total_episodes_completed = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.stages_completed = np.zeros(self.training_env.num_envs, dtype=np.uint32)
        self.arcade_runs_completed = np.zeros(self.training_env.num_envs, dtype=bool)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        infos = self.locals['infos']
        dones = self.locals['dones']
        
        for idx, info in enumerate(infos):
            if info["stage_done"]:
                self.stages_completed[idx] += 1

            if dones[idx]:
                self.arcade_runs_completed[idx] = True if info["stage_done"] else False
                self.logger.record("custom/arcade_runs_completed", self.arcade_runs_completed.sum())

                self.total_episodes_completed += 1
                self.total_stages_completed += self.stages_completed[idx]
                self.stages_completed[idx] = 0
        
        # Log running average
        if self.total_episodes_completed > 0:
            avg_stages = self.total_stages_completed / self.total_episodes_completed
            self.logger.record("custom/avg_stages_completed", avg_stages)

        return True
    