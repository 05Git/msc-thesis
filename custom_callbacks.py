import numpy as np
import warnings
import os
import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from utils import evaluate_policy_with_arcade_metrics
from typing import Any, Optional, Union


class ArcadeMetricsTrainCallback(BaseCallback):
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
                self.logger.record("rollout/arcade_runs_completed", self.arcade_runs_completed.sum())

                self.total_episodes_completed += 1
                self.total_stages_completed += self.stages_completed[idx]
                self.stages_completed[idx] = 0
        
        # Log running average
        if self.total_episodes_completed > 0:
            avg_stages = self.total_stages_completed / self.total_episodes_completed
            self.logger.record("rollout/avg_stages_completed", avg_stages)

        return True


###################################################################################################################################
# Extended EvalCallback to include custom arcade metrics                                                                          #
# Original code available at: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L341  #
###################################################################################################################################
class ArcadeMetricsEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        episode_num: int | None = None,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        ##########################################################################################
        # Logs will be written in ``evaluations.npz``
        # If more than one training episode (e.g. during sequential training), then
        # episode number is appended to file name to prevent old evaluation data from
        # being overwritten
        if log_path is not None:
            file_name = f"evaluations_{episode_num}" if episode_num is not None else "evaluations"
            log_path = os.path.join(log_path, file_name)
        ##########################################################################################
        self.log_path = log_path
        self.evaluations_results: list[list[float]] = []
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

        #############################################
        # Custom arcade metrics                     #
        self.stages_comp: list[list[float]] = []
        self.arcade_runs_comp: list[list[float]] = []
        #############################################
        
        ###############################################################################################
        # Ensure subsequent episodes don't overwrite best models from previous episodes               #
        self.model_save_name = f"{episode_num}_best_model" if episode_num is not None else "best_model"
        ###############################################################################################

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            ################################################################################################################
            # Use cutom evaluation function to return arcade metrics                                                       #
            episode_rewards, episode_lengths, stages_completed, arcade_runs_completed = evaluate_policy_with_arcade_metrics(
                model=self.model,
                env=self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            ################################################################################################################

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                #####################################################
                # Log custom metrics                                #
                self.stages_comp.append(stages_completed)
                self.arcade_runs_comp.append(arcade_runs_completed)
                #####################################################

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    file=self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    ##################################
                    stages=self.stages_comp,
                    arcade_runs=self.arcade_runs_comp,
                    ##################################
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            #################################################################################################
            # Calculate means and std of arcade metrics                                                     #
            mean_stages, std_stages = np.mean(stages_completed), np.std(stages_completed)
            mean_arcade_runs, std_arcade_runs = np.mean(arcade_runs_completed), np.std(arcade_runs_completed)
            #################################################################################################

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                #################################################################################
                # Print arcade metrics if verbose                                               #
                print(f"Stages completed: {mean_stages:.2f} +/- {std_stages:.2f}")
                print(f"Arcade runs completed: {mean_arcade_runs:.2f} +/- {std_arcade_runs:.2f}")
                #################################################################################
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            #######################################################################
            # Log custom metrics                                                  #
            self.logger.record("eval/mean_stages_completed", mean_stages)
            self.logger.record("eval/mean_arcade_runs_completed", mean_arcade_runs)
            #######################################################################

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    ##############################################################################
                    # Use self.model_save_name instead of just "best_model"                      #
                    self.model.save(os.path.join(self.best_model_save_path, self.model_save_name))
                    ##############################################################################
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class StudentSimilarityCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(StudentSimilarityCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        obs = self.locals["new_obs"]
        student_actions = np.array(self.locals["actions"])

        if isinstance(obs, dict):
            teacher_actions = np.array(obs["teacher_actions"])
        else:
            raise ValueError("Expected observation to be a dict with 'teacher_actions'")
        
        n_teachers = teacher_actions.shape[1]
        action_dim = student_actions.shape[1]
        for i in range(n_teachers):
            teacher_i_actions = teacher_actions[:, i, :]
            
            exact_match = np.all(student_actions == teacher_i_actions, axis=-1)
            exact_match_mean = np.mean(exact_match)
            self.logger.record(f"similarity/teacher_{i}/exact_match", exact_match_mean)

            for a_idx in range(action_dim):
                action_match = (student_actions[:, a_idx] == teacher_i_actions[:, a_idx])
                action_match_mean = np.mean(action_match)

                self.logger.record(f"similarity/teacher_{i}/action_{a_idx}", action_match_mean)

                if self.verbose > 0:
                    print(f"Teacher {i}: Action {a_idx} Similarity = {action_match_mean:.3f}")

        return True


class UniqueActionsCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super(UniqueActionsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        obs = self.locals["new_obs"]
        if isinstance(obs, dict):
            last_actions = np.array(obs["last_actions"])
        else:
            raise ValueError("Expected observation to be a dict with 'last_actions'")
        
        num_unique_actions = np.zeros(last_actions.shape[2], dtype=np.float32)
        for env_actions in last_actions:
            for idx, action_list in enumerate(env_actions.T):
                num_unique_actions[idx] += len(set(action_list))
        num_unique_actions /= last_actions.shape[0]

        for act_idx, unique_actions_i in enumerate(num_unique_actions):
            self.logger.record(f"unique_actions/average_num_unique_actions_idx_{act_idx}", unique_actions_i)
            if self.verbose > 0:
                print(f"Avergae number of unique actions at idx {act_idx}: {unique_actions_i:.4f}")

        return True
