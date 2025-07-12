import gymnasium as gym
import torch as th
import numpy as np

from sb3_distill.core import PolicyDistillationAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common import distributions
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F
from typing import Dict, Type


policy_aliases: Dict[str, Type[BasePolicy]] = {
    "MlpPolicy": ActorCriticPolicy,
    "CnnPolicy": ActorCriticCnnPolicy,
    "MultiInputPolicy": MultiInputActorCriticPolicy,
}

class MultiTeacherStudentDistil(PolicyDistillationAlgorithm, OnPolicyAlgorithm):
    """
    Adjusted version of sb3_distill's StudentDistill policy.
    Original code available at: https://github.com/spiglerg/sb3_distill/blob/main/sb3_distill/student_distill.py
    """
    def __init__(
        self, policy, env, teachers, learning_rate, n_steps, gamma, gae_lambda,
        ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq,
        rollout_buffer_class = None, rollout_buffer_kwargs = None,
        stats_window_size = 100, tensorboard_log = None, monitor_wrapper = True,
        policy_kwargs = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True,
        supported_action_spaces = None
    ):
        super().__init__(
            policy, env, learning_rate, n_steps, gamma, gae_lambda, ent_coef, vf_coef,
            max_grad_norm, use_sde, sde_sample_freq, rollout_buffer_class, rollout_buffer_kwargs,
            stats_window_size, tensorboard_log, monitor_wrapper, policy_kwargs,
            verbose, seed, device, _init_setup_model, supported_action_spaces
        )
        self.teachers = teachers

    def choose_teacher(self):
        """
        Returns a teacher to use for an episode.
        Currently  only implemented random selection.
        """
        return np.random.choice(self.teachers)

    def train(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Set teacher for this episode
        self.set_teacher(teacher_model=self.choose_teacher())

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            if hasattr(self, 'teacher_model') and self.teacher_model is not None:
                # Value-loss
                actions = rollout_data.actions

                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # Distillation-loss
                teacher_act_distribution = self.teacher_model.policy.get_distribution(rollout_data.observations)
                student_act_distribution = self.policy.get_distribution(rollout_data.observations)

                # Forward/reverse KL
                kl_divergence = gym.distributions.kl_divergence(teacher_act_distribution, student_act_distribution)

                if isinstance(teacher_act_distribution,
                              (distributions.DiagGaussianDistribution,
                               distributions.StateDependentNoiseDistribution)):
                    kl_divergence = distributions.sum_independent_dims(kl_divergence)
                kl_divergence = th.mean(kl_divergence)

                # distill both policy and value function from the teacher
                distillation_loss = th.mean(kl_divergence)
                teacher_values = self.teacher_model.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(th.squeeze(teacher_values), values)

                loss = distillation_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            else:
                print("ERROR: must first call model.set_teacher(teacher_model)")
                return

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/distillation_loss", distillation_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "multiteacher-student-distill",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )