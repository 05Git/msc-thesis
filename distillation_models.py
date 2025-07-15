import gymnasium as gym
import torch as th
import numpy as np

from sb3_distill.core import PolicyDistillationAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common import distributions
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F
from typing import Dict, Type

class StudentDistilSolver(PolicyDistillationAlgorithm, OnPolicyAlgorithm):
    """
    Adjusted version of sb3_distill's StudentDistill policy.
    Original code available at: https://github.com/spiglerg/sb3_distill/blob/main/sb3_distill/student_distill.py
    """
    def __init__(self, student: OnPolicyAlgorithm, teachers: dict[str: OnPolicyAlgorithm], probabilities: list[float] = None):
        super().__init__(
            policy=type(student.policy), env=student.env, learning_rate=student.learning_rate, n_steps=student.n_steps,
            gamma=student.gamma, gae_lambda=student.gae_lambda, ent_coef=student.ent_coef, 
            vf_coef=student.vf_coef, max_grad_norm=student.max_grad_norm, use_sde=student.use_sde,
            sde_sample_freq=student.sde_sample_freq, rollout_buffer_class=student.rollout_buffer_class, 
            rollout_buffer_kwargs=student.rollout_buffer_kwargs, stats_window_size=student._stats_window_size,
            tensorboard_log=student.tensorboard_log, policy_kwargs=student.policy_kwargs,
            verbose=student.verbose, seed=student.seed, device=student.device,
        )
        self.student = student      # Setting a separate student model allows non-distillation models to be finetuned via distillation
        self.teachers = teachers    # List of teacher policies
        self.policy = student.policy
        self.probabilities = probabilities

    def choose_teacher(self) -> OnPolicyAlgorithm:
        """
        Returns a teacher to use for an episode.
        Selection probability can be specified at init to favor certain teachers over others.
        """
        teacher_id = np.random.choice(list(self.teachers.keys()), p=self.probabilities)
        return self.teachers[teacher_id]

    def train(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.student.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.student.policy.optimizer)

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

                values, log_prob, entropy = self.student.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # Distillation-loss
                teacher_act_distribution = self.teacher_model.policy.get_distribution(rollout_data.observations)
                student_act_distribution = self.student.policy.get_distribution(rollout_data.observations)

                # Forward/reverse KL
                kl_divergence = distributions.kl_divergence(teacher_act_distribution, student_act_distribution)

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
                self.student.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.student.policy.parameters(), self.max_grad_norm)
                self.student.policy.optimizer.step()
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
        if hasattr(self.student.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.student.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
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

    def save(self, path, exclude = None, include = None):
        self.policy = self.student.policy
        return super().save(path, exclude, include)

    def _excluded_save_params(self):
        excluded = super()._excluded_save_params()
        return excluded + ["teachers"] + ["student"]
    