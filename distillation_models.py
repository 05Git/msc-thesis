import gymnasium as gym
import torch as th
import numpy as np

from torch.nn import functional as F
from sb3_distill.core import PolicyDistillationAlgorithm

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import distributions
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor


class StudentDistilSolver(PolicyDistillationAlgorithm, OnPolicyAlgorithm):
    """
    Adjusted version of sb3_distill's StudentDistill policy.
    Original code available at: https://github.com/spiglerg/sb3_distill/blob/main/sb3_distill/student_distill.py
    """
    def __init__(
        self,
        student: OnPolicyAlgorithm,
        teachers: dict[str: OnPolicyAlgorithm],
        probabilities: list[float] = None
    ):
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

class MultiTeacherSolver(OnPolicyAlgorithm):
    def __init__(
        self,
        student: OnPolicyAlgorithm,
        teachers: dict[str: OnPolicyAlgorithm]
    ):
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
        self.teacher_logits = {idx: [] for idx in range(self.student.env.action_space.shape[0])}
        self.teacher_values = []

    def save(self, path, exclude = None, include = None):
        self.policy = self.student.policy
        return super().save(path, exclude, include)

    def _excluded_save_params(self):
        excluded = super()._excluded_save_params()
        return excluded + ["teachers"] + ["student"]
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                #########
                teacher_values = []
                teacher_ids = []
                teacher_logits = []
                for t_id, teacher in self.teachers.items():
                    t_act, t_val, t_log = teacher.policy(obs_tensor)
                    teacher_values.append(t_val.squeeze())
                    teacher_ids.append(t_id)
                    dist = teacher.policy.get_distribution(obs_tensor)
                    logits = []
                    for category in dist.distribution:
                        logits.append(category.logits)
                    teacher_logits.append(logits)
                teacher_values = th.stack(teacher_values)
                max_values = th.max(teacher_values, dim=0).values
                self.teacher_values.append(max_values)
                best_teacher_indices = th.argmax(teacher_values, dim=0)
                for action_category in range(self.student.env.action_space.shape[0]):
                    for idx, teacher_idx in enumerate(best_teacher_indices):
                        self.teacher_logits[action_category].append(teacher_logits[teacher_idx][action_category][idx])
                ##########

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, gym.spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def train(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.student.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.student.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
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
            teacher_act_logits = [th.stack(self.teacher_logits[i]) for i in self.teacher_logits.keys()]
            teacher_act_logits = th.cat(teacher_act_logits, dim=1)
            teacher_act_distribution = distributions.MultiCategoricalDistribution(action_dims=self.student.env.action_space.nvec)
            teacher_act_distribution.proba_distribution(action_logits=teacher_act_logits)
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
            value_loss = F.mse_loss(th.squeeze(th.cat(self.teacher_values)), values)

            loss = distillation_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            # Optimization step
            self.student.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.student.policy.parameters(), self.max_grad_norm)
            self.student.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/distillation_loss", distillation_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/loss", loss.item())
        if hasattr(self.student.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.student.policy.log_std).mean().item())
        
        self.teacher_values.clear()
        for t_id in self.teacher_logits.keys():
            self.teacher_logits[t_id].clear()

    def learn(
        self,
        total_timesteps,
        callback = None,
        log_interval = 1,
        tb_log_name = "MultiTeacherSolver",
        reset_num_timesteps = True,
        progress_bar = False
    ):
        return super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            progress_bar
        )