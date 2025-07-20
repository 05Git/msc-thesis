import gymnasium as gym
import torch as th
import numpy as np
import torch.nn as nn

from torch.nn import functional as F
from sb3_distill.core import PolicyDistillationAlgorithm
from sb3_distill import StudentDistill
from typing import Optional, NamedTuple, Tuple, Callable

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import distributions
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutBufferSamples, PyTorchObs
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)


class MultiExpertFusionPolicy(ActorCriticCnnPolicy):
    """
    A policy which fuses the knowledge of multiple experts together, for calculating variables such as actions and state values.
    The fusion policy can either use hard switches to pick specific experts for each observation, or learn adaptive weights to
    adjust the effect of each expert's knowledge on the final distribution.
    The fusion policy can also rely solely on the experts' knowledge of state values, or learn a new critic during training.

    Contains modified functions from stable_baselines3's ActorCriticPolicy. All lines of code which have been added or changed
    are highlighted with ## MODIFIED ##.
    Original code available at: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L416
    """
    ###################### MODIFIED #######################
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.distribution_calls = 0
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
    #######################################################
    
    # MODIFIED: Set the list of experts for the fusion model.
    # NOTE: This isn't going to work with adaptive weighting trained on specific expert policies, consider reworking.
    def set_experts(self, experts: dict[str: OnPolicyAlgorithm]) -> None:
        """
        Set dictionary of experts.
        """
        self.experts = experts
        self.expert_selection_rate = {id: 0 for id in experts.keys()}
        # Freeze expert parameters
        # Training experts alongside a student currently falls outside the scope of this project
        for expert_net in self.experts.values():
            for param in expert_net.policy.parameters():
                param.requires_grad = False
    
    # MODIFIED: Set the list of options regarding how to use experts
    def set_expert_params(
        self,
        use_expert_extractors: Optional[bool] = False,
        predict_expert_values: Optional[bool] = False,
        expert_selection_method: Optional[str] = "value",
        log_selection_rate: Optional[int] = 1024,
        rand_select_probabilities: Optional[list[float]] = None,
        fixed_weights: Optional[list[float]] = None,
    ) -> None:
        """
        Set the options for how to use expert policies.
        """
        self.use_expert_extractors = use_expert_extractors
        self.predict_expert_values = predict_expert_values

        self.valid_selection_options = [
            "dummy",
            "value",
            "entropy",
            "random",
            "fixed_weights",
            "adaptive_weights",
        ]
        assert expert_selection_method in self.valid_selection_options, \
            f"""Invalid input for 'expert_selection_method': ({expert_selection_method}).
            \nValid options: {self.valid_selection_options}."""
        self.expert_selection_method = expert_selection_method
        
        assert log_selection_rate > 0
        self.log_selection_rate = log_selection_rate

        if rand_select_probabilities is not None:
            assert sum(rand_select_probabilities) == 1 \
                and len(rand_select_probabilities) == len(self.experts)
        self.rand_select_probabilities = rand_select_probabilities

        if fixed_weights is not None:
            assert sum(fixed_weights) == 1 and len(fixed_weights) == len(self.experts)
        self.fixed_weights = fixed_weights

    def forward(self, obs, deterministic = False):
        """
        Returns actions and log probabilities calculated from fusion of expert distributions.
        Also returns highest expert values or new learnt values, depending on option selected at initialisation.
        """
        ############################# MODIFIED #############################
        values = self.predict_values(obs)
        pi_features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs=obs)
        ####################################################################
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs: PyTorchObs) -> Distribution:
        """
        Pass a set of latent features to the action distributions of each expert, then extract their logits.
        Fuse each expert's logits together using learnt or fixed weights, or use a hard switch like
        max value or min entropy to choose a specific distribution.

        Could extract features in a couple ways: Use the features learnt by the experts, or learn a new
        feature extractor. Will try to investigate both.
        """
        ################################################## MODIFIED ######################################################
        # Add extra dimension at dim 0 if obs is unbatched
        if obs.ndim < 4:
            obs = obs.unsqueeze(dim=0)

        if self.expert_selection_method == "dummy":
            expert_mean_actions_tensor = self.action_net(latent_pi)
            action_weights = th.zeros((obs.shape[0], len(self.experts)))
        else:
            assert self.experts is not None, "Must set expert policies before predicting actions."

            expert_mean_actions = [] # Values of each expert's action distribution (logits, mean values, etc)
            expert_entropies = []
            expert_values = []

            for expert_net in self.experts.values():
                # Choose whether to use features extracted by each indiviual expert, or by the fusion network
                if self.use_expert_extractors:
                    features = expert_net.policy.extract_features(obs, expert_net.policy.pi_features_extractor)
                    expert_latent_pi = expert_net.policy.mlp_extractor.forward_actor(features)
                else:
                    expert_latent_pi = latent_pi
                # Calculate and collect each expert's predicted actions (logits), entropies and values
                # TODO: Check if there's drift between action_net(expert_latent_pi) and distribution.entropy() (they might not perfectly match up)
                expert_mean_actions.append(expert_net.policy.action_net(expert_latent_pi))
                expert_entropies.append(expert_net.policy.get_distribution(obs).entropy().squeeze(-1))
                expert_values.append(expert_net.policy.predict_values(obs).squeeze(-1))

            # Reshape mean actions to [batch_size, num_experts, ...]
            expert_mean_actions_tensor = th.stack(expert_mean_actions)
            expert_mean_actions_tensor = expert_mean_actions_tensor.permute(1, 0, *range(2, expert_mean_actions_tensor.ndim))

            # TODO: Add adaptive weighting
            # if self.expert_selection_method == "dummy":
                
            if self.expert_selection_method == "value":
                # Select actions for each observation according to highest expert values
                chosen_indices = th.stack(expert_values).argmax(dim=0)
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], len(self.experts)))
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1
            elif self.expert_selection_method == "entropy":
                # Select actions for each observation according to lowest expert entropies
                chosen_indices = th.stack(expert_entropies).argmin(dim=0)
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], len(self.experts)))
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1
            elif self.expert_selection_method == "random":
                # Select actions for each observation randomly
                chosen_indices = th.randint(low=0, high=len(self.experts), size=(obs.shape[0],)) # obs.shape[0] tells us the batch size
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], len(self.experts)))
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1
            elif self.expert_selection_method == "fixed_weights":
                # Weight expert actions by pre-chosen values
                action_weights = th.tensor(self.fixed_weights).repeat(obs.shape[0], 1)
            elif self.expert_selection_method == "adaptive_weights":
                raise NotImplementedError("Sorry, coming soon.")
            else:
                raise ValueError(f"""Invalid value ({self.expert_selection_method}) for 'expert_selection_method'.
                                \nValid options: {[self.valid_selection_options]}.""")

            # Mask expert mean actions and sum along actions dimension (dim 1)
            expert_mean_actions_tensor *= action_weights.unsqueeze(-1)
            expert_mean_actions_tensor  = th.sum(expert_mean_actions_tensor, dim=1)

        # Log which experts were selected and by how much (1 if hard switch, some float in [0,1] if weighted)
        action_weights = th.sum(action_weights, dim=0)
        for idx, expert_id in enumerate(self.experts.keys()):
            self.expert_selection_rate[expert_id] += action_weights[idx]
        self.distribution_calls += obs.shape[0]
        if self.distribution_calls % self.log_selection_rate == 0:
            # Log selection rates (normalized between [0,1]) to logger
            if hasattr(self, "logger") and self.logger is not None:
                for expert_id in self.experts.keys():
                    self.expert_selection_rate[expert_id] /= self.log_selection_rate
                    self.logger.record(f"experts/{expert_id}_selection_rate", self.expert_selection_rate[expert_id])
            # Print selection rates if policy is verbose
            if hasattr(self, "verbose") and self.verbose > 0:
                print(f"Expert selection rates: {self.expert_selection_rate}")
            # Reset selection rates and distribution calls
            for expert_id in self.expert_selection_rate.keys():
                self.expert_selection_rate[expert_id] = 0
            self.distribution_calls = 0
        ##################################################################################################################

        # MODIFIED: Distributions built using expert mean actions
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(expert_mean_actions_tensor, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=expert_mean_actions_tensor)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=expert_mean_actions_tensor)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=expert_mean_actions_tensor)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(expert_mean_actions_tensor, self.log_std, latent_pi) # NOTE: Might want to investigate using experts' latent_pi if possible
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs, actions):
        """
        Returns log probabilities and entropy calculated from fusion of expert distributions.
        Also returns highest expert values or new learnt values, depending on option selected at initialisation.
        """
        ############################# MODIFIED #############################
        values = self.predict_values(obs)
        pi_features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs=obs)
        ####################################################################
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Returns a distribution built from a combination of expert policies.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, obs=obs) # MODIFIED: Includes observation data
    
    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Either returns max values calculated by experts, or learnt by the fusion policy.
        """
        ############################################## MODIFIED ###################################################
        if not self.predict_expert_values:
            features = super().extract_features(obs, self.vf_features_extractor)
            latent_vf = self.mlp_extractor.forward_critic(features)
            return self.value_net(latent_vf)
        
        assert self.experts is not None, "Must set experts before calling them to evaluate state values."
        # Collect the values from each expert for each observation, and return the highest
        expert_values = [expert_net.policy.predict_values(obs).squeeze(-1) for expert_net in self.experts.values()]
        return th.stack(expert_values).max(dim=0).values
        ###########################################################################################################
    
    # MODIFIED: Exclude experts during saving to avoid errors/bugs.
    def _excluded_save_params(self):
        """
        Exclude policy specific parameters.
        """
        excluded = super()._excluded_save_params()
        return excluded + [
            "experts",
            "distribution_calls",
            "use_expert_extractors",
            "predict_expert_values",
            "expert_selection_method",
            "log_selection_rate",
            "rand_select_probabilities",
            "fixed_weights",
        ]


# Won't need anything below here anymore
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

                expert_values, log_prob, entropy = self.student.policy.evaluate_actions(rollout_data.observations, actions)
                expert_values = expert_values.flatten()

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
                teacher_expert_values = self.teacher_model.policy.predict_expert_values(rollout_data.observations)
                value_loss = F.mse_loss(th.squeeze(teacher_expert_values), expert_values)

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

        explained_var = explained_variance(self.rollout_buffer.expert_values.flatten(), self.rollout_buffer.returns.flatten())

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


class Teacherexpert_LogitsRolloutBuffer(RolloutBuffer):

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    expert_values: np.ndarray
    teacher_expert_logits: np.ndarray
    teacher_expert_values: np.ndarray

    def __init__(
        self, buffer_size, observation_space, action_space,
        device = "auto", gae_lambda = 1, gamma = 0.99, n_envs = 1
    ):
        super().__init__(
            buffer_size, observation_space, action_space,
            device, gae_lambda, gamma, n_envs
        )
        self.teacher_expert_logits = np.zeros((self.buffer_size, self.n_envs, sum(self.action_space.nvec)), dtype=np.float32)
        self.teacher_expert_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    
    def reset(self) -> None:
        self.teacher_expert_logits = np.zeros((self.buffer_size, self.n_envs, sum(self.action_space.nvec)), dtype=np.float32)
        self.teacher_expert_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        return super().reset()
    
    def add(
        self, obs, action, reward, episode_start, value, log_prob,
        teacher_expert_logits, teacher_expert_values
    ):
        self.teacher_expert_logits[self.pos] = np.array(teacher_expert_logits)
        self.teacher_expert_values[self.pos] = np.array(teacher_expert_values)
        return super().add(obs, action, reward, episode_start, value, log_prob)
    
    def get(self, batch_size: Optional[int] = None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "expert_values",
                "log_probs",
                "advantages",
                "returns",
                "teacher_expert_logits",
                "teacher_expert_values"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.expert_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.teacher_expert_logits[batch_inds],
            self.teacher_expert_values[batch_inds].flatten()
        )
        return MultiTeacherRolloutSamples(*tuple(map(self.to_torch, data)))


class MultiTeacherRolloutSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_expert_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    teacher_expert_logits: th.Tensor
    teacher_expert_values: th.Tensor


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
            sde_sample_freq=student.sde_sample_freq, rollout_buffer_class=Teacherexpert_LogitsRolloutBuffer, 
            rollout_buffer_kwargs=student.rollout_buffer_kwargs, stats_window_size=student._stats_window_size,
            tensorboard_log=student.tensorboard_log, policy_kwargs=student.policy_kwargs,
            verbose=student.verbose, seed=student.seed, device=student.device,
        )
        self.student = student      # Setting a separate student model allows non-distillation models to be finetuned via distillation
        self.teachers = teachers    # List of teacher policies
        self.policy = student.policy

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

        ######
        teacher_selection_rates = np.zeros((len(self.teachers), 1))
        ######
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, expert_values, log_probs = self.policy(obs_tensor)
                ############################################## MODIFIED ################################################
                teacher_expert_values = []
                teacher_expert_entropies = []
                teacher_ids = []
                teacher_expert_logits = []
                for t_id, teacher in self.teachers.items():
                    t_act, t_val, t_log = teacher.policy(obs_tensor)
                    teacher_expert_values.append(t_val.squeeze())
                    teacher_ids.append(t_id)
                    dist: distributions.MultiCategoricalDistribution = teacher.policy.get_distribution(obs_tensor)
                    expert_logits = []
                    for category in dist.distribution:
                        expert_logits.append(category.expert_logits)
                    expert_logits = th.cat(expert_logits, dim=1)
                    teacher_expert_logits.append(expert_logits)
                    entropy = dist.entropy()
                    teacher_expert_entropies.append(entropy)
                teacher_expert_values = th.stack(teacher_expert_values)
                max_val_returns = th.max(teacher_expert_values, dim=0)
                teacher_expert_entropies = th.stack(teacher_expert_entropies)
                min_ent_returns = th.min(teacher_expert_entropies, dim=0)
                min_expert_entropies = min_ent_returns.expert_values
                max_teacher_expert_values = max_val_returns.expert_values
                max_teacher_expert_values = max_teacher_expert_values.cpu().numpy()
                best_teacher_indices = min_ent_returns.indices # Use max expert_values or min expert_entropies
                best_teacher_expert_logits = th.zeros((self.n_envs, sum(self.action_space.nvec)))
                for idx, t_idx in enumerate(best_teacher_indices):
                    best_teacher_expert_logits[idx] = teacher_expert_logits[t_idx][idx]
                for idx, _ in enumerate(self.teachers.keys()):
                    teacher_selection_rates[idx][0] += th.sum(th.where(best_teacher_indices == idx, 1, 0)).cpu().numpy()
                ########################################################################################################

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
                        terminal_value = self.policy.predict_expert_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                expert_values,
                log_probs,
                best_teacher_expert_logits,
                max_teacher_expert_values
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            expert_values = self.policy.predict_expert_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_expert_values=expert_values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        ####
        teacher_selection_rates /= n_rollout_steps * self.n_envs
        for idx, t_id in enumerate(self.teachers.keys()):
            self.logger.record(f"teachers/{t_id}_selection_rate", teacher_selection_rates[idx][0])
        ####

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

            expert_values, log_prob, entropy = self.student.policy.evaluate_actions(rollout_data.observations, actions)
            expert_values = expert_values.flatten()

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            # Distillation-loss
            teacher_act_distribution = distributions.MultiCategoricalDistribution(action_dims=self.env.action_space.nvec)
            teacher_act_distribution.proba_distribution(action_expert_logits=rollout_data.teacher_expert_logits)
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
            value_loss = F.mse_loss(th.squeeze(rollout_data.teacher_expert_values), expert_values)

            loss = distillation_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            # Optimization step
            self.student.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.student.policy.parameters(), self.max_grad_norm)
            self.student.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.expert_values.flatten(), self.rollout_buffer.returns.flatten())

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