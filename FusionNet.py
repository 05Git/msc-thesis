import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np

from typing import Optional, Tuple, Callable, Union, TypeVar
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import PyTorchObs, GymEnv, MaybeCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    kl_divergence,
    sum_independent_dims,
)


class MultiExpertFusionPolicy(ActorCriticPolicy):
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
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
    #######################################################
    
    # MODIFIED: Set the list of experts for the fusion model.
    def set_experts(self, experts: dict[str: OnPolicyAlgorithm]) -> None:
        """
        Set dictionary of experts.
        If using hard switches, then the number and ordering of experts is arbitrary.
        However, if using learnt adaptive weights, you must preserve the number and
        ordering when loading a fusion policy after training.
        """
        self.experts = experts
        self.expert_selection_rate = {id: 0 for id in experts.keys()}
        self.n_experts = len(experts)
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
        expert_selection_method: Optional[str] = "dummy",
        fixed_weights: Optional[list[float]] = None,
        adaptive_weights_kwargs: Optional[list[str]] = None, # expert-student gaps, values, entropies
        adaptive_weights_optimizer_kwargs: Optional[dict] = {},
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
            "hard_weights",
            "soft_weights",
        ]
        assert expert_selection_method in self.valid_selection_options, \
            f"""Invalid input for 'expert_selection_method': ({expert_selection_method}).
            \nValid options: {self.valid_selection_options}."""
        self.expert_selection_method = expert_selection_method

        if "weights" in self.expert_selection_method:
            if "fixed" in self.expert_selection_method:
                assert sum(fixed_weights) == 1 and len(fixed_weights) == self.n_experts
                self.fixed_weights = fixed_weights
            else:
                # Use size of extracted features dim as input size (+ additional space if using extra info)
                # NOTE: Could add a build function for modularizing + experimenting with architecture
                latent_dim_weights = self.features_dim
                if adaptive_weights_kwargs is not None:
                    valid_kwargs = [
                        "expert_value",
                        "expert_entropy",
                    ]
                    for kwarg in adaptive_weights_kwargs:
                        assert kwarg in valid_kwargs, f"Invalid kwarg, must be in: {valid_kwargs}"
                        latent_dim_weights += self.n_experts
                self.adaptive_weights_kwargs = adaptive_weights_kwargs
                # TODO: Investigate 2-3 additional layers
                self.weights_net = nn.Sequential(
                    nn.Linear(in_features=latent_dim_weights, out_features=self.n_experts, device=self.device),
                    nn.Softmax(dim=-1) # Softmax weights to be between [0,1]
                )
                # weights_net specific optim used to calculate auxiliary loss
                self.weights_net_optimizer = th.optim.Adam(
                    params=self.weights_net.parameters(),
                    **adaptive_weights_optimizer_kwargs
                )

    def forward(self, obs, deterministic = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Returns actions and log probabilities calculated from fusion of expert distributions.
        Also returns highest expert values or new learnt values, depending on option selected at initialisation.
        """
        ############################# MODIFIED #############################
        values = self.predict_values(obs)
        latent_pi = self.extract_latent_features(obs, network="pi")
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
        # NOTE: Haven't tested with dict obs yet, but this probably breaks it.
        # Will need to readjust to work with dict obs, if the time comes for it.
        if obs.ndim < 4:
            obs = obs.unsqueeze(dim=0)

        if self.expert_selection_method == "dummy":
            # Pass model through training without learning adaptive weights (loss.backward() won't work since expert params are frozen)
            expert_mean_actions_tensor = self.action_net(latent_pi)
            action_weights = th.zeros((obs.shape[0], self.n_experts), device=self.device)
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

            # Reshape mean actions to [n_envs, n_experts, ...]
            expert_mean_actions_tensor = th.stack(expert_mean_actions)
            expert_mean_actions_tensor = expert_mean_actions_tensor.permute(1, 0, *range(2, expert_mean_actions_tensor.ndim))

            if self.expert_selection_method == "value":
                # Select actions for each observation according to highest expert values
                chosen_indices = th.stack(expert_values).argmax(dim=0)
                if chosen_indices.ndim == 0:
                    chosen_indices = th.unsqueeze(chosen_indices, dim=-1)
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], self.n_experts), device=self.device)
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1

            elif self.expert_selection_method == "entropy":
                # Select actions for each observation according to lowest expert entropies
                chosen_indices = th.stack(expert_entropies).argmin(dim=0)
                if chosen_indices.ndim == 0:
                    chosen_indices = th.unsqueeze(chosen_indices, dim=-1)
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], self.n_experts), device=self.device)
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1

            elif self.expert_selection_method == "random":
                # Select actions for each observation randomly
                chosen_indices = th.randint(low=0, high=self.n_experts, size=(obs.shape[0],)) # obs.shape[0] tells us the batch size
                if chosen_indices.ndim == 0:
                    chosen_indices = th.unsqueeze(chosen_indices, dim=-1)
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], self.n_experts), device=self.device)
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1
            
            elif "weights" in self.expert_selection_method:
                if "fixed" in self.expert_selection_method:
                    # Weight expert actions by pre-chosen values
                    action_weights = th.tensor(self.fixed_weights, device=self.device).repeat(obs.shape[0], 1)
                else:
                    # Weight expert actions by learnt weights
                    latent_weights = latent_pi
                    if self.adaptive_weights_kwargs is not None:
                        # Add extra info to the latent space if specified by kwargs
                        for kwarg in self.adaptive_weights_kwargs:
                            if kwarg == "expert_value":
                                reshaped_values = th.stack(expert_values, dim=1) # Stack along n_envs dimension (dim 1) to match latent_weights
                                latent_weights = th.cat((latent_weights, reshaped_values), dim=1)
                            elif kwarg == "expert_entropy":
                                reshaped_entropies = th.stack(expert_entropies, dim=1)
                                latent_weights = th.cat((latent_weights, reshaped_entropies), dim=1)
                    action_weights = self.weights_net(latent_weights)
                    if "hard" in self.expert_selection_method:
                        action_weights_indices = th.argmax(action_weights, dim=1)
                        for action_idx, chosen_idx in enumerate(action_weights_indices):
                            action_weights[action_idx][:] = 0
                            action_weights[action_idx][chosen_idx] = 1
                
            else:
                raise ValueError(f"""Invalid value ({self.expert_selection_method}) for 'expert_selection_method'.
                                \nValid options: {[self.valid_selection_options]}.""")

            # Mask expert mean actions and sum along actions dimension (dim 1)
            expert_mean_actions_tensor = expert_mean_actions_tensor * action_weights.unsqueeze(-1)
            expert_mean_actions_tensor = th.sum(expert_mean_actions_tensor, dim=1)

        # Log which experts were selected and by how much (1 if hard switch, some float in [0,1] if weighted)
        action_weights = th.sum(action_weights, dim=0)
        for idx, expert_id in enumerate(self.experts.keys()):
            self.expert_selection_rate[expert_id] += action_weights[idx].item()
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
        
    # MODIFIED: Get the expert selection rates' current values, then reset them. Used for logging metrics.
    def get_expert_selection_rates(self) -> th.Tensor:
        """
        Return the selection rate of each expert for metric logging.
        """
        selection_rates = self.expert_selection_rate.copy()
        total_weight = sum(self.expert_selection_rate.values())
        if total_weight == 0:
            # In case this gets called by a dummy policy
            total_weight += 1e-8
        for expert_id in self.expert_selection_rate.keys():
            selection_rates[expert_id] /= total_weight
            self.expert_selection_rate[expert_id] = 0
        return selection_rates

    def evaluate_actions(self, obs, actions) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Returns log probabilities and entropy calculated from fusion of expert distributions.
        Also returns highest expert values or new learnt values, depending on option selected at initialisation.
        """
        ############################# MODIFIED #############################
        values = self.predict_values(obs)
        latent_pi = self.extract_latent_features(obs, network="pi")
        distribution = self._get_action_dist_from_latent(latent_pi, obs=obs)
        ####################################################################
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Returns a distribution built from a combination of expert policies.
        """
        latent_pi = self.extract_latent_features(obs, network="pi")
        return self._get_action_dist_from_latent(latent_pi, obs=obs) # MODIFIED: Includes observation data
    
    def extract_latent_features(self, obs: PyTorchObs, network: str = "pi") -> th.Tensor:
        """
        Extract latent features vector. Primarily used as helper function during auxiliary loss calculations.
        """
        assert network in ["pi", "vf"]
        if network == "pi":
            features = super().extract_features(obs, self.pi_features_extractor)
            return self.mlp_extractor.forward_actor(features)
        else:
            features = super().extract_features(obs, self.vf_features_extractor)
            return self.mlp_extractor.forward_critic(features)        
    
    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Either returns max values calculated by experts, or learnt by the fusion policy.
        """
        ############################################## MODIFIED ###################################################
        if not self.predict_expert_values:
            latent_vf = self.extract_latent_features(obs, network="vf")
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
            "expert_selection_rate",
            "use_expert_extractors",
            "predict_expert_values",
            "expert_selection_method",
            "fixed_weights",
        ]


SelfFusionNet = TypeVar("SelfFusionNet", bound="MultiExpertFusionNet")

class MultiExpertFusionNet(PPO):
    """
    An on-policy algorithm sub-classed from PPO.
    Extends the train() function to train a MultiExpertFusionPolicy's weights_net using auxiliary loss.
    """
    def __init__(
        self,
        policy: MultiExpertFusionPolicy,
        env: Union[GymEnv, str],
        auxiliary_loss_coef: float = 1.,
        div_loss_coef_init: float = 1.,
        div_loss_coef_end: float = 0.,
        div_loss_coef_fraction: float = 0.9,
        weights_net_n_epochs: int = None,
        max_weights_net_grad_norm: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            policy,
            env,
            *args,
            **kwargs
        )
        assert isinstance(self.policy, MultiExpertFusionPolicy)
        assert auxiliary_loss_coef >= 0 and auxiliary_loss_coef <= 1
        self.auxiliary_loss_coef = auxiliary_loss_coef
        
        assert div_loss_coef_init <= 1
        assert div_loss_coef_init >= div_loss_coef_end
        assert div_loss_coef_end >= 0
        assert div_loss_coef_fraction > 0 and div_loss_coef_fraction <= 1
        self.div_loss_coef_init = div_loss_coef_init
        self.div_loss_coef_end = div_loss_coef_end
        self.div_loss_coef_fraction = div_loss_coef_fraction
        self.div_loss_schedule = get_linear_fn(
            self.div_loss_coef_init,
            self.div_loss_coef_end,
            self.div_loss_coef_fraction
        )

        if weights_net_n_epochs is None:
            self.weights_net_n_epochs = self.n_epochs
        else:
            assert weights_net_n_epochs > 0
            self.weights_net_n_epochs = weights_net_n_epochs

        if max_weights_net_grad_norm is None:
            self.max_weights_net_grad_norm = self.max_grad_norm
        else:
            assert max_weights_net_grad_norm > 0
            self.max_weights_net_grad_norm = max_weights_net_grad_norm

    def train(self) -> None:
        """
        After calling the base PPO train loop, calculate weights_net auxiliary loss.
        """
        super().train()
        if hasattr(self.policy, "weights_net"):
            assert self.policy.experts is not None, "Must set experts before training weights_net"
            self.policy.weights_net.train()
            divergence_losses = []
            auxiliary_losses = []
            div_loss_coef = self.div_loss_schedule(self._current_progress_remaining)
            self.logger.record("weights_net/div_loss_coef", div_loss_coef)
            for _ in range(self.weights_net_n_epochs):
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    observations = rollout_data.observations

                    # Get distributions from each expert and the fusion policy
                    expert_dists = [
                        expert_net.policy.get_distribution(observations)
                        for expert_net in self.policy.experts.values()
                    ]
                    fusion_dist = self.policy.get_distribution(observations)

                    # Calculate divergence between each expert's distribution and the fused distribution
                    divergence_weight = 1 / self.policy.n_experts # Weight each expert's divergence uniformally
                    divergence_loss = th.stack([
                        kl_divergence(expert_dist, fusion_dist) * divergence_weight
                        for expert_dist in expert_dists
                    ])
                    
                    if isinstance(self.policy.action_dist, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
                        divergence_loss = th.stack([
                            sum_independent_dims(expert_div_loss)
                            for expert_div_loss in divergence_loss
                        ])
                    divergence_loss = th.mean(divergence_loss)
                    divergence_losses.append(divergence_loss.item())

                    weights_net_auxiliary_loss = (divergence_loss * div_loss_coef) * self.auxiliary_loss_coef
                    auxiliary_losses.append(weights_net_auxiliary_loss.item())

                    self.policy.weights_net_optimizer.zero_grad()
                    weights_net_auxiliary_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.weights_net.parameters(),
                        self.max_weights_net_grad_norm
                    )
                    self.policy.weights_net_optimizer.step()
            
            self.logger.record("weights_net/divergence_loss", np.mean(divergence_losses))
            self.logger.record("weights_net/auxiliary_loss", np.mean(auxiliary_losses))

    def learn(
        self: SelfFusionNet,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MultiExpertFusionNet",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFusionNet:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
