import gymnasium as gym
import torch as th
import torch.nn as nn

from typing import Optional, Tuple, Callable, Any
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
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
        fixed_weights: Optional[list[float]] = None,
        adaptive_weights_dist_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Set the options for how to use expert policies.
        """
        self.use_expert_extractors = use_expert_extractors
        self.predict_expert_values = predict_expert_values
        
        assert log_selection_rate > 0
        self.log_selection_rate = log_selection_rate

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

        if self.expert_selection_method == "fixed_weights":
            assert sum(fixed_weights) == 1 and len(fixed_weights) == len(self.experts)
            self.fixed_weights = fixed_weights
        elif self.expert_selection_method == "adaptive_weights":
            # Use size of extracted features dim as input size (+ additional space if using extra info)
            # NOTE: Could add a build function for modularizing + experimenting with architecture
            # TODO: Add modular sizes for info like teacher-student gaps
            latent_dim_weights = self.features_dim
            self.weights_net = nn.Sequential(
                nn.Linear(in_features=latent_dim_weights, out_features=len(self.experts), device=self.device),
                nn.Softmax() # Softmax weights to be between [0,1]
            )

    def forward(self, obs, deterministic = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
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
            # Pass model through training without learning adaptive weights (loss.backward() won't work since expert params are frozen)
            expert_mean_actions_tensor = self.action_net(latent_pi)
            action_weights = th.zeros((obs.shape[0], len(self.experts)), device=self.device)
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

            if self.expert_selection_method == "value":
                # Select actions for each observation according to highest expert values
                chosen_indices = th.stack(expert_values).argmax(dim=0)
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], len(self.experts)), device=self.device)
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1

            elif self.expert_selection_method == "entropy":
                # Select actions for each observation according to lowest expert entropies
                chosen_indices = th.stack(expert_entropies).argmin(dim=0)
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], len(self.experts)), device=self.device)
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1

            elif self.expert_selection_method == "random":
                # Select actions for each observation randomly
                chosen_indices = th.randint(low=0, high=len(self.experts), size=(obs.shape[0],)) # obs.shape[0] tells us the batch size
                # Weight actions from chosen experts by 1, others by 0
                action_weights = th.zeros((obs.shape[0], len(self.experts)), device=self.device)
                for action_idx, chosen_idx in enumerate(chosen_indices):
                    action_weights[action_idx][chosen_idx] = 1

            elif self.expert_selection_method == "fixed_weights":
                # Weight expert actions by pre-chosen values
                action_weights = th.tensor(self.fixed_weights, device=self.device).repeat(obs.shape[0], 1)

            elif self.expert_selection_method == "adaptive_weights":
                # Weight expert actions by learnt weights
                # TODO: Check shape of returned tensor (should just work, right?)
                action_weights = self.weights_net(latent_pi)
                
            else:
                raise ValueError(f"""Invalid value ({self.expert_selection_method}) for 'expert_selection_method'.
                                \nValid options: {[self.valid_selection_options]}.""")

            # Mask expert mean actions and sum along actions dimension (dim 1)
            expert_mean_actions_tensor *= action_weights.unsqueeze(-1)
            expert_mean_actions_tensor  = th.sum(expert_mean_actions_tensor, dim=1)

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
        for expert_id in self.expert_selection_rate.keys():
            self.expert_selection_rate[expert_id] = 0
        return selection_rates

    def evaluate_actions(self, obs, actions) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
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
            "expert_selection_rate",
            "use_expert_extractors",
            "predict_expert_values",
            "expert_selection_method",
            "log_selection_rate",
            "fixed_weights",
        ]
