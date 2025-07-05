"""
Code taken from https://github.com/jcwleo/random-network-distillation-pytorch
"""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor


class RNDPPO(PPO):
    def __init__(
        self,
        policy,
        env,
        int_beta: float = 5e-2,
        rnd_model_args = {            
            "image_shape": (4, 84, 84),
            "action_size": 2,
            "vec_fc_size": 128,
            "feature_size": 512,
            "rnd_type": "state",
            "optim_args": {
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.,
            }
        },
        learning_rate = 0.0003,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.2,
        clip_range_vf = None,
        normalize_advantage = True,
        ent_coef = 0,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        use_sde = False,
        sde_sample_freq = -1,
        rollout_buffer_class = None,
        rollout_buffer_kwargs = None,
        target_kl = None,
        stats_window_size = 100,
        tensorboard_log = None,
        policy_kwargs = None,
        verbose = 0,
        seed = None,
        device = "auto",
        _init_setup_model = True
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model
        )
        self.rnd_model = RNDModel(**rnd_model_args).to(device)
        self.int_beta = int_beta
        self.rnd_running_mean = 0
        self.rnd_running_std = 1
    
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
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
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            ########################################################################
            if self.rnd_model.rnd_type == "state-action":
                obs_tensor = {"image": obs_tensor, "actions": th.tensor(clipped_actions, device=self.device)}
            predicted_features, target_features = self.rnd_model.forward(obs_tensor)
            int_rewards = (target_features - predicted_features).pow(2).sum(1) / 2
            int_rewards = int_rewards.detach().cpu().numpy()
            self.rnd_running_mean = 0.99 * self.rnd_running_mean + 0.01 * np.mean(int_rewards)
            self.rnd_running_std = 0.99 * self.rnd_running_std + 0.01 * np.std(int_rewards)
            if self.rnd_running_std == 0:
                self.rnd_running_std += 1e-8
            int_rewards = (int_rewards - self.rnd_running_mean) / self.rnd_running_std
            int_rewards_clamped = np.clip(int_rewards, -5.0, 5.0)
            rewards += self.int_beta * int_rewards_clamped
            predictor_loss = F.mse_loss(predicted_features, target_features)
            self.rnd_model.optimizer.zero_grad()
            predictor_loss.backward()
            self.rnd_model.optimizer.step()
            self.logger.record("RND/intrinsic_reward", np.mean(int_rewards_clamped))
            self.logger.record("RND/int_reward_unclamped", np.mean(int_rewards))
            self.logger.record("RND/rnd_model_loss", np.mean(predictor_loss.detach().cpu().numpy()))
            ########################################################################

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
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


class RNDModel(nn.Module):
    def __init__(
        self,
        image_shape,
        action_size: int = 2,
        vec_fc_size: int = 128,
        feature_size: int = 512,
        rnd_type: str = "state",
        optim_args = {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.,
        }
    ):
        super(RNDModel, self).__init__()
        assert rnd_type in ["state", "state-action"]
        self.rnd_type = rnd_type

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=image_shape[0],
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        dummy_input = th.zeros(1, *image_shape)
        with th.no_grad():
            conv_out_size = self.conv_layer(dummy_input).shape[1]

        self.vector_layer = nn.Sequential(
            nn.Linear(action_size, vec_fc_size),
            nn.ReLU(),
            nn.Linear(vec_fc_size, vec_fc_size),
            nn.ReLU()
        )

        feature_output = conv_out_size + vec_fc_size if self.rnd_type == "state-action" else conv_out_size

        self.predictor = nn.Sequential(
            nn.Linear(feature_output, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size)
        )

        self.target = nn.Sequential(
            nn.Linear(feature_output, feature_size)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False
        
        self.optimizer = th.optim.Adam(self.predictor.parameters(), **optim_args)

    def forward(self, next_obs):
        if self.rnd_type == "state-action":
            image = next_obs["image"].float()
            actions = next_obs["actions"].float()
            conv_features = self.conv_layer(image)
            vec_features = self.vector_layer(actions)
            input_features = th.concat((conv_features, vec_features), dim=1)
        else:
            image = next_obs.float()
            input_features = self.conv_layer(image)
            
        target_feature = self.target(input_features)
        predict_feature = self.predictor(input_features)

        return predict_feature, target_feature
