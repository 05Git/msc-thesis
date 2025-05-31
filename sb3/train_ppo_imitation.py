import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList

from diambra.arena.utils.diambra_data_loader import DiambraDataLoader
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger

import custom_wrappers
import custom_callbacks
import utils

# diambra run -s 8 python sb3/imitation_ppo.py --policyCfg config_files/transfer-cfg-ppo.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --datasetPath _ --trainID _ --charTransfer _

class BCLossMonitor:
    def __init__(self, bc_trainer, patience=5, min_delta=1e-4):
        self.bc_trainer = bc_trainer
        self.patience = patience
        self.min_delta = min_delta

        self.losses = []
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self):
        current_loss = self.bc_trainer._last_batch_loss.item()
        self.losses.append(current_loss)

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping: loss hasn't improved in {self.patience} batches.")
            self.should_stop = True

def get_transitions(data_loader: DiambraDataLoader):
    n_episodes = len(data_loader.episode_files) # Number of datasets
    trajectories = []
    for i in range(n_episodes):
        _ = data_loader.reset() # Load next episode data
        obs = np.array([
            cv2.imdecode( # Decode frames to correct shape
                np.frombuffer(
                    data["obs"]["frame"], dtype=np.uint8 # Read frame data for each step in the episode
                ),
                cv2.IMREAD_UNCHANGED,
            )
            for data in data_loader.episode_data
        ])
        obs = np.stack([obs[i:i+4] for i in range(len(obs) - 3)], axis=0) # Stack frames together, this is what the policy expects as input
        last_frame_stack = np.stack([obs[-1]] * 4)
        obs = np.concatenate([obs, last_frame_stack], axis=0) # Concat 4 extra frame stacks to match trajectory's required shape
        acts = np.array([data["action"] for data in data_loader.episode_data], dtype=np.uint8)
        rews = np.array([data["reward"] for data in data_loader.episode_data], dtype=np.float16)
        dones = np.array([data["terminated"] for data in data_loader.episode_data])
        terminal = np.any([done == True for done in dones]) # Does this episode end in a terminal state?
        infos = np.array([data["info"] for data in data_loader.episode_data])

        trajectories.append(TrajectoryWithRew(
            obs=obs,
            acts=acts,
            rews=rews,
            terminal=terminal,
            infos=infos
        ))
        print(f"Trajectory {i + 1} loaded")
    
    return rollout.flatten_trajectories_with_rew(trajectories)

def main(policy_cfg: str, settings_cfg: str, dataset_path_input: str, train_id: str | None, char_transfer: bool):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    
    if train_id not in game_ids:
        train_id = None

    if dataset_path_input is not None:
        dataset_path = dataset_path_input
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, "dataset")

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read the cfg files
    policy_file = open(policy_cfg)
    policy_params = yaml.load(policy_file, Loader=yaml.FullLoader)
    policy_file.close()

    settings_file = open(settings_cfg)
    settings_params = yaml.load(settings_file, Loader=yaml.FullLoader)
    settings_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "model"
    )
    tensor_board_folder = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "tb"
    )
    os.makedirs(model_folder, exist_ok=True)


    # Set up imitation transitions
    imitation_data_loader = DiambraDataLoader(dataset_path)
    transitions = get_transitions(imitation_data_loader)


    # PPO settings
    ppo_settings = policy_params["ppo_settings"]
    policy = ppo_settings["policy_type"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]
    n_eval_episodes = ppo_settings["n_eval_episodes"]
    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]
    gae_lambda = ppo_settings["gae_lambda"]
    ent_coef = ppo_settings["ent_coef"]
    vf_coef = ppo_settings["vf_coef"]
    max_grad_norm = ppo_settings["max_grad_norm"]
    use_sde = ppo_settings["use_sde"]
    sde_sample_freq = ppo_settings["sde_sample_freq"]
    normalize_advantage = ppo_settings["normalize_advantage"]
    stats_window_size = ppo_settings["stats_window_size"]
    target_kl = ppo_settings["target_kl"]
    time_steps = ppo_settings["time_steps"]
    autosave_freq = ppo_settings["autosave_freq"]
    seeds = ppo_settings["seeds"]

    # Policy kwargs
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}

    # Load wrappers settings as dictionary
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    # Load shared settings
    settings = settings_params["settings"]["shared"]
    # Set action space type
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE

    envs_settings = []
    # Load game specific settings
    if train_id:
        game_settings = settings_params["settings"][train_id]
        if char_transfer:
            for character in game_settings["characters"]:
                game_settings["characters"] = character
                env_settings = settings.copy()
                env_settings.update(game_settings)
                env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings)
                envs_settings.append(env_settings)
        else:
            game_settings["characters"] = game_settings["characters"][0]
            env_settings = settings.copy()
            env_settings.update(game_settings)
            env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings)
            envs_settings.append(env_settings)
    else:
        for game_id in game_ids:
            game_settings = settings_params["settings"][game_id]
            game_settings["characters"] = game_settings["characters"][0]
            env_settings = settings.copy()
            env_settings.update(game_settings)
            env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings)
            envs_settings.append(env_settings)

    eval_results = {}
    for seed in seeds:
        eval_results.update({seed: {}})
        utils.set_global_seed(seed)
        for epoch in range(len(envs_settings)):
            epoch_settings = envs_settings[epoch]
            env, num_envs = make_sb3_env(epoch_settings.game_id, epoch_settings, wrappers_settings, seed=seed)
            if epoch_settings.action_space == SpaceTypes.DISCRETE:
                env = custom_wrappers.VecEnvDiscreteTransferWrapper(env)
            else:
                env = custom_wrappers.VecEnvMDTransferWrapper(env)
            env = VecTransposeImage(env)
            
            print(f"\nOriginal action space: {env.unwrapped.action_space}")
            print(f"Wrapped action space: {env.action_space}")
            print("\nActivated {} environment(s)".format(num_envs))

            # Load policy params if checkpoint exists, else make a new agent
            checkpoint_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            if int(model_checkpoint) > 0 and os.path.exists(checkpoint_path):
                print("\n Checkpoint found, loading model.")
                agent = PPO.load(
                    checkpoint_path,
                    env=env,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensor_board_folder,
                    device=device,
                    custom_objects={
                        "action_space" : env.action_space,
                        "observation_space" : env.observation_space,
                    }
                )
            else:
                print("\nNo or invalid checkpoint given, creating new model")
                agent = PPO(
                    policy,
                    env,
                    verbose=1,
                    gamma=gamma,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    n_steps=n_steps,
                    learning_rate=learning_rate,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
                    policy_kwargs=policy_kwargs,
                    gae_lambda=gae_lambda,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    use_sde=use_sde,
                    sde_sample_freq=sde_sample_freq,
                    normalize_advantage=normalize_advantage,
                    stats_window_size=stats_window_size,
                    target_kl=target_kl,
                    tensorboard_log=tensor_board_folder,
                    device=device,
                    seed=seed
                )

            # Print policy network architecture
            print("Policy architecture:")
            print(agent.policy)

            # Evalluate before BC training
            mean_reward_before, std_reward_before = evaluate_policy(
                model=agent,
                env=env,
                n_eval_episodes=n_eval_episodes,
                deterministic=False,
                render=False,
            )

            # Set new logger
            log_path = os.path.join(model_folder, f"seed_{seed}", "imit_log")
            imit_log = imit_logger.configure(log_path, ["stdout", "csv", "tensorboard"])

            # Train behavioural clone trainer on transitions
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=np.random.default_rng(seed=seed),
                policy=agent.policy,
                device=device,
                custom_logger=imit_log,
            )

            # Start imitation training
            imitation_settings = policy_params["imitation_settings"]
            max_imitation_epochs = imitation_settings["max_train_epochs"]
            bc_trainer.train(n_epochs=max_imitation_epochs, progress_bar=True)

            # Save BC policy
            imitation_folder = os.path.join(model_folder, f"seed_{seed}", "bc_trainer")
            agent.policy = bc_trainer.policy
            agent.save(os.path.join(imitation_folder, "agent_policy"))

            # Eval imitation agent
            mean_reward_after, std_reward_after = evaluate_policy(
                model=agent,
                env=env,
                n_eval_episodes=n_eval_episodes,
                deterministic=False,
                render=False,
            )

            print(f"Mean reward of BC agent before training: {mean_reward_before}")
            print(f"Std of Reward: {std_reward_before}")
            print(f"Mean reward of BC agent after training: {mean_reward_after}")
            print(f"Std of Reward: {std_reward_after}")

            # Create the callback: autosave every few steps
            auto_save_callback = AutoSave(
                check_freq=autosave_freq,
                num_envs=num_envs,
                save_path=os.path.join(model_folder, f"seed_{seed}"),
                filename_prefix=model_checkpoint + "_"
            )
            eval_callback = custom_callbacks.DiambraEvalCallback(verbose=0)
            callback_list = CallbackList([auto_save_callback, eval_callback])

            agent.learn(
                total_timesteps=time_steps,
                callback=callback_list,
                reset_num_timesteps=False,
                progress_bar=True,
            )
            env.close()

            # Save the agent
            model_checkpoint = str(int(model_checkpoint) + time_steps)
            model_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            agent.save(model_path)

            if not train_id or char_transfer:
                eval_envs = envs_settings[:epoch]
            else:
                eval_envs = [epoch_settings]

            mean_rwd_results = []
            std_rwd_results = []
            for eval_settings in eval_envs:
                env, num_envs = make_sb3_env(eval_settings.game_id, eval_settings, wrappers_settings, seed=seed)
                if epoch_settings.action_space == SpaceTypes.DISCRETE:
                    env = custom_wrappers.VecEnvDiscreteTransferWrapper(env)
                else:
                    env = custom_wrappers.VecEnvMDTransferWrapper(env)
                env = VecTransposeImage(env)

                mean_reward, std_reward = evaluate_policy(
                    model=agent,
                    env=env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=False,
                    render=False,
                )

                env.close()

                mean_rwd_results.append(mean_reward)
                std_rwd_results.append(std_reward)

            mean_rwd = sum(mean_rwd_results) / len(mean_rwd_results)
            std_rwd = sum(std_rwd_results) / len(std_rwd_results)
            print("Evaluation Reward: {} (avg) ± {} (std)".format(mean_rwd, std_rwd))
            eval_results[seed].update({
                epoch: {
                    "mean_rwd": mean_rwd,
                    "std_rwd": std_rwd
                }
            })


    # Save results
    file_path = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "model",
        "evaluation_results.json"
    )
    with open(file_path, "w") as f:
        json.dump(eval_results, f, indent=4)


    print("-----------------------------")
    print("-----Evaluation Results------")
    print("-----------------------------")
    print("----------See Plot-----------")
    print("-----------------------------")

    x = np.linspace(1, len(envs_settings), num=len(envs_settings))
    colours = ["r", "g", "b", "y", "m", "c", "k"]
    for idx, seed in enumerate(seeds):
        mean = [eval_results[seed][epoch]["mean_rwd"] for epoch in eval_results[seed]]
        std = [eval_results[seed][epoch]["std_rwd"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean, std)]
        neg_std = [ya - yb for ya, yb in zip(mean, std)]
        plt.plot(x, mean, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.grid()
    plt.legend()
    plt.ylabel("Average Reward Across Evaluation Episodes")
    if train_id:
        if char_transfer:
            plt.xlabel("Number of Characters")
        else:
            plt.xlabel("Training Episodes")
    else:
        plt.xlabel("Number of Games")
    plt.show()
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--datasetPath", type=str, required=True, help="Path to imitation datasets")
    parser.add_argument("--trainID", type=str, required=False, help="Specific game to train on")
    parser.add_argument('--charTransfer', type=int, required=True, help="Evaluate character transfer or not")
    opt = parser.parse_args()

    main(opt.policyCfg, opt.settingsCfg, opt.datasetPath, opt.trainID, bool(opt.charTransfer))
