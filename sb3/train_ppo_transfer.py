import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import custom_wrappers

# diambra run -s 8 python sb3/train_ppo_transfer.py ---policyCfg config_files/transfer-cfg-ppo.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --trainID _ --charTransfer/--no-charTransfer

def main(policy_cfg: str, settings_cfg: str, train_id: str | None, char_transfer: bool):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read the cfg files
    policy_file = open(policy_cfg)
    policy_params = yaml.load(policy_cfg, Loader=yaml.FullLoader)
    policy_file.close()

    settings_file = open(settings_cfg)
    settings_params = yaml.load(settings_cfg, Loader=yaml.FullLoader)
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


    # Sequential training: Train model sequentially on games, then eval on each one trained so far
    # Single game training: Focus on one particular game
    # First, check if there's a train ID. If so, only want one character one game.
    # If no train ID, check char or game.
    # Load an environment for each game or each character.
    # Train for each base env, eval on each env or each char.

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

    # Policy kwargs
    policy_kwargs = policy_params["policy_kwargs"]

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
                env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings.update(game_settings))
                envs_settings.append(env_settings)
        else:
            train_epochs = len(game_settings["characters"])
            game_settings["characters"] = game_settings["characters"][0]
            env_settings = settings.copy()
            env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings.update(game_settings))
            envs_settings = [env_settings for _ in range(train_epochs)]
    else:
        for game_id in game_ids:
            game_settings = settings_params["settings"][game_id]
            game_settings["characters"] = game_settings["characters"][0]
            env_settings = settings.copy()
            env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings.update(game_settings))
            envs_settings.append(env_settings)

    eval_results = {}
    for seed in settings["seeds"]:
        for epoch in range(len(envs_settings)):
            epoch_settings = envs_settings[epoch]
            env, num_envs = make_sb3_env(epoch_settings.game_id, epoch_settings, wrappers_settings, seed=seed)
            if epoch_settings.action_space == SpaceTypes.DISCRETE:
                env = custom_wrappers.VecEnvDiscreteTransferActionWrapper(env)
            else:
                env = custom_wrappers.VecEnvMDTransferActionWrapper(env)
            print(f"\nOriginal action space: {env.unwrapped.action_space}")
            print(f"Wrapped action space: {env.action_space}")
            print("\nActivated {} environment(s)".format(num_envs))

            # Load policy params if checkpoint exists, else make a new agent
            checkpoint_path = os.path.join(model_folder, model_checkpoint)
            if int(model_checkpoint) > 0 and os.path.exists(checkpoint_path):
                print("\n Checkpoint found, loading model.")
                agent = PPO.load(
                    os.path.join(model_folder, model_checkpoint),
                    env=env,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensor_board_folder,
                    device=device,
                    custom_objects={ "action_space" : env.action_space }
                )
            else:
                print("\n No or invalid checkpoint given, creating new model")
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

            # Create the callback: autosave every few steps
            auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                          save_path=model_folder, filename_prefix=model_checkpoint + "_")

            agent.learn(total_timesteps=time_steps, callback=auto_save_callback)
            env.close()

            if not train_id or char_transfer:
                eval_envs = envs_settings[:epoch]
            else:
                eval_envs = [epoch_settings]

            mean_rwd_results = []
            std_rwd_results = []
            for eval_settings in eval_envs:
                env, num_envs = make_sb3_env(eval_settings.game_id, eval_settings, wrappers_settings, seed=seed)
                if epoch_settings.action_space == SpaceTypes.DISCRETE:
                    env = custom_wrappers.VecEnvDiscreteTransferActionWrapper(env)
                else:
                    env = custom_wrappers.VecEnvMDTransferActionWrapper(env)
                mean_reward, std_reward = evaluate_policy(
                    model=agent,
                    env=env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=True
                )
                env.close()
                mean_rwd_results.append(mean_reward)
                std_rwd_results.append(std_reward)

            mean_rwd = sum(mean_rwd_results) / len(mean_rwd_results)
            std_rwd = sum(std_rwd_results) / len(std_rwd_results)
            print("Evaluation Reward: {} (avg) ± {} (std)".format(mean_rwd, std_rwd))
            eval_results.update({
                seed: {
                    epoch: {
                        "mean_rwd": mean_rwd,
                        "std_rwd": std_rwd
                    }
                }
            })

            # Save the agent
            model_checkpoint = str(int(model_checkpoint) + time_steps)
            model_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            agent.save(model_path)


    print("-----------------------------")
    print("-----Evaluation Results------")
    print("-----------------------------")
    print("----------See Plot-----------")
    print("-----------------------------")

    x = np.linspace(1, len(envs_settings), num=len(envs_settings))
    colours = ["r", "g", "b", "y", "m", "c", "k"]
    for seed, idx in enumerate(settings["seed"]):
        mean = [eval_results[seed][epoch]["mean_rwd"] for epoch in eval_results[seed]]
        std = [eval_results[seed][epoch]["std_rwd"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean, std)]
        neg_std = [ya - yb for ya, yb in zip(mean, std)]
        plt.plot(x, mean, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.grid()
    plt.legend()
    plt.ylabel("Average Reward Across 10 Evaluation Episodes")
    if train_id:
        if char_transfer:
            plt.xlabel("Number of Characters")
        else:
            plt.xlabel("Training Episodes")
    else:
        plt.xlabel("Number of Games")
    plt.show()

    # Save results
    file = { "evaluation_results.json" : eval_results }
    with open(model_folder, "w") as f:
        json.dump(file, f, indent=4)

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--trainID", type=str, required=False, help="Specific game to train on")
    parser.add_argument('--charTransfer', action=argparse.BooleanOptionalAction, required=True, help="Evaluate character transfer or not")
    opt = parser.parse_args()
    print(opt)

    main(opt.policyCfg, opt.settingsCfg, opt.trainID, opt.charTransfer)
