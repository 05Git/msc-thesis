import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

import custom_wrappers
import custom_callbacks
import utils

# diambra run -s 8 python sb3/evaluate_ppo.py --settingsCfg config_files/transfer-cfg-settings.yaml --policyCfg config_files/transfer-cfg-ppo.yaml --evalCfg config_files/eval-cfg.py --deterministic

def main(policy_cfg: str, settings_cfg: str, eval_cfg: str, deterministic: bool):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read the cfg files
    policy_file = open(policy_cfg)
    policy_params = yaml.load(policy_file, Loader=yaml.FullLoader)
    policy_file.close()

    settings_file = open(settings_cfg)
    settings_params = yaml.load(settings_file, Loader=yaml.FullLoader)
    settings_file.close()

    eval_file = open(eval_cfg)
    eval_params = yaml.load(eval_file, Loader=yaml.FullLoader)
    eval_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))

    # PPO settings
    ppo_settings = policy_params["ppo_settings"]
    seeds = ppo_settings["seeds"]
    n_eval_episodes = ppo_settings["n_eval_episodes"]

    # Policy kwargs
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}

    # Evaluation settings
    model_paths = eval_params["model_paths"]
    eval_type = eval_params["transfer_type"]
    eval_id = eval_params["eval_id"]
    if eval_type == "game" or eval_id not in game_ids:
        eval_id = None

    envs_settings = []
    settings = settings_params["settings"]["shared"]
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    if eval_id:
        game_settings = settings_params["settings"][eval_id].copy()
        if eval_params["transfer_type"] == "character":
            for char in eval_params["characters"]:
                game_settings["characters"] = char
                env_settings = settings.copy()
                env_settings.update(game_settings)
                envs_settings.append(env_settings)
        else:
            game_settings["characters"] = eval_params["characters"][0]
            env_settings = settings.copy()
            env_settings.update(game_settings)
            env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings)
            envs_settings.append(env_settings)
    else:
        for game_list in eval_params["games"]:
            env_settings = settings.copy()
            env_settings["characters"] = {}
            for game_id in game_list:
                game_settings = settings_params["settings"][game_id].copy()
                game_settings["characters"] = { game_id : game_settings["characters"][0] }
                env_settings.update(game_settings)
            envs_settings.append(env_settings)
    
    # Load wrappers settings as dictionary
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])

    eval_results = {}
    for seed in seeds:
        eval_results.update({seed : {}})
        utils.set_global_seed(seed)

        if eval_type == "game":
            # Cross game transfer
            for idx_1, path in enumerate(model_paths):
                eval_settings = envs_settings[idx_1].copy()
                eval_characters = eval_settings["characters"].copy()
                # Initialize vectors to store evaluation info
                mean_rewards, std_rewards = np.zeros(idx_1 + 1, dtype=np.float64), np.zeros(idx_1 + 1, dtype=np.float64)
                mean_stages, std_stages = np.zeros(idx_1 + 1, dtype=np.float64), np.zeros(idx_1 + 1, dtype=np.float64)
                mean_arcade_runs, std_arcade_runs = np.zeros(idx_1 + 1, dtype=np.float64), np.zeros(idx_1 + 1, dtype=np.float64)
                for idx_2, game_id in enumerate(eval_characters.keys()):
                    # Set up env
                    eval_settings["game_id"] = game_id
                    eval_settings["characters"] = eval_characters[game_id]
                    settings = load_settings_flat_dict(EnvironmentSettings, eval_settings)
                    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, seed=seed)
                    if settings.action_space == SpaceTypes.DISCRETE:
                        env = custom_wrappers.VecEnvDiscreteTransferWrapper(env, stack_frames=wrappers_settings.stack_frames)
                    else:
                        env = custom_wrappers.VecEnvMDTransferWrapper(env, stack_frames=wrappers_settings.stack_frames)
                    env = VecTransposeImage(env)

                    # Set up agent
                    agent = PPO.load(
                        path,
                        env=env,
                        policy_kwargs=policy_kwargs,
                        device=device,
                        custom_objects={
                            "action_space" : env.action_space,
                            "observation_space" : env.observation_space,
                        }
                    )

                    rwd_info, stages_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
                        model=agent,
                        env=env,
                        n_eval_episodes=n_eval_episodes * num_envs,
                        deterministic=deterministic,
                        render=False,
                    )
                    env.close()

                    # Store evaluation info
                    mean_rewards[idx_2], std_rewards[idx_2] = rwd_info
                    mean_stages[idx_2], std_stages[idx_2] = stages_info
                    mean_arcade_runs[idx_2], std_arcade_runs[idx_2] = arcade_info
                
                # Average out results across characters
                mean_rewards, std_rewards = np.mean(mean_rewards), np.mean(std_rewards)
                mean_stages, std_stages = np.mean(mean_stages), np.mean(std_stages)
                mean_arcade_runs, std_arcade_runs = np.mean(mean_arcade_runs), np.mean(std_arcade_runs)
                eval_results[seed].update({
                    f"Model: {path}, Games: {list(eval_characters.keys())}": {
                        "mean_reward": mean_rewards,
                        "std_reward": std_rewards,
                        "mean_stages": mean_stages,
                        "std_stages": std_stages,
                        "mean_arcade_runs": mean_arcade_runs,
                        "std_arcade_runs": std_arcade_runs,
                    }
                })
        elif eval_type == "character":
            # Cross character transfer
            for idx_1, path in enumerate(model_paths):
                eval_settings = envs_settings[idx_1].copy()
                eval_characters = eval_settings["characters"].copy()
                # Initialize vectors to store evaluation info
                mean_rewards, std_rewards = np.zeros(len(eval_characters), dtype=np.float64), np.zeros(len(eval_characters), dtype=np.float64)
                mean_stages, std_stages = np.zeros(len(eval_characters), dtype=np.float64), np.zeros(len(eval_characters), dtype=np.float64)
                mean_arcade_runs, std_arcade_runs = np.zeros(len(eval_characters), dtype=np.float64), np.zeros(len(eval_characters), dtype=np.float64)
                for idx_2, character in enumerate(eval_characters):
                    eval_settings["characters"] = character
                    settings = load_settings_flat_dict(EnvironmentSettings, eval_settings)
                    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, seed=seed)
                    if settings.action_space == SpaceTypes.DISCRETE:
                        env = custom_wrappers.VecEnvDiscreteTransferWrapper(env, stack_frames=wrappers_settings.stack_frames)
                    else:
                        env = custom_wrappers.VecEnvMDTransferWrapper(env, stack_frames=wrappers_settings.stack_frames)
                    env = VecTransposeImage(env)

                    # Set up agent
                    agent = PPO.load(
                        path,
                        env=env,
                        policy_kwargs=policy_kwargs,
                        device=device,
                        custom_objects={
                            "action_space" : env.action_space,
                            "observation_space" : env.observation_space,
                        }
                    )

                    rwd_info, stages_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
                        model=agent,
                        env=env,
                        n_eval_episodes=n_eval_episodes * num_envs,
                        deterministic=deterministic,
                        render=False,
                    )
                    env.close()

                    # Store evaluation info
                    mean_rewards[idx_2], std_rewards[idx_2] = rwd_info
                    mean_stages[idx_2], std_stages[idx_2] = stages_info
                    mean_arcade_runs[idx_2], std_arcade_runs[idx_2] = arcade_info
                
                # Average out results across characters
                mean_rewards, std_rewards = np.mean(mean_rewards), np.mean(std_rewards)
                mean_stages, std_stages = np.mean(mean_stages), np.mean(std_stages)
                mean_arcade_runs, std_arcade_runs = np.mean(mean_arcade_runs), np.mean(std_arcade_runs)
                eval_results[seed].update({
                    f"Model: {path}, Characters: {eval_characters}": {
                        "mean_reward": mean_rewards,
                        "std_reward": std_rewards,
                        "mean_stages": mean_stages,
                        "std_stages": std_stages,
                        "mean_arcade_runs": mean_arcade_runs,
                        "std_arcade_runs": std_arcade_runs,
                    }
                })
        else:
            # Single character, single game evaluation
            eval_settings = envs_settings[0]
            env, num_envs = make_sb3_env(
                game_id=eval_settings.game_id,
                env_settings=eval_settings,
                wrappers_settings=wrappers_settings,
                seed=seed,
            )
            if eval_settings.action_space == SpaceTypes.DISCRETE:
                env = custom_wrappers.VecEnvDiscreteTransferWrapper(env, stack_frames=wrappers_settings.stack_frames)
            else:
                env = custom_wrappers.VecEnvMDTransferWrapper(env, stack_frames=wrappers_settings.stack_frames)
            env = VecTransposeImage(env)
            for path in model_paths:
                agent = PPO.load(
                    path,
                    env=env,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    custom_objects={
                        "action_space" : env.action_space,
                        "observation_space" : env.observation_space,
                    }
                )
                reward_info, stages_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
                    model=agent,
                    env=env,
                    n_eval_episodes=n_eval_episodes * num_envs,
                    deterministic=deterministic,
                    render=False,
                )
                mean_reward, std_reward = reward_info
                mean_stages, std_stages = stages_info
                mean_arcades, std_arcades = arcade_info
                eval_results[seed].update({
                    f"Model: {path}": {
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                        "mean_stages": mean_stages,
                        "std_stages": std_stages,
                        "mean_arcade_runs": mean_arcades,
                        "std_arcade_runs": std_arcades,
                    }
                })
            env.close()

    # Save evaluation results
    save_path = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "model",
    )
    if eval_type == "character":
        save_path = os.path.join(save_path, "char_transfer_results.json")
    elif eval_type == "game":
        save_path = os.path.join(save_path, "game_transfer_results.json")
    else:
        save_path = os.path.join(save_path, "evaluation_results.json")
    with open(save_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print("-----------------------------")
    print("-----Evaluation Results------")
    print("-----------------------------")
    print("----------See Plots----------")
    print("-----------------------------")

    x = np.linspace(1, len(model_paths), num=len(model_paths))
    colours = ["r", "g", "b", "y", "m", "c", "k"]
    if eval_type == "character":
        x_label = "Number of Characters"
    elif eval_type == "game":
        x_label = "Number of Games"
    else:
        x_label = "Model No. Evaluated"
    figure_save_path = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "model",
    )

    plt.figure()
    plt.xlabel(x_label)
    for idx, seed in enumerate(seeds):
        mean_rwd = [eval_results[seed][epoch]["mean_reward"] for epoch in eval_results[seed]]
        std_rwd = [eval_results[seed][epoch]["std_reward"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean_rwd, std_rwd)]
        neg_std = [ya - yb for ya, yb in zip(mean_rwd, std_rwd)]
        plt.plot(x, mean_rwd, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.ylabel("Average Reward Across Evaluation Episodes")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, "reward_plot.png"))

    plt.figure()
    plt.xlabel(x_label)
    for idx, seed in enumerate(seeds):
        mean_stages = [eval_results[seed][epoch]["mean_stages"] for epoch in eval_results[seed]]
        std_stages = [eval_results[seed][epoch]["std_stages"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean_stages, std_stages)]
        neg_std = [ya - yb for ya, yb in zip(mean_stages, std_stages)]
        plt.plot(x, mean_stages, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.ylabel("Average No. of Stages Completed Across Evaluation Episodes")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, "stages_plot.png"))

    plt.figure()
    plt.xlabel(x_label)
    for idx, seed in enumerate(seeds):
        mean_arcade_runs = [eval_results[seed][epoch]["mean_arcade_runs"] for epoch in eval_results[seed]]
        std_arcade_runs = [eval_results[seed][epoch]["std_arcade_runs"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean_arcade_runs, std_arcade_runs)]
        neg_std = [ya - yb for ya, yb in zip(mean_arcade_runs, std_arcade_runs)]
        plt.plot(x, mean_arcade_runs, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.ylabel("Average No. of Successful Arcade Runs Across Evaluation Episodes")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, "arcade_runs_plot.png"))

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=False, help="Policy settings config", default="config_files/transfer-cfg-ppo.yaml")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/transfer-cfg-settings.yaml")
    parser.add_argument("--evalCfg", type=str, required=False, help="Evaluation settings config", default="config_files/eval-cfg.yaml")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Evaluate deterministic or stochastic policy", default=True)
    opt = parser.parse_args()

    main(
        policy_cfg=opt.policyCfg, 
        settings_cfg=opt.settingsCfg, 
        eval_cfg=opt.evalCfg,
        deterministic=opt.deterministic,
    )
