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
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList

import custom_wrappers
import custom_callbacks
import utils

# diambra run -s 8 python sb3/evaluate_ppo.py --settingsCfg config_files/transfer-cfg-settings.yaml --policyCfg config_files/transfer-cfg-ppo.yaml --evalCfg config_files/eval_cgf.yaml --evalID _ 

def main(policy_cfg: str, settings_cfg: str, eval_cfg: str, eval_id: str | None):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    
    if eval_id not in game_ids:
        eval_id = None
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
    # Load wrappers settings as dictionary
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    stack_frames = settings_params["wrappers_settings"]["stack_frames"]
    # Load shared settings
    settings = settings_params["settings"]["shared"]
    # Set action space type
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    game_settings = settings_params["settings"][eval_id]

    # Evaluation settings
    model_paths = eval_params["model_paths"]
    eval_chars = eval_params["eval_chars"][eval_id]

    eval_results = {}
    for seed in seeds:
        eval_results.update({seed : {}})
        utils.set_global_seed(seed)
        for idx_1, path in enumerate(model_paths):
            # Initialize array of characters to evaluate
            chars_to_eval = eval_chars[:idx_1 + 1]

            # Initialize vectors to store evaluation info
            mean_rewards, std_rewards = np.zeros(idx_1 + 1, dtype=np.float64), np.zeros(idx_1 + 1, dtype=np.float64)
            mean_stages, std_stages = np.zeros(idx_1 + 1, dtype=np.float64), np.zeros(idx_1 + 1, dtype=np.float64)
            mean_arcade_runs, std_arcade_runs = np.zeros(idx_1 + 1, dtype=np.float64), np.zeros(idx_1 + 1, dtype=np.float64)
            for idx_2, char in enumerate(chars_to_eval):
                # Set up env
                game_settings["characters"] = char
                settings.update(game_settings)
                env_settings = load_settings_flat_dict(EnvironmentSettings, settings)
                env, num_envs = make_sb3_env(env_settings.game_id, env_settings, wrappers_settings, seed=seed)
                if env_settings.action_space == SpaceTypes.DISCRETE:
                    env = custom_wrappers.VecEnvDiscreteTransferWrapper(env, stack_frames)
                else:
                    env = custom_wrappers.VecEnvMDTransferWrapper(env, stack_frames)
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
                    n_eval_episodes=n_eval_episodes,
                    deterministic=False,
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

            # Print and save results for plotting later
            print("Evaluation Reward: {} (avg) ± {} (std)".format(mean_rewards, std_rewards))
            print("Evaluation Stages Completed: {} (avg) ± {} (std)".format(mean_stages, std_stages))
            print("Evaluation Arcade Runs Completed: {} (avg) ± {} (std)".format(mean_arcade_runs, std_arcade_runs))
            eval_results[seed].update({
                f"Characters: {idx_1 + 1}": {
                    "mean_rwd": mean_rewards,
                    "std_rwd": std_rewards,
                    "mean_stages": mean_stages,
                    "std_stages": std_stages,
                    "mean_arcade_runs": mean_arcade_runs,
                    "std_arcade_runs": std_arcade_runs,
                }
            })

    # Save evaluation results
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
    print("----------See Plots----------")
    print("-----------------------------")

    x = np.linspace(1, len(eval_chars), num=len(eval_chars))
    colours = ["r", "g", "b", "y", "m", "c", "k"]
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False)
    for idx, seed in enumerate(seeds):
        mean_rwd = [eval_results[seed][epoch]["mean_rwd"] for epoch in eval_results[seed]]
        std_rwd = [eval_results[seed][epoch]["std_rwd"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean_rwd, std_rwd)]
        neg_std = [ya - yb for ya, yb in zip(mean_rwd, std_rwd)]
        axs[0].plot(x, mean_rwd, color=colours[idx], label=f"Seed: {seed}")
        axs[0].fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)

        mean_stages = [eval_results[seed][epoch]["mean_stages"] for epoch in eval_results[seed]]
        std_stages = [eval_results[seed][epoch]["std_stages"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean_stages, std_stages)]
        neg_std = [ya - yb for ya, yb in zip(mean_stages, std_stages)]
        axs[1].plot(x, mean_stages, color=colours[idx], label=f"Seed: {seed}")
        axs[1].fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)

        mean_arcade_runs = [eval_results[seed][epoch]["mean_arcade_runs"] for epoch in eval_results[seed]]
        std_arcade_runs = [eval_results[seed][epoch]["std_arcade_runs"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean_arcade_runs, std_arcade_runs)]
        neg_std = [ya - yb for ya, yb in zip(mean_arcade_runs, std_arcade_runs)]
        axs[2].plot(x, mean_arcade_runs, color=colours[idx], label=f"Seed: {seed}")
        axs[2].fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)

    axs[0].set_ylabel("Average Reward Across Evaluation Episodes")
    axs[1].set_ylabel("Average No. of Stages Completed Across Evaluation Episodes")
    axs[2].set_ylabel("Average No. of Successful Arcade Runs Across Evaluation Episodes")
    if eval_id:
        x_label = "Number of Characters"
    else:
        x_label = "Number of Games"

    for ax in axs:
        ax.set_xlabel(x_label)
        ax.grid(True)
        ax.legend()

    plt.savefig("eval_results.png")
    plt.show()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy settings config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--evalCfg", type=str, required=True, help="Evaluation settings config")
    parser.add_argument("--evalID", type=str, required=False, help="Specific game to evaluate")
    opt = parser.parse_args()

    main(opt.policyCfg, opt.settingsCfg, opt.evalCfg, opt.evalID)
