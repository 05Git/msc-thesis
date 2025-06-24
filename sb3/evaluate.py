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

# diambra run -s 4 python sb3/evaluate.py --settingsCfg config_files/settings-cfg.yaml --policyCfg config_files/ppo-cfg.yaml --evalCfg config_files/eval-cfg.py --deterministic --dirName _

def main(policy_cfg: str, settings_cfg: str, eval_cfg: str, deterministic: bool, dir_name: str):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "samsh5sp"
        "kof98umh",
        "umk3",
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
    eval_id = eval_params["eval_id"]
    assert eval_id in game_ids, f"({eval_id} not in valid game IDs: {game_ids})"

    # envs_settings = []
    settings = settings_params["settings"]["shared"]
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    game_settings = settings_params["settings"][eval_id]
    game_settings["characters"] = eval_params["characters"][0]
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettings, settings)
    
    # Load wrappers settings as dictionary
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])

    eval_results = {}
    eval_characters = eval_params["characters"]
    for seed in seeds:
        eval_results.update({seed : {}})
        utils.set_global_seed(seed)

        env, num_envs = make_sb3_env(
            game_id=settings.game_id,
            env_settings=settings,
            wrappers_settings=wrappers_settings,
            seed=seed
        )
        if settings.action_space == SpaceTypes.DISCRETE:
            env = custom_wrappers.VecEnvDiscreteTransferWrapper(
                venv=env,
                stack_frames=wrappers_settings.stack_frames,
            )
        else:
            env = custom_wrappers.VecEnvMDTransferWrapper(
                venv=env,
                stack_frames=wrappers_settings.stack_frames,
            )
        env = VecTransposeImage(env)

        reward_infos = np.zeros(len(model_paths), dtype=np.float64)
        episode_lengths = np.zeros(len(model_paths), dtype=np.float64)
        stages_infos = np.zeros(len(model_paths), dtype=np.float64)
        arcade_infos = np.zeros(len(model_paths), dtype=np.float64)
        for idx, path in enumerate(model_paths):
            agent = PPO.load(
                path=path,
                env=env,
                policy_kwargs=policy_kwargs,
                device=device,
                custom_objects={
                    "action_space": env.action_space,
                    "observation_space": env.observation_space,
                }
            )

            episode_options = {
                "characters": eval_characters[idx] # Evaluate next character
            }
            env.env_method("reset", options=episode_options)

            reward_infos[idx], episode_lengths[idx], stages_infos[idx], arcade_infos[idx] = custom_callbacks.evaluate_policy_with_arcade_metrics(
                model=agent,
                env=env,
                n_eval_episodes=n_eval_episodes * num_envs,
                deterministic=deterministic,
                return_episode_rewards=True,
            )

        eval_results[seed].update({
            "characters": eval_characters,
            "rewards_infos": reward_infos.tolist(),
            "episode_lengths": episode_lengths.tolist(),
            "stages_infos": stages_infos.tolist(),
            "arcade_runs_infos": arcade_infos.tolist(),
            "mean_reward": reward_infos.mean(axis=1).tolist(),
            "std_reward": reward_infos.std(axis=1).tolist(),
            "mean_stages": stages_infos.mean(axis=1).tolist(),
            "std_stages": stages_infos.std(axis=1).tolist(),
            "mean_arcade_runs": arcade_infos.mean(axis=1).tolist(),
            "std_arcade_runs": arcade_infos.std(axis=1).tolist(),
        })
        env.close()

    # Save evaluation results
    base_save_path = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "evaluations",
        dir_name,
    )
    os.makedirs(base_save_path, exist_ok=True)
    json_save_path = os.path.join(base_save_path, "results.json")
    with open(json_save_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    x = np.linspace(1, len(model_paths), num=len(model_paths))
    x_label = ""
    colours = ["r", "g", "b", "y", "m", "c", "k"]

    plt.figure()
    plt.xlabel(x_label)
    for idx, seed in enumerate(seeds):
        mean_rwd = eval_results[seed]["mean_reward"]
        std_rwd = eval_results[seed]["std_reward"]
        pos_std = [sum(y) for y in zip(mean_rwd, std_rwd)]
        neg_std = [ya - yb for ya, yb in zip(mean_rwd, std_rwd)]
        plt.plot(x, mean_rwd, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.ylabel("Average Reward Across Evaluation Episodes")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(base_save_path, "reward_plot.png"))

    plt.figure()
    plt.xlabel(x_label)
    for idx, seed in enumerate(seeds):
        mean_stages = eval_results[seed]["mean_stages"]
        std_stages = eval_results[seed]["std_stages"]
        pos_std = [sum(y) for y in zip(mean_stages, std_stages)]
        neg_std = [ya - yb for ya, yb in zip(mean_stages, std_stages)]
        plt.plot(x, mean_stages, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.ylabel("Average No. of Stages Completed Across Evaluation Episodes")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(base_save_path, "stages_plot.png"))

    plt.figure()
    plt.xlabel(x_label)
    for idx, seed in enumerate(seeds):
        mean_arcade_runs = eval_results[seed]["mean_arcade_runs"]
        std_arcade_runs = eval_results[seed]["std_arcade_runs"]
        pos_std = [sum(y) for y in zip(mean_arcade_runs, std_arcade_runs)]
        neg_std = [ya - yb for ya, yb in zip(mean_arcade_runs, std_arcade_runs)]
        plt.plot(x, mean_arcade_runs, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.ylabel("Average No. of Successful Arcade Runs Across Evaluation Episodes")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(base_save_path, "arcade_runs_plot.png"))
    
    print(f"\nEvaluations complete, see evaluations/{dir_name} for results.")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=False, help="Policy settings config", default="config_files/ppo-cfg.yaml")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/settings-cfg.yaml")
    parser.add_argument("--evalCfg", type=str, required=False, help="Evaluation settings config", default="config_files/eval-cfg.yaml")
    parser.add_argument("--dirName", type=str, required=False, help="Name of evaluations directory", default="evaluation_results")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Evaluate deterministic or stochastic policy", default=True)
    opt = parser.parse_args()

    main(
        policy_cfg=opt.policyCfg, 
        settings_cfg=opt.settingsCfg, 
        eval_cfg=opt.evalCfg,
        deterministic=opt.deterministic,
        dir_name=opt.dirName,
    )
