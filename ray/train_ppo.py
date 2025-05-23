import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

from diambra.arena import SpaceTypes, EnvironmentSettings, WrappersSettings, load_settings_flat_dict
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import pretty_print

import torch
# import custom_wrappers
import utils

def main(policy_cfg: str, settings_cfg: str, train_id: str | None, char_transfer: bool):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    
    if train_id not in game_ids:
        train_id = None

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

    # Policy kwargs
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}

    # PPO settings
    ppo_settings = policy_params["ppo_settings"]
    model_checkpoint = ppo_settings["model_checkpoint"]

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
            config = {
                # Define and configure the environment
                "env": DiambraArena,
                "env_config": {
                    "game_id": epoch_settings.game_id,
                    "settings": epoch_settings,
                    "wrappers_settings": wrappers_settings,
                },
                "num_workers": 0,
                "train_batch_size": 200,
                "framework": "torch",
            }
            # Update config file
            config = preprocess_ray_config(config)

            # Create the RLlib Agent.
            agent = PPO(config=config)
            print("Policy architecture =\n{}".format(agent.get_policy().model))
            print("Training agent...")
            results = agent.train()
            print("\n .. training completed.")
            print("Training results:\n{}".format(pretty_print(results)))

            checkpoint = agent.save().checkpoint.path
            print("Checkpoint saved at {}".format(checkpoint))

            print("\nStarting evaluation ...\n")
            results = agent.evaluate()
            print("\n... evaluation completed.\n")
            print("Evaluation results:\n{}".format(pretty_print(results)))

            return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--trainID", type=str, required=False, help="Specific game to train on")
    parser.add_argument('--charTransfer', type=int, required=True, help="Evaluate character transfer or not")
    opt = parser.parse_args()

    main(opt.policyCfg, opt.settingsCfg, opt.trainID, bool(opt.charTransfer))