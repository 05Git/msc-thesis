import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent, WrappersSettings, load_settings_flat_dict
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

import torch
import custom_wrappers
import utils

# diambra run -g -s 9 python ray/self_play_ppo.py --policyCfg config_files/self-play-ppo.yaml --settingsCfg config_files/self-play-settings.yaml --trainID _

def env_creator(env_config):
    env = custom_wrappers.DiscreteTransferWrapper(DiambraArena(env_config), stack_frames=4)
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    return env

def main(policy_cfg: str, settings_cfg: str, train_id: str):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    
    if train_id not in game_ids:
        train_id = game_ids[0]

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
    time_steps = ppo_settings["time_steps"]
    seeds = ppo_settings["seeds"]

    # Load wrappers settings as dictionary
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    stack_frames = settings_params["wrappers_settings"]["stack_frames"]
    # Load shared settings
    settings = settings_params["settings"]["shared"]
    # Set action space type
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings["action_space"] = (settings["action_space"], settings["action_space"])

    # Load game specific settings
    game_settings = settings_params["settings"][train_id]
    game_settings["characters"] = (game_settings["characters"][0], game_settings["characters"][0])
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, settings)

    envs_settings = [settings]
    register_env("DiambraArena", env_creator)
    eval_results = {}
    for seed in seeds:
        eval_results.update({seed: {}})
        utils.set_global_seed(seed)
        for epoch in range(len(envs_settings)):
            epoch_settings = envs_settings[epoch]
            config = {
                # Define and configure the environment
                "env": "DiambraArena",
                "env_config": {
                    "game_id": epoch_settings.game_id,
                    "settings": epoch_settings,
                    "wrappers_settings": wrappers_settings,
                },
                "num_workers": 3,
                "num_envs_per_worker": 2,
                "train_batch_size": time_steps,
                "framework": "torch",
                # Evaluate once per two training iterations
                "evaluation_interval": 2,
                # Run evaluation on (at least) one episode
                "evaluation_duration": 1,
                "evaluation_num_workers": 1,
                "evaluation_config": {
                    "render_env": False,
                },
            }
            # Update config file
            config = preprocess_ray_config(config)
            
            # Create the RLlib Agent.
            agent = PPO(config=config)
            # print("Policy architecture =\n{}".format(agent.get_policy().model))
            
            for i in range(1, 2):
                print("Epoch {}".format(i))
                print("Training agent...")
                results = agent.train()
                print("\n .. training completed.")
                print("Training results:\n{}".format(pretty_print(results)))

            # Save agent
            model_checkpoint = str(int(model_checkpoint) + time_steps)
            model_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            agent.save(model_path)
            print("Agent saved at {}".format(model_path))
            
            # # Evaluate
            # print("\nStarting evaluation ...\n")
            # results = agent.evaluate()
            # print("\n... evaluation completed.\n")
            # print("Evaluation results:\n{}".format(pretty_print(results)))

            return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--trainID", type=str, required=True, help="Specific game to train on")
    opt = parser.parse_args()

    main(opt.policyCfg, opt.settingsCfg, opt.trainID)