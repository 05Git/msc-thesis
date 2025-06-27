# Imports
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

import torchcam

def main():
    settings_config = "config_files/transfer-cfg-settings.yaml"
    policy_config = "config_files/transfer-cfg-ppo.yaml"
    settings_file = open(settings_config)
    settings_params = yaml.load(settings_file, Loader=yaml.FullLoader)
    settings_file.close()
    policy_file = open(policy_config)
    policy_params = yaml.load(policy_file, Loader=yaml.FullLoader)
    policy_file.close()

    # PPO settings
    ppo_settings = policy_params["ppo_settings"]
    seeds = ppo_settings["seeds"]
    # Policy kwargs
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}
    # Wrappers settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    stack_frames = settings_params["wrappers_settings"]["stack_frames"]

    game_id = "sfiii3n"
    settings = settings_params["settings"]["shared"]
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    game_settings = settings_params["settings"][game_id]
    game_settings["characters"] = game_settings["characters"][0]
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettings, settings)

    utils.set_global_seed(seed=seeds[0])
    env, _ = utils.make_sb3_envs(
        game_id=game_id,
        num_train_envs=1,
        num_eval_envs=1,
        env_settings=settings,
        wrappers_settings=wrappers_settings,
        seed=seeds[0],
        no_vec=False, # We want a DummyVecEnv
    )
    if settings.action_space == SpaceTypes.DISCRETE:
        env = custom_wrappers.DiscreteTransferWrapper(env, stack_frames)
    else:
        env = custom_wrappers.MDTransferWrapper(env, stack_frames)
    env = VecTransposeImage(env)
    # Set up agent
    path = "sb3/ppo_agents/sf3_ryu_bc_easy/model/seed_0/game_transfer/5000000.zip"
    agent = PPO.load(
        path,
        env=env,
        policy_kwargs=policy_kwargs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        custom_objects={
            "action_space" : env.action_space,
            "observation_space" : env.observation_space,
        }
    )
    cnn = agent.policy.features_extractor.cnn 
    print(cnn)  # Check network architecture
    env.close()


if __name__ == "__main__":
    main()