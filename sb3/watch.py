import os
import yaml
import json
import argparse
import torch
import numpy as np
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import custom_wrappers
import utils

# diambra run -g python sb3/watch.py --policyCfg config_files/transfer-cfg-ppo.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --gameID _

def main(policy_cfg: str, settings_cfg: str, game_id: str, agent_path: str | None):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load wrappers settings as dictionary
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    # Load shared settings
    settings = settings_params["settings"]["shared"]
    # Set action space type
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings["step_ratio"] = 1
    game_settings = settings_params["settings"][game_id]
    # game_settings["characters"] = game_settings["characters"]["train"][0]
    game_settings["characters"] = "Ryu"
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettings, settings)
    seed = policy_params['ppo_settings']['seeds'][0]

    # Create environment
    env, _ = make_sb3_env(
        game_id=settings.game_id,
        env_settings=settings,
        wrappers_settings=wrappers_settings,
        seed=seed,
        no_vec=True
    )
    utils.set_global_seed(seed)

    if settings.action_space == SpaceTypes.DISCRETE:
        env = custom_wrappers.DiscreteTransferWrapper(env=env, stack_frames=wrappers_settings.stack_frames, characters=[settings.characters])
    else:
        env = custom_wrappers.MDTransferWrapper(env=env, stack_frames=wrappers_settings.stack_frames, characters=[settings.characters])
    env = VecTransposeImage(DummyVecEnv([lambda: env]))

    # Policy param
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}

    agent = PPO.load(
        path=agent_path if agent_path is not None else os.path.join(
            model_folder,
            f"seed_{seed}",
            policy_params["ppo_settings"]["model_checkpoint"]
        ),
        env=env,
        device=device,
        policy_kwargs=policy_kwargs,
        custom_objects={
            "action_space" : env.action_space,
            "observation_space" : env.observation_space,
        }
    )

    obs = env.reset()
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        #if settings.action_space == SpaceTypes.DISCRETE:
        #    action = int(action)
        # print(f"Action: {action}")
        obs, rew, done, info = env.step(action)
        # print(type(info))
        # print(f"Observation: {obs}")
        # print(f"Reward: {rew}")
        # print(f"Dones: {dones}")
        # print(f"Info: {info}")
        
        if done:
            break
    env.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=False, help="Policy config", default="config_files/ppo-cfg.yaml")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/settings-cfg.yaml")
    parser.add_argument("--gameID", type=str, required=False, help="Specific game to evaluate", default="sfiii3n")
    parser.add_argument("--agentPath", type=str, required=False, help="Path to pre-trained agent", default=None)
    opt = parser.parse_args()

    main(opt.policyCfg, opt.settingsCfg, opt.gameID, opt.agentPath)