import os
import yaml
import json
import argparse
import torch
import numpy as np
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import custom_wrappers
import utils

# diambra run -g python sb3/play_recurrent_ppo.py --policyCfg config_files/transfer-cfg-ppo.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --gameID _

def main(policy_cfg: str, settings_cfg: str, game_id: str):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    stack_frames = settings_params["wrappers_settings"]["stack_frames"]
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    # Load shared settings
    settings = settings_params["settings"]["shared"]
    # Set action space type
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings["step_ratio"] = 1
    game_settings = settings_params["settings"][game_id]
    game_settings["characters"] = game_settings["characters"][0]
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettings, settings)

    seed = policy_params['ppo_settings']['seeds'][0]
    utils.set_global_seed(seed)

    # Create environment
    env, num_envs = make_sb3_env(
        settings.game_id,
        settings,
        wrappers_settings,
        seed=seed,
        no_vec=True
    )
    if settings.action_space == SpaceTypes.DISCRETE:
        env = custom_wrappers.DiscreteTransferWrapper(env, stack_frames=stack_frames)
    else:
        env = custom_wrappers.MDTransferWrapper(env, stack_frames=stack_frames)
    env = VecTransposeImage(DummyVecEnv([lambda: env]))

    # Policy param
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}
    
    agent = RecurrentPPO.load(
        os.path.join(
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

    # obs = env.reset()
    # lstm_states = None
    # episode_starts = np.ones((num_envs,), dtype=bool)
    # while True:
    #     action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
    #     print(f"Action: {action}")
    #     obs, rew, dones, info = env.step(action)
    #     episode_starts = dones
    #     if dones:
    #         break

    mean_reward, std_reward = evaluate_policy(
        model=agent,
        env=env,
        n_eval_episodes=5,
        deterministic=False,
        render=False
    )
    print(f"Mean reward: {mean_reward}")
    print(f"Std of reward: {std_reward}")

    env.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--gameID", type=str, required=True, help="Game to evaluate")
    opt = parser.parse_args()

    main(opt.policyCfg, opt.settingsCfg, opt.gameID)