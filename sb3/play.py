import os
import yaml
import json
import argparse
import torch
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO, DQN
import custom_wrappers

# PPO
# diambra run python sb3/play.py --policyCfg config_files/transfer-cfg-ppo.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --gameID _

# DQN
# diambra run python sb3/play.py --policyCfg config_files/transfer-cfg-dqn.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --gameID _

def main(policy_cfg: str, settings_cfg: str, game_id: str):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Create environment
    env, num_envs = make_sb3_env(
        settings.game_id,
        settings,
        wrappers_settings,
        render_mode="human",
        no_vec=True,
        use_subprocess=False
    )
    if settings.action_space == SpaceTypes.DISCRETE:
        env = custom_wrappers.DiscretePlayWrapper(env)
    else:
        env = custom_wrappers.MDPlayWrapper(env)

    # Policy param
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}

    # agent = PPO.load(
    #     r"/sb3/transfer_agents/test_ppo_agent_cnn/model/seed_0/0_autosave_6000000",
    #     env=env,
    #     device=device,
    #     policy_kwargs=policy_kwargs,
    #     custom_objects={
    #         "action_space" : env.action_space,
    #         "observation_space" : env.observation_space,
    #     }
    # )
    # agent = PPO.load(
    #     os.path.join(
    #         model_folder,
    #         f"seed_{policy_params['ppo_settings']['seeds'][0]}",
    #         policy_params["ppo_settings"]["model_checkpoint"]
    #     ),
    #     env=env,
    #     device=device,
    #     policy_kwargs=policy_kwargs,
    #     custom_objects={
    #         "action_space" : env.action_space,
    #         "observation_space" : env.observation_space,
    #     }
    # )
    agent = DQN.load(
        r"D:\University\Qmul 24-25\ECS750P MSc Thesis\Diambra\sb3\transfer_agents\test_dqn_agent_cnn\model\seed_0\250000_autosave_300000",
        env=env,
        policy_kwargs=policy_kwargs,
        device=device,
        custom_objects={
            "action_space" : env.action_space,
            "observation_space" : env.observation_space,
        }
    )
    # agent = DQN.load(
    #     os.path.join(
    #         model_folder,
    #         f"seed_{policy_params['dqn_settings']['seeds'][0]}",
    #         policy_params["dqn_settings"]["model_checkpoint"]
    #     ),
    #     env=env,
    #     policy_kwargs=policy_kwargs,
    #     device=device,
    #     custom_objects={
    #         "action_space" : env.action_space,
    #         "observation_space" : env.observation_space,
    #     }
    # )

    obs, _ = env.reset()
    while True:
        env.render()
        action, _state = agent.predict(obs, deterministic=True)
        observation, reward, done, trunc, info = env.step(int(action))
        if done or trunc:
            break

    env.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--gameID", type=str, required=True, help="Game to evaluate")
    opt = parser.parse_args()

    main(opt.policyCfg, opt.settingsCfg, opt.gameID)