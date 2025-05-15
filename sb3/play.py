import os
import yaml
import json
import argparse
import torch
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO, DQN, A2C

# PPO
# diambra run -r "D:\University\Qmul 24-25\ECS750P MSc Thesis\Diambra\roms" python play.py --cfgFile config_files/_/test_ppo_cfg.yaml

# A2C
# diambra run -r "D:\University\Qmul 24-25\ECS750P MSc Thesis\Diambra\roms" python play.py --cfgFile config_files/_/test_a2c_cfg.yaml

# DQN
# diambra run -r "D:\University\Qmul 24-25\ECS750P MSc Thesis\Diambra\roms" python play.py --cfgFile config_files/_/test_dqn_cfg.yaml

def main(cfg_file):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, render_mode="human")
    print("Activated {} environment(s)".format(num_envs))

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")
    ppo_settings = params["ppo_settings"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    # Policy param
    # policy_kwargs = params["policy_kwargs"]
    policy_kwargs = {}
    # agent = PPO.load(os.path.join(model_folder, model_checkpoint), env=env, device=device, policy_kwargs=policy_kwargs)
    agent = PPO.load(r'/results/sfiii3n/sfiii3n_ryu_ppo_no_action_stack/model/500000.zip', env=env, device=device, policy_kwargs=policy_kwargs)
    # agent = A2C.load(os.path.join(model_folder, model_checkpoint), env=env, device=device, policy_kwargs=policy_kwargs)
    # agent = DQN.load(os.path.join(model_folder, model_checkpoint), env=env, device=device, policy_kwargs=policy_kwargs)

    # Environment reset
    observation = env.reset()

    # Agent-Environment interaction loop
    while True:
        # (Optional) Environment rendering
        env.render()

        action, _state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

        # Episode end (Done condition) check
        if done:
            observation = env.reset()
            break

    # Environment shutdown
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile)