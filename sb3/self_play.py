import os
import yaml
import json
import argparse
import torch
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO

def main(cfg_file):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                       params["folders"]["model_name"], "tb")

    os.makedirs(model_folder, exist_ok=True)

    # Settings
    if params["settings"]["action_space"] == "discrete":
        params["settings"]["action_space"] = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)
    else:
        params["settings"]["action_space"] = (SpaceTypes.MULTI_DISCRETE, SpaceTypes.MULTI_DISCRETE)
    settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    policy_type = ppo_settings["policy_type"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]
    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    gae_lambda = ppo_settings["gae_lambda"]
    ent_coef = ppo_settings["ent_coef"]
    vf_coef = ppo_settings["vf_coef"]
    max_grad_norm = ppo_settings["max_grad_norm"]
    use_sde = ppo_settings["use_sde"]
    sde_sample_freq = ppo_settings["sde_sample_freq"]
    rollout_buffer_class = ppo_settings["rollout_buffer_class"]
    rollout_buffer_kwargs = ppo_settings["rollout_buffer_kwargs"]
    normalize_advantage = ppo_settings["normalize_advantage"]
    stats_window_size = ppo_settings["stats_window_size"]
    target_kl = ppo_settings["target_kl"]
    n_steps = ppo_settings["n_steps"]
    seed = ppo_settings["seed"]

    # Create environment
    # env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, seed=seed)
    # print("Activated {} environment(s)".format(num_envs))
    env = diambra.arena.make(settings.game_id, settings, wrappers_settings)

    if model_checkpoint == "0":
        # Initialize the agent
        agent = PPO(policy=policy_type, env=env, verbose=1, gamma=gamma, n_steps=n_steps, learning_rate=learning_rate,
                    gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    use_sde=use_sde, sde_sample_freq=sde_sample_freq, rollout_buffer_class=rollout_buffer_class,
                    rollout_buffer_kwargs=rollout_buffer_kwargs, normalize_advantage=normalize_advantage,
                    stats_window_size=stats_window_size, policy_kwargs=policy_kwargs, target_kl=target_kl,
                    tensorboard_log=tensor_board_folder, device=device, seed=seed)
    else:
        # Load the trained agent
        agent = PPO.load(os.path.join(model_folder, model_checkpoint), env=env,
                         gamma=gamma, learning_rate=learning_rate, policy_kwargs=policy_kwargs,
                         tensorboard_log=tensor_board_folder, device=device)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = ppo_settings["autosave_freq"]
    auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                  save_path=model_folder, filename_prefix=model_checkpoint + "_")

    # Train the agent
    time_steps = ppo_settings["time_steps"]
    agent.learn(total_timesteps=time_steps, callback=auto_save_callback)

    # Save the agent
    new_model_checkpoint = str(int(model_checkpoint) + time_steps)
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile)