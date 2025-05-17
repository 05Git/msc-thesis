import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import custom_wrappers

# diambra run -s 8 python sb3/train_ppo_arcade.py --cfgDir config_files --gameID _ --cfgFile base_ppo_cfg.yaml

def main(cfg_dir, train_id, config):
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read the cfg file
    cfg_file = os.path.join(cfg_dir, train_id, config)
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
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    policy = ppo_settings["policy_type"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]
    epochs = ppo_settings["n_train_epochs"]
    n_eval_episodes = ppo_settings["n_eval_episodes"]

    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]
    gae_lambda = ppo_settings["gae_lambda"]
    ent_coef = ppo_settings["ent_coef"]
    vf_coef = ppo_settings["vf_coef"]
    max_grad_norm = ppo_settings["max_grad_norm"]
    use_sde = ppo_settings["use_sde"]
    sde_sample_freq = ppo_settings["sde_sample_freq"]
    normalize_advantage = ppo_settings["normalize_advantage"]
    stats_window_size = ppo_settings["stats_window_size"]
    target_kl = ppo_settings["target_kl"]

    # Train the agent
    time_steps = ppo_settings["time_steps"]
    environment_settings = params["env_settings"]
    seed = environment_settings["seed"]
    train_eval_results = {}
    for epoch in range(1, epochs + 1):
        # Create environment
        env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, seed=seed)
        if params["settings"]["action_space"] == SpaceTypes.DISCRETE:
            env = custom_wrappers.VecEnvDiscreteTransferActionWrapper(env)
        else:
            env = custom_wrappers.VecEnvMDTransferActionWrapper(env)
        print(f"\nOriginal action space: {env.unwrapped.action_space}")
        print(f"Wrapped action space: {env.action_space}")
        print("\nActivated {} environment(s)".format(num_envs))

        if model_checkpoint == "0":
            # Initialize the agent
            agent = PPO(policy, env, verbose=1, gamma=gamma, batch_size=batch_size, n_epochs=n_epochs, n_steps=n_steps,
                        learning_rate=learning_rate, clip_range=clip_range, clip_range_vf=clip_range_vf,
                        policy_kwargs=policy_kwargs,
                        gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                        use_sde=use_sde,
                        sde_sample_freq=sde_sample_freq, normalize_advantage=normalize_advantage,
                        stats_window_size=stats_window_size,
                        target_kl=target_kl, tensorboard_log=tensor_board_folder, device=device, seed=seed)
        else:
            # Load the trained agent
            policy_kwargs = {}
            agent = PPO.load(os.path.join(model_folder, model_checkpoint), env=env,
                             gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                             clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                             tensorboard_log=tensor_board_folder, device=device,
                             custom_objects={"action_space": env.action_space})

        # Print policy network architecture
        print("Policy architecture:")
        print(agent.policy)

        # Create the callback: autosave every few steps
        autosave_freq = ppo_settings["autosave_freq"]
        auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                      save_path=model_folder, filename_prefix=model_checkpoint + "_")

        agent.learn(total_timesteps=time_steps, callback=auto_save_callback)
        mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval_episodes)
        print("Evaluation Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))
        train_eval_results.update({epoch: {"mean_rwd": mean_reward, "std_rwd": std_reward}})

        # Save the agent
        model_checkpoint = str(int(model_checkpoint) + time_steps)
        model_path = os.path.join(model_folder, model_checkpoint)
        agent.save(model_path)

        env.close()

    # Evaluate character transfer
    eval_chars = environment_settings["eval_chars"]
    char_eval_results = {}
    policy_kwargs = {}
    for character in eval_chars:
        params["settings"]["characters"] = character
        settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
        env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, seed=seed)
        if params["settings"]["action_space"] == SpaceTypes.DISCRETE:
            env = custom_wrappers.VecEnvDiscreteTransferActionWrapper(env)
        else:
            env = custom_wrappers.VecEnvMDTransferActionWrapper(env)

        # Load the trained agent
        agent = PPO.load(model_path, env=env,
                         gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                         clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                         tensorboard_log=tensor_board_folder, device=device,
                         custom_objects={"action_space": env.action_space})

        mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval_episodes)
        print("Character: {}\nReward: {} (avg) ± {} (std)".format(character, mean_reward, std_reward))
        char_eval_results.update({character: {"mean_rwd": mean_reward, "std_rwd": std_reward}})
        env.close()

    game_eval_results = {}
    for id in game_ids:
        if id == train_id:
            continue

        # Read the cfg file
        cfg_file = os.path.join(cfg_dir, id, config)
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
        env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, seed=seed)
        if params["settings"]["action_space"] == SpaceTypes.DISCRETE:
            env = custom_wrappers.VecEnvDiscreteTransferActionWrapper(env)
        else:
            env = custom_wrappers.VecEnvMDTransferActionWrapper(env)

        # Load the trained agent
        agent = PPO.load(model_path, env=env,
                         gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                         clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                         tensorboard_log=tensor_board_folder, device=device,
                         custom_objects={"action_space": env.action_space})
        mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval_episodes)
        print("Game: {}\nReward: {} (avg) ± {} (std)".format(id, mean_reward, std_reward))
        game_eval_results.update({id: {"mean_rwd": mean_reward, "std_rwd": std_reward}})
        env.close()

    print("-----------------------------")
    print("----------Training-----------")
    print("-----Evaluation Results------")
    print("-----------------------------")
    print("----------See Plot-----------")
    print("-----------------------------")

    x = np.linspace(1, epochs, num=epochs)
    y_mean = [train_eval_results[e]["mean_rwd"] for e in train_eval_results]
    y_std = [train_eval_results[e]["std_rwd"] for e in train_eval_results]
    plt.plot(x, y_mean, color="r", label="Mean Evaluation Reward")
    plt.plot(x, y_std, color="g", label="Standard Deviation of Evaluation Reward")
    plt.legend()
    plt.grid()
    plt.show()

    print("-----------------------------")
    print("------Character Transfer-----")
    print("------Evaluation Results-----")
    print("-----------------------------")
    for char, results in char_eval_results.items():
        print("-----------------------------")
        print(f"Character:     {char}")
        print(f"Mean Reward:   {results['mean_rwd']}")
        print(f"Std of Reward: {results['std_rwd']}")
        print("-----------------------------")

    print("-----------------------------")
    print("--------Game Transfer--------")
    print("------Evaluation Results-----")
    print("-----------------------------")
    for id, results in game_eval_results.items():
        print("-----------------------------")
        print(f"Character:     {id}")
        print(f"Mean Reward:   {results['mean_rwd']}")
        print(f"Std of Reward: {results['std_rwd']}")
        print("-----------------------------")

    files = {
        "train_eval_results.json": train_eval_results,
        "char_transfer_eval_results.json": char_eval_results,
        "game_transfer_eval_results.json": game_eval_results,
    }

    # Save results
    for filename, file in files.items():
        with open(model_folder, "w") as f:
            json.dump(file, f, indent=4)

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgDir", type=str, required=True, help="Config file directory")
    parser.add_argument("--gameID", type=str, required=True, help="Game to train on")
    parser.add_argument("--cfgFile", type=str, required=True, help="Config file for training")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgDir, opt.gameID, opt.cfgFile)
