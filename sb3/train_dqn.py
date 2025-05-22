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
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import custom_wrappers
import utils

# diambra run python sb3/train_dqn.py --policyCfg config_files/transfer-cfg-dqn.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --trainID _ --charTransfer/--no-charTransfer

def main(policy_cfg: str, settings_cfg: str, train_id: str | None, char_transfer: bool):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]

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


    # DQN settings
    dqn_settings = policy_params["dqn_settings"]
    policy = dqn_settings["policy_type"]
    gamma = dqn_settings["gamma"]
    model_checkpoint = dqn_settings["model_checkpoint"]
    learning_rate = linear_schedule(dqn_settings["learning_rate"][0], dqn_settings["learning_rate"][1])
    buffer_size = dqn_settings["buffer_size"]
    learning_starts = dqn_settings["learning_starts"]
    batch_size = dqn_settings["batch_size"]
    tau = dqn_settings["tau"]
    train_freq = dqn_settings["train_freq"]
    gradient_steps = dqn_settings["gradient_steps"]
    replay_buffer_class = dqn_settings["replay_buffer_class"]
    replay_buffer_kwargs = dqn_settings["replay_buffer_kwargs"]
    stats_window_size = dqn_settings["stats_window_size"]
    target_update_interval = dqn_settings["target_update_interval"]
    exploration_fraction = dqn_settings["exploration_fraction"]
    exploration_initial_eps = dqn_settings["exploration_initial_eps"]
    exploration_final_eps = dqn_settings["exploration_final_eps"]
    max_grad_norm = dqn_settings["max_grad_norm"]
    autosave_freq = dqn_settings["autosave_freq"]
    time_steps = dqn_settings["time_steps"]
    n_eval_episodes = dqn_settings["n_eval_episodes"]
    seeds = dqn_settings["seeds"]

    # Policy kwargs
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}

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
            env, num_envs = make_sb3_env(
                epoch_settings.game_id,
                epoch_settings,
                wrappers_settings,
                seed=seed,
                no_vec=True,
                use_subprocess=False
            )
            env = custom_wrappers.DiscreteTransferActionWrapper(env)
            print(f"\nOriginal action space: {env.unwrapped.action_space}")
            print(f"Wrapped action space: {env.action_space}")
            print("\nActivated {} environment(s)".format(num_envs))

            # Load policy params if checkpoint exists, else make a new agent
            checkpoint_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            if int(model_checkpoint) > 0 and os.path.exists(checkpoint_path):
                print("\n Checkpoint found, loading model.")
                agent = DQN.load(
                    checkpoint_path,
                    env=env,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensor_board_folder,
                    device=device,
                    custom_objects={
                        "action_space" : env.action_space,
                        "observation_space" : env.observation_space,
                    }
                )
            else:
                print("\nNo or invalid checkpoint given, creating new model")
                agent = DQN(
                    policy,
                    env,
                    verbose=1,
                    gamma=gamma,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    learning_starts=learning_starts,
                    buffer_size=buffer_size,
                    tau=tau,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps,
                    replay_buffer_class=replay_buffer_class,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                    policy_kwargs=policy_kwargs,
                    max_grad_norm=max_grad_norm,
                    stats_window_size=stats_window_size,
                    tensorboard_log=tensor_board_folder,
                    device=device,
                    seed=seed
                )

            # Print policy network architecture
            print("Policy architecture:")
            print(agent.policy)

            # Create the callback: autosave every few steps
            auto_save_callback = AutoSave(
                check_freq=autosave_freq,
                num_envs=num_envs,
                save_path=os.path.join(model_folder, f"seed_{seed}"),
                filename_prefix=model_checkpoint + "_"
            )

            agent.learn(
                total_timesteps=time_steps,
                callback=auto_save_callback,
                reset_num_timesteps=False,
                progress_bar=True
            )
            env.close()

            # Save the agent
            model_checkpoint = str(int(model_checkpoint) + time_steps)
            model_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            agent.save(model_path)

            if not train_id or char_transfer:
                eval_envs = envs_settings[:epoch]
            else:
                eval_envs = [epoch_settings]

            mean_rwd_results = []
            std_rwd_results = []
            for eval_settings in eval_envs:
                env, num_envs = make_sb3_env(
                    eval_settings.game_id,
                    eval_settings,
                    wrappers_settings,
                    seed=seed,
                    no_vec=True,
                    use_subprocess=False
                )
                env = custom_wrappers.DiscreteTransferActionWrapper(env)
                mean_reward, std_reward = evaluate_policy(
                    model=agent,
                    env=env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=True
                )
                env.close()
                mean_rwd_results.append(mean_reward)
                std_rwd_results.append(std_reward)

            mean_rwd = sum(mean_rwd_results) / len(mean_rwd_results)
            std_rwd = sum(std_rwd_results) / len(std_rwd_results)
            print("Evaluation Reward: {} (avg) ± {} (std)".format(mean_rwd, std_rwd))
            eval_results[seed].update({
                epoch: {
                    "mean_rwd": mean_rwd,
                    "std_rwd": std_rwd
                }
            })


    # Save results
    file_path = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "model",
        "evaluation_results.json"
    )
    with open(file_path, "w") as f:
        json.dump(eval_results, f, indent=4)


    print("-----------------------------")
    print("-----Evaluation Results------")
    print("-----------------------------")
    print("----------See Plot-----------")
    print("-----------------------------")

    x = np.linspace(1, len(envs_settings), num=len(envs_settings))
    colours = ["r", "g", "b", "y", "m", "c", "k"]
    for idx, seed in enumerate(seeds):
        mean = [eval_results[seed][epoch]["mean_rwd"] for epoch in eval_results[seed]]
        std = [eval_results[seed][epoch]["std_rwd"] for epoch in eval_results[seed]]
        pos_std = [sum(y) for y in zip(mean, std)]
        neg_std = [ya - yb for ya, yb in zip(mean, std)]
        plt.plot(x, mean, color=colours[idx], label=f"Seed: {seed}")
        plt.fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)
    plt.grid()
    plt.legend()
    plt.ylabel("Average Reward Across Evaluation Episodes")
    if train_id:
        if char_transfer:
            plt.xlabel("Number of Characters")
        else:
            plt.xlabel("Training Episodes")
    else:
        plt.xlabel("Number of Games")
    plt.show()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument("--trainID", type=str, required=False, help="Specific game to train on")
    parser.add_argument('--charTransfer', action=argparse.BooleanOptionalAction, required=True, help="Evaluate character transfer or not")
    opt = parser.parse_args()
    print(opt)

    main(opt.policyCfg, opt.settingsCfg, opt.trainID, opt.charTransfer)
