import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import custom_wrappers
import custom_callbacks
import utils

# diambra run -s 12 python sb3/train_ppo.py --policyCfg config_files/transfer-cfg-ppo.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --no-charTransfer --numTrainEnvs 8 --numEvalEnvs 4 --trainID _

def main(policy_cfg: str, settings_cfg: str, train_id: str | None, char_transfer: bool, num_train_envs: int, num_eval_envs: int):
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


    # PPO settings
    ppo_settings = policy_params["ppo_settings"]
    policy = ppo_settings["policy_type"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]
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
    time_steps = ppo_settings["time_steps"]
    autosave_freq = ppo_settings["autosave_freq"]
    eval_freq = ppo_settings["eval_freq"]
    seeds = ppo_settings["seeds"]

    # Policy kwargs
    policy_kwargs = policy_params["policy_kwargs"]
    if not policy_kwargs:
        policy_kwargs = {}

    # Load wrappers settings as dictionary
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    stack_frames = settings_params["wrappers_settings"]["stack_frames"]
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

    train_env, eval_env = utils.make_sb3_envs(
        game_id=envs_settings[0].game_id,
        num_train_envs=num_train_envs,
        num_eval_envs=num_eval_envs,
        env_settings=envs_settings[0],
        wrappers_settings=wrappers_settings, 
        seed=0,
        use_subprocess=True,
    )

    train_env.close()
    eval_env.close()

    return 0

    for seed in [0,1,2,3]:
        eval_results.update({seed: {}})
        utils.set_global_seed(seed)
        for epoch in range(len(envs_settings)):
            epoch_settings = envs_settings[epoch]

            # Initialise env and wrap in transfer wrapper
            env, num_envs = utils.make_env(
                game_id=epoch_settings.game_id,
                num_envs=num_train_envs,
                env_settings=epoch_settings, 
                wrappers_settings=wrappers_settings, 
                seed=seed,
                use_subprocess=True,
            )
            if epoch_settings.action_space == SpaceTypes.DISCRETE:
                env = custom_wrappers.VecEnvDiscreteTransferWrapper(env, stack_frames)
            else:
                env = custom_wrappers.VecEnvMDTransferWrapper(env, stack_frames)
            env = VecTransposeImage(env)
            print(f"\nOriginal action space: {env.unwrapped.action_space}")
            print(f"Wrapped action space: {env.action_space}")
            print("\nActivated {} environment(s)".format(env))

            # Load policy params if checkpoint exists, else make a new agent
            checkpoint_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            if int(model_checkpoint) > 0 and os.path.exists(checkpoint_path):
                print("\n Checkpoint found, loading model.")
                agent = PPO.load(
                    checkpoint_path,
                    env=env,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
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
                agent = PPO(
                    policy,
                    env,
                    verbose=1,
                    gamma=gamma,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    n_steps=n_steps,
                    learning_rate=learning_rate,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
                    policy_kwargs=policy_kwargs,
                    gae_lambda=gae_lambda,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    use_sde=use_sde,
                    sde_sample_freq=sde_sample_freq,
                    normalize_advantage=normalize_advantage,
                    stats_window_size=stats_window_size,
                    target_kl=target_kl,
                    tensorboard_log=tensor_board_folder,
                    device=device,
                    seed=seed
                )

            # Print policy network architecture
            print("Policy architecture:")
            print(agent.policy)

            # Create callback list: autosave every few steps, track average number of stages and arcade runs completed
            auto_save_callback = AutoSave(
                check_freq=autosave_freq,
                num_envs=num_envs,
                save_path=os.path.join(model_folder, f"seed_{seed}"),
                filename_prefix=model_checkpoint + "_"
            )
            diambra_eval_callback = custom_callbacks.DiambraEvalCallback(verbose=0)

            # Set up evaluation env for EvalCallback
            eval_env, num_eval_envs = utils.make_env(
                game_id=epoch_settings.game_id,
                num_envs=num_eval_envs,
                env_settings=epoch_settings, 
                wrappers_settings=wrappers_settings,
                seed=seed,
                no_vec=False,
                use_subprocess=True,
            )
            if epoch_settings.action_space == SpaceTypes.DISCRETE:
                eval_env = custom_wrappers.VecEnvDiscreteTransferWrapper(eval_env, stack_frames)
            else:
                eval_env = custom_wrappers.VecEnvMDTransferWrapper(eval_env, stack_frames)
            eval_env = VecTransposeImage(eval_env)

            eval_callback = EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes * num_eval_envs, # Ensure each env completes required num of eval episodes
                eval_freq=eval_freq,
                log_path=None,
                best_model_save_path=None,
                deterministic=False,
                render=False,
                verbose=1,
            )
            callback_list = CallbackList([auto_save_callback, diambra_eval_callback, eval_callback])

            agent.learn(
                total_timesteps=time_steps,
                callback=callback_list,
                reset_num_timesteps=True,
                progress_bar=True,
            )

            # Save the agent
            model_checkpoint = str(int(model_checkpoint) + time_steps)
            model_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            agent.save(model_path)

            del agent

            env.close()
            # eval_env.close()

            # if not train_id or char_transfer:
            #     eval_envs = envs_settings[:epoch + 1]
            # else:
            #     eval_envs = [epoch_settings]

            # mean_rewards = np.zeros(len(eval_envs), dtype=np.float64)
            # mean_stages = np.zeros(len(eval_envs), dtype=np.float64)
            # mean_arcade_runs = np.zeros(len(eval_envs), dtype=np.float64)
            # std_rewards = np.zeros(len(eval_envs), dtype=np.float64)
            # std_stages = np.zeros(len(eval_envs), dtype=np.float64)
            # std_arcade_runs = np.zeros(len(eval_envs), dtype=np.float64)
            # for idx, eval_settings in enumerate(eval_envs):
            #     env, num_envs = make_sb3_env(eval_settings.game_id, eval_settings, wrappers_settings, seed=seed)
            #     if eval_settings.action_space == SpaceTypes.DISCRETE:
            #         env = custom_wrappers.VecEnvDiscreteTransferWrapper(env, stack_frames)
            #     else:
            #         env = custom_wrappers.VecEnvMDTransferWrapper(env, stack_frames)
            #     env = VecTransposeImage(env)
            #     agent = PPO.load(
            #         model_path,
            #         env=env,
            #         policy_kwargs=policy_kwargs,
            #         tensorboard_log=tensor_board_folder,
            #         device=device,
            #         custom_objects={
            #             "action_space" : env.action_space,
            #             "observation_space" : env.observation_space,
            #         }
            #     )

            #     rwd_info, stages_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
            #         model=agent,
            #         env=env,
            #         n_eval_episodes=n_eval_episodes,
            #         deterministic=False,
            #         render=False,
            #         return_episode_rewards=False,
            #     )
            #     env.close()
            #     del agent
                
            #     # Store eval results
            #     mean_rewards[idx], std_rewards[idx] = rwd_info
            #     mean_stages[idx], std_stages[idx] = stages_info
            #     mean_arcade_runs[idx], std_arcade_runs[idx] = arcade_info

            # # Average out results across all eval envs
            # mean_rwd, std_rwd = np.mean(mean_rewards), np.mean(std_rewards)
            # mean_stages, std_stages = np.mean(mean_stages), np.mean(std_stages)
            # mean_arcade_runs, std_arcade_runs = np.mean(mean_arcade_runs), np.mean(std_arcade_runs)

            # # Print and save results for plotting later
            # print("Evaluation Reward: {} (avg) ± {} (std)".format(mean_rwd, std_rwd))
            # print("Evaluation Stages Completed: {} (avg) ± {} (std)".format(mean_stages, std_stages))
            # print("Evaluation Arcade Runs Completed: {} (avg) ± {} (std)".format(mean_arcade_runs, std_arcade_runs))
            # eval_results[seed].update({
            #     epoch: {
            #         "mean_rwd": mean_rwd,
            #         "std_rwd": std_rwd,
            #         "mean_stages": mean_stages,
            #         "std_stages": std_stages,
            #         "mean_arcade_runs": mean_arcade_runs,
            #         "std_arcade_runs": std_arcade_runs,
            #     }
            # })


    # Save results
    # file_path = os.path.join(
    #     base_path,
    #     policy_params["folders"]["parent_dir"],
    #     policy_params["folders"]["model_name"],
    #     "model",
    #     "evaluation_results.json"
    # )
    # with open(file_path, "w") as f:
    #     json.dump(eval_results, f, indent=4)


    # print("-----------------------------")
    # print("-----Evaluation Results------")
    # print("-----------------------------")
    # print("----------See Plots----------")
    # print("-----------------------------")

    # x = np.linspace(1, len(envs_settings), num=len(envs_settings))
    # colours = ["r", "g", "b", "y", "m", "c", "k"]
    # fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False)
    # for idx, seed in enumerate(seeds):
    #     mean_rwd = [eval_results[seed][epoch]["mean_rwd"] for epoch in eval_results[seed]]
    #     std_rwd = [eval_results[seed][epoch]["std_rwd"] for epoch in eval_results[seed]]
    #     pos_std = [sum(y) for y in zip(mean_rwd, std_rwd)]
    #     neg_std = [ya - yb for ya, yb in zip(mean_rwd, std_rwd)]
    #     axs[0].plot(x, mean_rwd, color=colours[idx], label=f"Seed: {seed}")
    #     axs[0].fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)

    #     mean_stages = [eval_results[seed][epoch]["mean_stages"] for epoch in eval_results[seed]]
    #     std_stages = [eval_results[seed][epoch]["std_stages"] for epoch in eval_results[seed]]
    #     pos_std = [sum(y) for y in zip(mean_stages, std_stages)]
    #     neg_std = [ya - yb for ya, yb in zip(mean_stages, std_stages)]
    #     axs[1].plot(x, mean_stages, color=colours[idx], label=f"Seed: {seed}")
    #     axs[1].fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)

    #     mean_arcade_runs = [eval_results[seed][epoch]["mean_arcade_runs"] for epoch in eval_results[seed]]
    #     std_arcade_runs = [eval_results[seed][epoch]["std_arcade_runs"] for epoch in eval_results[seed]]
    #     pos_std = [sum(y) for y in zip(mean_arcade_runs, std_arcade_runs)]
    #     neg_std = [ya - yb for ya, yb in zip(mean_arcade_runs, std_arcade_runs)]
    #     axs[2].plot(x, mean_arcade_runs, color=colours[idx], label=f"Seed: {seed}")
    #     axs[2].fill_between(x, pos_std, neg_std, facecolor=colours[idx], alpha=0.5)

    # axs[0].set_ylabel("Average Reward Across Evaluation Episodes")
    # axs[1].set_ylabel("Average No. of Stages Completed Across Evaluation Episodes")
    # axs[2].set_ylabel("Average No. of Successful Arcade Runs Across Evaluation Episodes")
    # if train_id:
    #     if char_transfer:
    #         x_label = "Number of Characters"
    #     else:
    #         x_label = "Training Episodes"
    # else:
    #     x_label = "Number of Games"

    # for ax in axs:
    #     ax.set_xlabel(x_label)
    #     ax.grid(True)
    #     ax.legend()

    # plt.savefig("eval_results.png")
    # plt.show()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=True, help="Policy config", default="config_files/transfer-cgf-ppo.yaml")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config", default="config_files/transfer-cfg-settings.yaml")
    parser.add_argument("--trainID", type=str, required=False, help="Specific game to train on", default="sfiii3n")
    parser.add_argument('--charTransfer', action=argparse.BooleanOptionalAction, required=True, help="Evaluate character transfer or not", default=False)
    parser.add_argument('--numTrainEnvs', type=int, required=True, help="Number of training environments", default=8)
    parser.add_argument('--numEvalEnvs', type=int, required=True, help="Number of evaluation environments", default=4)
    opt = parser.parse_args()

    main(
        policy_cfg=opt.policyCfg, 
        settings_cfg=opt.settingsCfg, 
        train_id=opt.trainID, 
        char_transfer=opt.charTransfer, 
        num_train_envs=opt.numTrainEnvs,
        num_eval_envs=opt.numEvalEnvs,
    )
