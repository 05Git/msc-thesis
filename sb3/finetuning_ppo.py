import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement

import custom_wrappers
import custom_callbacks
import utils

def main(policy_cfg: str, settings_cfg: str, source_id: str | None, char_transfer: bool, num_train_envs: int, num_eval_envs: int):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    
    if source_id not in game_ids:
        print(f"Error: Source game ID not found in valid game IDs.\nValid game IDs: {game_ids}\nID provided: {source_id}")
        return 1

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
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]
    n_eval_episodes = ppo_settings["n_eval_episodes"]
    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
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
    if char_transfer:
        game_settings = settings_params["settings"][source_id]
        for character in game_settings["characters"][1:]: # Don't bother with character trained on
            game_settings["characters"] = character
            env_settings = settings.copy()
            env_settings.update(game_settings)
            env_settings = load_settings_flat_dict(EnvironmentSettings, env_settings)
            envs_settings.append(env_settings)
    else:
        for game_id in game_ids:
            if game_id == source_id:
                continue # Don't bother with source game
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

            # Initialise envs and wrap in transfer wrappers
            train_env, eval_env = utils.make_sb3_envs(
                game_id=epoch_settings.game_id,
                num_train_envs=num_train_envs,
                num_eval_envs=num_eval_envs,
                env_settings=epoch_settings, 
                wrappers_settings=wrappers_settings, 
                seed=seed,
                use_subprocess=True,
            )
            if epoch_settings.action_space == SpaceTypes.DISCRETE:
                train_env = custom_wrappers.VecEnvDiscreteTransferWrapper(train_env, stack_frames)
                eval_env = custom_wrappers.VecEnvDiscreteTransferWrapper(eval_env, stack_frames)
            else:
                train_env = custom_wrappers.VecEnvMDTransferWrapper(train_env, stack_frames)
                eval_env = custom_wrappers.VecEnvMDTransferWrapper(eval_env, stack_frames)
            train_env = VecTransposeImage(train_env)
            eval_env = VecTransposeImage(eval_env)

            # Load policy params if checkpoint exists, else make a new agent
            checkpoint_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            if int(model_checkpoint) > 0 and os.path.exists(checkpoint_path):
                print("\nCheckpoint found, loading model.")
                agent = PPO.load(
                    checkpoint_path,
                    env=train_env,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensor_board_folder,
                    device=device,
                    custom_objects={
                        "action_space" : train_env.action_space,
                        "observation_space" : train_env.observation_space,
                    }
                )
            else:
                print(f"\nNo or invalid checkpoint given, please provide a valid checkpoint.\nCheckpoint provided: {model_checkpoint}")
                return 1
            
            # Print policy network architecture
            print("Policy architecture:")
            print(agent.policy)

            # Create callback list: track average number of stages and arcade runs completed, evaluate and autosave at regular intervals
            auto_save_callback = AutoSave(
                check_freq=autosave_freq,
                num_envs=num_train_envs,
                save_path=os.path.join(model_folder, f"seed_{seed}"),
                filename_prefix=model_checkpoint + "_"
            )
            diambra_eval_callback = custom_callbacks.DiambraEvalCallback(verbose=0)
            stop_training = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
            eval_callback = EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes * num_eval_envs, # Ensure each env completes required num of eval episodes
                eval_freq=eval_freq // num_train_envs,
                log_path=os.path.join(model_folder, f"seed_{seed}", epoch_settings.game_id, epoch_settings.characters),
                best_model_save_path=os.path.join(model_folder, f"seed_{seed}", epoch_settings.game_id, epoch_settings.characters),
                deterministic=True,
                render=False,
                callback_after_eval=stop_training,
                verbose=1,
            )
            callback_list = CallbackList([auto_save_callback, diambra_eval_callback, eval_callback])

            # Evaluate before training
            reward_info, stage_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
                model=agent,
                env=eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
            mean_reward_before, std_reward_before = reward_info
            mean_stages_before, std_stages_before = stage_info
            mean_arcades_before, std_arcades_before = arcade_info

            try:
                agent.learn(
                    total_timesteps=time_steps,
                    callback=callback_list,
                    reset_num_timesteps=True,
                    progress_bar=True,
                )
            except KeyboardInterrupt:
                print("Training interrupted. Saving model before exiting.")

            # Save the agent
            model_checkpoint = str(int(model_checkpoint) + time_steps)
            model_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
            agent.save(model_path)

            # Evaluate after training
            reward_info, stage_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
                model=agent,
                env=eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
            mean_reward_after, std_reward_after = reward_info
            mean_stages_after, std_stages_after = stage_info
            mean_arcades_after, std_arcades_after = arcade_info

            eval_results[seed].update({
                f"Game: {epoch_settings.game_id}, Character: {epoch_settings.characters}" : {
                    "before_fine_tuning" : {
                        "eval_reward" : {
                            "mean" : mean_reward_before,
                            "std" : std_reward_before,
                        },
                        "eval_stages" : {
                            "mean" : mean_stages_before,
                            "std" : std_stages_before,
                        },
                        "eval_arcade_runs" : {
                            "mean" : mean_arcades_before,
                            "std" : std_arcades_before,
                        },
                    },
                    "after_fine_tuning" : {
                        "eval_reward" : {
                            "mean" : mean_reward_after,
                            "std" : std_reward_after,
                        },
                        "eval_stages" : {
                            "mean" : mean_stages_after,
                            "std" : std_stages_after,
                        },
                        "eval_arcade_runs" : {
                            "mean" : mean_arcades_after,
                            "std" : std_arcades_after,
                        },
                    },
                },
            })

            train_env.close()
            eval_env.close()

    # Training finished
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=False, help="Policy config", default="config_files/transfer-cfg-ppo.yaml")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/transfer-cfg-settings.yaml")
    parser.add_argument("--sourceID", type=str, required=False, help="Source game model was trained on", default="sfiii3n")
    parser.add_argument('--charTransfer', action=argparse.BooleanOptionalAction, required=False, help="Evaluate character transfer or not", default=False)
    parser.add_argument('--numTrainEnvs', type=int, required=False, help="Number of training environments", default=8)
    parser.add_argument('--numEvalEnvs', type=int, required=False, help="Number of evaluation environments", default=4)
    opt = parser.parse_args()

    main(
        policy_cfg=opt.policyCfg, 
        settings_cfg=opt.settingsCfg, 
        source_id=opt.sourceID, 
        char_transfer=opt.charTransfer, 
        num_train_envs=opt.numTrainEnvs,
        num_eval_envs=opt.numEvalEnvs,
    )
