import os
import yaml
import json
import argparse
import numpy as np
import cv2
import tempfile

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena import EnvironmentSettings, EnvironmentSettingsMultiAgent, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement

import custom_callbacks
import utils
from custom_wrappers import PixelObsWrapper, ActionWrapper2P, AttTrainWrapper, DefTrainWrapper

def main(
    policy_cfg: str, 
    settings_cfg: str,
    train_id: str, 
    num_train_envs: int,
    num_eval_envs: int,
):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    assert train_id in game_ids, f"Invalid game id ({train_id}), available ids: [{game_ids}]"

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
    n_eval_episodes = ppo_settings["n_eval_episodes"]
    learning_rate = linear_schedule(ppo_settings["train_lr"][0], ppo_settings["train_lr"][1])
    clip_range = linear_schedule(ppo_settings["train_cr"][0], ppo_settings["train_cr"][1])
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
    settings_params["wrappers_settings"]["role_relative"] = False
    custom_wrappers_settings = {"wrappers": [
        [AttTrainWrapper, {}], # Important to place reward shaping wrappers before obs filtering wrappers
        [PixelObsWrapper, {"stack_frames": settings_params["wrappers_settings"]["stack_frames"]}],
        [ActionWrapper2P, {"action_space": settings_params["settings"]["shared"]["action_space"], "opp_type": "random"}],
    ]}
    settings_params["wrappers_settings"].update(custom_wrappers_settings)
    wrappers_settings = load_settings_flat_dict(WrappersSettings, settings_params["wrappers_settings"])
    # Load shared settings
    settings = settings_params["settings"]["shared"]
    # Set action space type
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    # Load game specific settings
    game_settings = settings_params["settings"][train_id]
    train_characters = game_settings["characters"]["train"]
    eval_characters = game_settings["characters"]["eval"]
    game_settings["characters"] = game_settings["characters"]["train"][0]
    settings["action_space"] = (settings["action_space"], settings["action_space"])
    settings["outfits"] = (settings["outfits"], settings["outfits"])
    game_settings["characters"] = (game_settings["characters"][0], game_settings["characters"][1])
    if train_id == "sfiii3n":
        game_settings["super_art"] = (game_settings["super_art"], game_settings["super_art"])
    if train_id == "kof98umh":
        game_settings["fighting_style"] = (game_settings["fighting_style"], game_settings["fighting_style"])
        game_settings["ultimate_style"] = (game_settings["ultimate_style"], game_settings["ultimate_style"])
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, settings)

    eval_results = {}
    for seed in seeds:
        model_checkpoint = ppo_settings["model_checkpoint"]
        save_path = os.path.join(model_folder, f"seed_{seed}")
        eval_results.update({seed: {}})

        # Initialise envs and wrap in transfer wrappers
        train_env, eval_env = utils.train_eval_split(
            game_id=settings.game_id,
            num_train_envs=num_train_envs,
            num_eval_envs=num_eval_envs,
            train_settings=settings,
            train_wrappers=wrappers_settings,
            eval_settings=settings,
            eval_wrappers=wrappers_settings,
            seed=seed
        )
    
        # Load policy params if checkpoint exists, else make a new agent
        checkpoint_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
        if int(model_checkpoint) > 0 and os.path.isfile(checkpoint_path + ".zip"):
            learning_rate = linear_schedule(ppo_settings["finetune_lr"][0], ppo_settings["finetune_lr"][1])
            clip_range = linear_schedule(ppo_settings["finetune_cr"][0], ppo_settings["finetune_cr"][1])
            clip_range_vf = clip_range
            print("\n Checkpoint found, loading model.")
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
            print("\nNo or invalid checkpoint given, creating new model")
            agent = PPO(
                policy,
                train_env,
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

        # Create callback list: track average number of stages and arcade runs completed, evaluate and autosave at regular intervals
        auto_save_callback = AutoSave(
            check_freq=autosave_freq,
            num_envs=num_train_envs,
            save_path=save_path,
            filename_prefix=model_checkpoint + "_"
        )
        arcade_metrics_callback = custom_callbacks.ArcadeMetricsTrainCallback(verbose=0)
        # stop_training = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
        # eval_callback = custom_callbacks.ArcadeMetricsEvalCallback(
        #     eval_env=eval_env,
        #     n_eval_episodes=n_eval_episodes * num_eval_envs, # Ensure each env completes required num of eval episodes
        #     eval_freq=max(eval_freq // num_train_envs, 1),
        #     log_path=save_path,
        #     best_model_save_path=save_path,
        #     deterministic=True,
        #     render=False,
        #     # callback_after_eval=stop_training,
        #     verbose=1,
        #     episode_num=None,
        # )
        callback_list = CallbackList([auto_save_callback, arcade_metrics_callback])
        
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

        train_env.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=False, help="Policy config", default="config_files/ppo-cfg.yaml")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/settings-cfg.yaml")
    parser.add_argument("--trainID", type=str, required=False, help="Specific game to train on", default="sfiii3n")
    parser.add_argument("--numTrainEnvs", type=int, required=False, help="Number of training environments", default=8)
    parser.add_argument("--numEvalEnvs", type=int, required=False, help="Number of evaluation environments", default=4)
    opt = parser.parse_args()

    main(
        policy_cfg=opt.policyCfg,
        settings_cfg=opt.settingsCfg,
        train_id=opt.trainID,
        num_train_envs=opt.numTrainEnvs,
        num_eval_envs=opt.numEvalEnvs,
    )