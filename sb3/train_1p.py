import os
import yaml
import argparse
import torch
import copy

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecTransposeImage

import custom_callbacks
import utils
from custom_wrappers import PixelObsWrapper, ActionWrapper1P, InterleavingWrapper

# diambra run -s _ python sb3/train.py --policy_cfg config_files/ppo-cfg.yaml --settings_cfg config_files/settings-cfg.yaml --train_id _ --train_type _ --num_train_envs _ --num_eval_envs _

def main(
    policy_cfg: str,
    settings_cfg: str, 
    train_id: str | None, 
    train_type: str,
    num_train_envs: int,
    num_eval_envs: int,
):    
    # Game IDs
    game_ids = [
        "sfiii3n",
        "samsh5sp",
        "kof98umh",
        "umk3",
    ]
    assert train_id in game_ids or not train_id, f"Invalid game id ({train_id}), available ids: [{game_ids}]"

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
    custom_wrappers_settings = {"wrappers": [
        [PixelObsWrapper, {"stack_frames": settings_params["wrappers_settings"]["stack_frames"]}],
        [ActionWrapper1P, {"action_space": settings_params["settings"]["shared"]["action_space"]}],
    ]}
    settings_params["wrappers_settings"].update(custom_wrappers_settings)
    # Load shared settings
    settings = settings_params["settings"]["shared"]
    # Set action space type
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE

    envs_settings = []
    env_settings = settings.copy()
    # Load game specific settings
    if train_id:
        game_settings = settings_params["settings"][train_id]
        if train_type == "sequential":
            characters_list = game_settings["characters"].copy()
            for i in range(len(characters_list["train"])):
                game_settings["characters"]["train"] = [characters_list["train"][i]]
                game_settings["characters"]["eval"] = characters_list["eval"]
                env_settings.update(game_settings)
                envs_settings.append(copy.deepcopy(env_settings))
        else:
            env_settings.update(game_settings)
            envs_settings.append(copy.deepcopy(env_settings))
    else:
        for game_id in game_ids:
            game_settings = settings_params["settings"][game_id]
            if train_type == "sequential":
                characters_list = game_settings["characters"].copy()
                for i in range(len(characters_list["train"])):
                    game_settings["characters"]["train"] = [characters_list["train"][i]]
                    game_settings["characters"]["eval"] = characters_list["eval"]
                    env_settings.update(game_settings)
                    envs_settings.append(copy.deepcopy(env_settings))
            else:
                env_settings.update(game_settings)
                envs_settings.append(copy.deepcopy(env_settings))

    # For saving evals to npx file without overwriitng old data
    more_than_one_episode = len(envs_settings) > 1

    for seed in seeds:
        save_path = os.path.join(model_folder, f"seed_{seed}")
        model_checkpoint = ppo_settings["model_checkpoint"]
        
        for epoch in range(len(envs_settings)):
            checkpoint_path = os.path.join(save_path, model_checkpoint)

            # Set up separate settings for train and evaluation envs
            train_settings = envs_settings[epoch].copy()
            train_settings["characters"] = train_settings["characters"]["train"]
            eval_settings = envs_settings[epoch].copy()
            eval_settings["characters"] = eval_settings["characters"]["eval"]

            train_characters = train_settings["characters"].copy() # Make a list of characters to train on
            eval_characters = eval_settings["characters"].copy()

            train_settings["characters"] = train_settings["characters"][0] # Set initial character to the first one one the list, so that the dict can be loaded
            train_settings = load_settings_flat_dict(EnvironmentSettings, train_settings)
            eval_settings["characters"] = eval_settings["characters"][0]
            eval_settings = load_settings_flat_dict(EnvironmentSettings, eval_settings)

            train_wrappers = settings_params["wrappers_settings"].copy()
            if len(train_characters) > 1:
                train_wrappers["wrappers"].append([InterleavingWrapper, {"character_list": train_characters}])
            train_wrappers = load_settings_flat_dict(WrappersSettings, train_wrappers)

            eval_wrappers = settings_params["wrappers_settings"].copy()
            if len(eval_characters) > 1:
                eval_wrappers["wrappers"].append([InterleavingWrapper, {"character_list": eval_characters}])
            eval_wrappers = load_settings_flat_dict(WrappersSettings, eval_wrappers)

            # Initialise envs and wrap in transfer wrappers
            train_env, eval_env = utils.train_eval_split(
                game_id=train_settings.game_id,
                num_train_envs=num_train_envs,
                num_eval_envs=num_eval_envs,
                train_settings=train_settings,
                eval_settings=eval_settings,
                train_wrappers=train_wrappers,
                eval_wrappers=eval_wrappers,
                seed=seed
            )
            print(f"\nOriginal action space: {train_env.unwrapped.action_space}")
            print(f"Wrapped action space: {train_env.action_space}")
            print(f"\nActivated {num_train_envs + num_eval_envs} environment(s)")

            # Load policy params if checkpoint exists, else make a new agent
            if int(model_checkpoint) > 0 and os.path.isfile(checkpoint_path + ".zip"):
                # Finetune settings
                learning_rate = linear_schedule(ppo_settings["finetune_lr"][0], ppo_settings["finetune_lr"][1])
                clip_range = linear_schedule(ppo_settings["finetune_cr"][0], ppo_settings["finetune_cr"][1])
                clip_range_vf = clip_range

                print("\nCheckpoint found, loading model.")
                agent = PPO.load(
                    path=checkpoint_path,
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
                    policy=policy,
                    env=train_env,
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
            stop_training = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
            eval_callback = custom_callbacks.ArcadeMetricsEvalCallback(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes * num_eval_envs, # Ensure each env completes required num of eval episodes
                eval_freq=max(eval_freq // num_train_envs, 1),
                log_path=save_path,
                best_model_save_path=save_path,
                deterministic=True,
                render=False,
                # callback_after_eval=stop_training,
                verbose=1,
                episode_num=epoch if more_than_one_episode else None,
            )
            callback_list = CallbackList([auto_save_callback, arcade_metrics_callback, eval_callback])

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
            model_path = os.path.join(save_path, model_checkpoint)
            agent.save(model_path)

            train_env.close()
            eval_env.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_cfg", type=str, required=False, help="Policy config", default="config_files/ppo-cfg.yaml")
    parser.add_argument("--settings_cfg", type=str, required=False, help="Env settings config", default="config_files/settings-cfg.yaml")
    parser.add_argument("--train_id", type=str, required=False, help="Specific game to train on", default="sfiii3n")
    parser.add_argument("--train_type", type=str, required=False, help="Sequential or interleaved training", default="interleaved")
    parser.add_argument("--num_train_envs", type=int, required=False, help="Number of train envs", default=8)
    parser.add_argument("--num_eval_envs", type=int, required=False, help="Number of evaluation envs", default=4)
    opt = parser.parse_args()

    main(
        policy_cfg=opt.policy_cfg, 
        settings_cfg=opt.settings_cfg, 
        train_id=opt.train_id, 
        train_type=opt.train_type,
        num_train_envs=opt.num_train_envs,
        num_eval_envs=opt.num_eval_envs,
    )
