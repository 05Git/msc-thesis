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
from diambra.arena.utils.diambra_data_loader import DiambraDataLoader

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement

from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet, CnnRewardNet
from imitation.util.networks import RunningNorm
from imitation.data import rollout
from imitation.util import logger as imit_logger

import custom_wrappers
import custom_callbacks
import utils

# diambra run -s 12 python sb3/train_ppo_imitation.py --policyCfg config_files/transfer-cfg-ppo.yaml --settingsCfg config_files/transfer-cfg-settings.yaml --datasetPath _ --trainID _ --charTransfer _

def get_transitions(data_loader: DiambraDataLoader, agent_num: int | None):
    n_episodes = len(data_loader.episode_files) # Number of datasets
    trajectories = []
    for i in range(n_episodes):
        _ = data_loader.reset() # Load next episode data
        obs = np.array([
            cv2.imdecode( # Decode frames to correct shape
                np.frombuffer(
                    data["obs"]["frame"], dtype=np.uint8 # Read frame data for each step in the episode
                ),
                cv2.IMREAD_UNCHANGED,
            )
            for data in data_loader.episode_data
        ])
        obs = np.stack([obs[i:i+4] for i in range(len(obs) - 3)], axis=0) # Stack frames together, this is what the policy expects as input
        last_frame_stack = np.stack([obs[-1]] * 4)
        obs = np.concatenate([obs, last_frame_stack], axis=0) # Concat 4 extra frame stacks to match trajectory's required shape
        if agent_num is not None:
            acts = np.array([data["action"][f"agent_{agent_num}"] for data in data_loader.episode_data], dtype=np.uint8)
        else:
            acts = np.array([data["action"] for data in data_loader.episode_data], dtype=np.uint8)
        rews = np.array([data["reward"] for data in data_loader.episode_data], dtype=np.float16)
        dones = np.array([data["terminated"] for data in data_loader.episode_data])
        terminal = np.any(dones) # Does this episode end in a terminal state?
        infos = np.array([data["info"] for data in data_loader.episode_data])

        trajectories.append(TrajectoryWithRew(
            obs=obs,
            acts=acts,
            rews=rews,
            terminal=terminal,
            infos=infos
        ))
        print(f"Trajectory {i + 1} loaded")
    
    return rollout.flatten_trajectories_with_rew(trajectories)

def main(
    policy_cfg: str, 
    settings_cfg: str,
    dataset_path_input: str, 
    train_id: str | None, 
    num_train_envs: int, 
    num_eval_envs: int,
    agent_num: int | None,
):
    # Game IDs
    game_ids = [
        "sfiii3n",
        "kof98umh",
        "umk3",
        "samsh5sp"
    ]
    assert train_id in game_ids or not train_id, f"Invalid game id ({train_id}), available ids: [{game_ids}]"

    if dataset_path_input is not None:
        dataset_path = dataset_path_input
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, "dataset")

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

    # Check if we're training with training mode
    is_multi_agent = agent_num is not None

    # Set up imitation transitions
    imitation_data_loader = DiambraDataLoader(dataset_path)
    transitions = get_transitions(imitation_data_loader, agent_num=agent_num)
    print("\nTransitions loaded")

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
    
    # Imitation settings
    imitation_settings = policy_params["imitation_settings"]
    max_imitation_epochs = imitation_settings["max_train_epochs"]
    n_imitation_steps = imitation_settings["n_imitation_steps"]

    # Load wrappers settings as dictionary
    wrappers_settings = settings_params["wrappers_settings"]
    if is_multi_agent:
        wrappers_settings["role_relative"] = False
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
    if is_multi_agent:
        settings["action_space"] = (settings["action_space"], settings["action_space"])
        settings["outfits"] = (settings["outfits"], settings["outfits"])
        game_settings["characters"] = (game_settings["characters"][0], game_settings["characters"][1])
        if train_id == "sfiii3n":
            game_settings["super_art"] = (game_settings["super_art"], game_settings["super_art"])
        if train_id == "kof98umh":
            game_settings["fighting_style"] = (game_settings["fighting_style"], game_settings["fighting_style"])
            game_settings["ultimate_style"] = (game_settings["ultimate_style"], game_settings["ultimate_style"])
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, settings) if is_multi_agent else load_settings_flat_dict(EnvironmentSettings, settings)
    
    eval_results = {}
    for seed in seeds:
        model_checkpoint = ppo_settings["model_checkpoint"]
        save_path = os.path.join(model_folder, f"seed_{seed}")
        eval_results.update({seed: {}})

        # Initialise envs and wrap in transfer wrappers
        train_env, eval_env = utils.make_sb3_envs(
            game_id=settings.game_id,
            num_train_envs=num_train_envs,
            num_eval_envs=num_eval_envs,
            train_characters=train_characters,
            eval_characters=eval_characters,
            multi_agent=is_multi_agent,
            defensive_training=True,
            train_env_settings=settings,
            eval_env_settings=settings, 
            wrappers_settings=wrappers_settings, 
            seed=seed,
            use_subprocess=True,
        )
        print(f"\nOriginal action space: {train_env.unwrapped.action_space}")
        print(f"Wrapped action space: {train_env.action_space}")
        print("\nActivated {} environment(s)".format(num_eval_envs + num_train_envs))

        # Load policy params if checkpoint exists, else make a new agent
        checkpoint_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
        if int(model_checkpoint) > 0 and os.path.isfile(checkpoint_path + ".zip"):
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
        
        # Set new logger
        log_path = os.path.join(model_folder, f"seed_{seed}", "imit_log")
        imit_log = imit_logger.configure(log_path, ["stdout", "csv", "tensorboard"])

        # Set up GAIL trainer
        # reward_net = CnnRewardNet(
        #     observation_space=train_env.observation_space,
        #     action_space=train_env.action_space,
        #     normalize_input_layer=RunningNorm,
        #     hwc_format=False,
        # )
        # gail_trainer = GAIL(
        #     demonstrations=transitions,
        #     demo_batch_size=1028,
        #     gen_replay_buffer_capacity=512,
        #     n_disc_updates_per_round=8,
        #     venv=train_env,
        #     gen_algo=agent,
        #     reward_net=reward_net,
        #     custom_logger=imit_log,
        # )
        
        # Train behavioural clone trainer on transitions
        bc_trainer = bc.BC(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            demonstrations=transitions,
            rng=np.random.default_rng(seed=seed),
            policy=agent.policy,
            device=device,
            custom_logger=imit_log,
        )

        # Evaluate before imitation training
        # reward_info, stage_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
        #     model=agent,
        #     env=train_env,
        #     n_eval_episodes=n_eval_episodes * num_eval_envs,
        #     deterministic=True,
        #     render=False,
        # )
        # mean_reward_before, std_reward_before = reward_info
        # mean_stages_before, std_stages_before = stage_info
        # mean_arcades_before, std_arcades_before = arcade_info

        # Imitation training
        # gail_trainer.train(n_imitation_steps, callback=None)
        # bc_trainer.train(n_epochs=max_imitation_epochs)

        # DAgger training
        # with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        #     print(tmpdir)
        #     dagger_trainer = SimpleDAggerTrainer(
        #         venv=train_env,
        #         scratch_dir=tmpdir,
        #         expert_policy=agent,
        #         bc_trainer=bc_trainer,
        #         rng=np.random.default_rng(seed=seed),
        #         custom_logger=imit_log,
        #     )
        #     dagger_trainer.train(n_imitation_steps)

        # Save imitated policy
        # imitation_folder = os.path.join(model_folder, f"seed_{seed}", "bc_trainer")
        # agent.policy = bc_trainer.policy
        # agent.save(os.path.join(imitation_folder, "bc_agent_policy"))

        # Eval imitation agent
        # reward_info, stage_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
        #     model=agent,
        #     env=train_env,
        #     n_eval_episodes=n_eval_episodes * num_eval_envs,
        #     deterministic=True,
        #     render=False,
        # )
        # mean_reward_after, std_reward_after = reward_info
        # mean_stages_after, std_stages_after = stage_info
        # mean_arcades_after, std_arcades_after = arcade_info

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
            episode_num=None,
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
        model_path = os.path.join(model_folder, f"seed_{seed}", model_checkpoint)
        agent.save(model_path)
        """
        # Evaluate finetuned policy
        reward_info, stage_info, arcade_info = custom_callbacks.evaluate_policy_with_arcade_metrics(
            model=agent,
            env=eval_env,
            n_eval_episodes=n_eval_episodes * num_eval_envs,
            deterministic=True,
            render=False,
        )
        mean_reward_final, std_reward_final = reward_info
        mean_stages_final, std_stages_final = stage_info
        mean_arcades_final, std_arcades_final = arcade_info
        """
        train_env.close()
        eval_env.close()

        eval_results[seed].update({
            # "before_imitation" : {
            #     "eval_reward" : {
            #         "mean" : mean_reward_before,
            #         "std" : std_reward_before,
            #     },
            #     "eval_stages" : {
            #         "mean" : mean_stages_before,
            #         "std" : std_stages_before,
            #     },
            #     "eval_arcade_runs" : {
            #         "mean" : mean_arcades_before,
            #         "std" : std_arcades_before,
            #     },
            # },
            # "after_imitation" : {
            #     "eval_reward" : {
            #         "mean" : mean_reward_after,
            #         "std" : std_reward_after,
            #     },
            #     "eval_stages" : {
            #         "mean" : mean_stages_after,
            #         "std" : std_stages_after,
            #     },
            #     "eval_arcade_runs" : {
            #         "mean" : mean_arcades_after,
            #         "std" : std_arcades_after,
            #     },
            # },
            # "after_training" : {
            #     "eval_reward" : {
            #         "mean" : mean_reward_final,
            #         "std" : std_reward_final,
            #     },
            #     "eval_stages" : {
            #         "mean" : mean_stages_final,
            #         "std" : std_stages_final,
            #     },
            #     "eval_arcade_runs" : {
            #         "mean" : mean_arcades_final,
            #         "std" : std_arcades_final,
            #     },
            # }
        })

    # Save evaluation results
    file_path = os.path.join(
        base_path,
        policy_params["folders"]["parent_dir"],
        policy_params["folders"]["model_name"],
        "model",
        "training_results.json"
    )
    # with open(file_path, "w") as f:
    #     json.dump(eval_results, f, indent=4)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policyCfg", type=str, required=False, help="Policy config", default="config_files/ppo-cfg.yaml")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/settings-cfg.yaml")
    parser.add_argument("--datasetPath", type=str, required=True, help="Path to imitation trajectories")
    parser.add_argument("--trainID", type=str, required=False, help="Specific game to train on", default="sfiii3n")
    parser.add_argument("--numTrainEnvs", type=int, required=False, help="Number of training environments", default=8)
    parser.add_argument("--numEvalEnvs", type=int, required=False, help="Number of evaluation environments", default=4)
    parser.add_argument("--agentNum", type=int, required=False, help="Agent number (if trajectories come from multiagent env)", default=None)
    opt = parser.parse_args()

    main(
        policy_cfg=opt.policyCfg,
        settings_cfg=opt.settingsCfg,
        dataset_path_input=opt.datasetPath,
        train_id=opt.trainID,
        num_train_envs=opt.numTrainEnvs,
        num_eval_envs=opt.numEvalEnvs,
        agent_num=opt.agentNum
    )
