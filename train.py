import os
import argparse
import configs
import torch as th
import custom_wrappers as cw

from utils import train_eval_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, CallbackList
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from custom_callbacks import ArcadeMetricsTrainCallback, ArcadeMetricsEvalCallback, StudentSimilarityCallback
from diambra.arena import SpaceTypes

# diambra run -s _ python train.py --train_id _ --num_players _ --policy_path _ --episode_num _ --eval_deterministic

def main(
    train_id: str,
    num_players: int,
    policy_path: str,
    episode_num: int,
    eval_deterministic: bool,
):
    assert train_id in configs.game_ids, f"Invalid game id ({train_id}), available ids: [{configs.game_ids}]"

    # Load configs
    settings_config = configs.env_settings
    ppo_config = configs.ppo_settings

    # Load envs
    assert num_players in [1,2]
    if num_players == 1:
        train_settings, eval_settings, train_wrappers, eval_wrappers = configs.load_1p_settings(game_id=train_id)
    else:
        train_settings, eval_settings, train_wrappers, eval_wrappers = configs.load_2p_settings(game_id=train_id)
    if eval_deterministic:
        eval_wrappers.wrappers.append([cw.NoOpWrapper, {
            "action_space_type": "discrete" if eval_settings.action_space == SpaceTypes.DISCRETE else "multi_discrete",
            "no_attack": 0,
        }])
    
    num_train_envs = settings_config["num_train_envs"] 
    num_eval_envs = settings_config["num_eval_envs"]
    train_env, eval_env = train_eval_split(
        game_id=train_id,
        num_train_envs=num_train_envs,
        num_eval_envs=num_eval_envs,
        train_settings=train_settings,
        eval_settings=eval_settings,
        train_wrappers=train_wrappers,
        eval_wrappers=eval_wrappers,
        seed=settings_config["seed"]
    )
    train_env, eval_env = VecTransposeImage(train_env), VecTransposeImage(eval_env)
    print(f"\nActivated {num_train_envs + num_eval_envs} environment(s)")
    
    model_checkpoint = ppo_config["model_checkpoint"]
    save_path = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}")
    checkpoint_path = os.path.join(save_path, model_checkpoint) if not policy_path else policy_path
    # Load policy params if checkpoint exists, else make a new agent
    if os.path.isfile(checkpoint_path + ".zip"):
        print("\nCheckpoint found, loading policy")
        agent = PPO.load(
            path=checkpoint_path,
            env=train_env,
            gamma=ppo_config["gamma"],
            learning_rate=linear_schedule(ppo_config["finetune_lr"][0], ppo_config["finetune_lr"][1]),
            clip_range=linear_schedule(ppo_config["finetune_cr"][0], ppo_config["finetune_cr"][1]),
            clip_range_vf=linear_schedule(ppo_config["finetune_cr"][0], ppo_config["finetune_cr"][1]),
            policy_kwargs=configs.policy_kwargs,
            tensorboard_log=configs.tensor_board_folder,
            device=configs.ppo_settings["device"],
            custom_objects={
                "action_space" : train_env.action_space,
                "observation_space" : train_env.observation_space,
            }
        )
    else:
        print("\nNo or invalid checkpoint given, creating new policy")
        agent = PPO(
            policy=ppo_config["policy"],
            env=train_env,
            verbose=1,
            gamma=ppo_config["gamma"],
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            n_steps=ppo_config["n_steps"],
            learning_rate=linear_schedule(ppo_config["train_lr"][0], ppo_config["train_lr"][1]),
            clip_range=linear_schedule(ppo_config["train_cr"][0], ppo_config["train_cr"][1]),
            clip_range_vf=linear_schedule(ppo_config["train_cr"][0], ppo_config["train_cr"][1]),
            policy_kwargs=configs.policy_kwargs,
            gae_lambda=ppo_config["gae_lambda"],
            ent_coef=ppo_config["ent_coef"],
            vf_coef=ppo_config["vf_coef"],
            max_grad_norm=ppo_config["max_grad_norm"],
            use_sde=ppo_config["use_sde"],
            sde_sample_freq=ppo_config["sde_sample_freq"],
            normalize_advantage=ppo_config["normalize_advantage"],
            stats_window_size=ppo_config["stats_window_size"],
            target_kl=ppo_config["target_kl"],
            tensorboard_log=configs.tensor_board_folder,
            device=configs.ppo_settings["device"],
            seed=settings_config["seed"]
        )
    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create callback list: track average number of stages and arcade runs completed, evaluate and autosave at regular intervals
    callbacks_config = configs.callbacks_settings
    auto_save_callback = AutoSave(
        check_freq=callbacks_config["autosave_freq"],
        num_envs=num_train_envs,
        save_path=save_path,
        filename_prefix=model_checkpoint + "_"
    )
    arcade_metrics_callback = ArcadeMetricsTrainCallback(verbose=0)
    stop_training = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback = ArcadeMetricsEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=callbacks_config["n_eval_episodes"], # Ensure each env completes required num of eval episodes
        eval_freq=max(callbacks_config["eval_freq"] // num_train_envs, 1),
        log_path=save_path,
        best_model_save_path=save_path,
        deterministic=True,
        render=False,
        callback_after_eval=stop_training if callbacks_config["stop_training_if_no_improvement"] else None,
        verbose=1,
        episode_num=episode_num,
    )
    callback_list = CallbackList([auto_save_callback, arcade_metrics_callback])
    if callbacks_config["evaluate_during_training"]:
        callback_list.callbacks.append(eval_callback)
    if configs.teacher_paths:
        callback_list.callbacks.append(StudentSimilarityCallback(verbose=0))

    try:
        agent.learn(
            total_timesteps=ppo_config["time_steps"],
            callback=callback_list,
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model before exiting.")

    # Save the agent
    model_checkpoint = str(int(model_checkpoint) + ppo_config["time_steps"])
    model_path = os.path.join(save_path, model_checkpoint)
    agent.save(model_path)

    train_env.close()
    eval_env.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_id", type=str, required=False, help="Specific game to train on", default="sfiii3n")
    parser.add_argument("--num_players", type=int, required=False, help="Number of players in the env", default=1)
    parser.add_argument("--policy_path", type=str, required=False, help="Path to load pre-trained policy", default=None)
    parser.add_argument("--episode_num", type=int, required=False, help="Number of players in the env", default=0)
    parser.add_argument("--eval_deterministic", action=argparse.BooleanOptionalAction, required=False, help="Evaluate deterministic or stochastic policy", default=True)
    opt = parser.parse_args()

    main(
        train_id=opt.train_id,
        num_players=opt.num_players,
        policy_path=opt.policy_path,
        episode_num=opt.episode_num,
        eval_deterministic=opt.eval_deterministic,
    )
