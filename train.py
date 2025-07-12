"""
train.py: Train a policy to play a given fighitng game.
"""
import os
import argparse
import configs
import torch as th
import custom_wrappers as cw
import custom_callbacks as cc

from utils import train_eval_split, load_agent
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, CallbackList
from diambra.arena.stable_baselines3.sb3_utils import AutoSave
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
        # Due to DIAMBRA's implementation, selecting the same action each frame corresponds to holding down tthe button for that action.
        # This means that an attack the policy is trying to spam each frame won't come out after the first button press, as holding
        # an attack button leads to nothing happening. To allow the policy to play the game as it intends, this wrapper helps set any attack
        # which is the same as the previous frame's to 0, which stops the env interpreting that attack button as held down and allows the
        # policy to send out another attack. This is particularly important for deterministic policies, as they often converge to relying
        # on one or two attacks no matter the situation, and are particularly prone to this button hold "bug".
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
    # Transpose the env's images so that they have shape (C,H,W) instead of (H,W,C) (stable_baselines3 requires channel first observations)
    train_env, eval_env = VecTransposeImage(train_env), VecTransposeImage(eval_env)
    print(f"\nActivated {num_train_envs + num_eval_envs} environment(s)")
    
    # Load a policy
    model_checkpoint = ppo_config["model_checkpoint"]
    save_path = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}")
    checkpoint_path = os.path.join(save_path, model_checkpoint) if not policy_path else policy_path
    agent = load_agent(env=train_env, seed=settings_config["seed"], policy_path=checkpoint_path)

    # Create callback list: track average number of stages and arcade runs completed, evaluate and autosave at regular intervals
    callbacks_config = configs.callbacks_settings
    auto_save_callback = AutoSave(
        check_freq=callbacks_config["autosave_freq"],
        num_envs=num_train_envs,
        save_path=save_path,
        filename_prefix=model_checkpoint + "_"
    )
    arcade_metrics_callback = cc.ArcadeMetricsTrainCallback(verbose=0)
    stop_training = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback = cc.ArcadeMetricsEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=callbacks_config["n_eval_episodes"],
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
    if configs.wrappers_options["use_teachers"]:
        callback_list.callbacks.append(cc.StudentSimilarityCallback(verbose=0))
    if callbacks_config["measure_action_similarity"]:
        callback_list.callbacks.append(cc.UniqueActionsCallback(verbose=0))

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
        train_id=opt.train_id,                      # ID of game to train on
        num_players=opt.num_players,                # 1 or 2 player env
        policy_path=opt.policy_path,                # Path to a specific policy to use
        episode_num=opt.episode_num,                # Useful for not overriding previous evaluation data if doing finetuning
        eval_deterministic=opt.eval_deterministic,  # Whether to use the deterministi or stochastic policy during evaluation
    )
