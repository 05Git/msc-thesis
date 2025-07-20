"""
train.py: Train a policy to play a given fighitng game.
"""
import os
import argparse
import custom_wrappers as cw
import custom_callbacks as cc

from utils import train_eval_split, load_agent
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, CallbackList
from stable_baselines3.common.utils import set_random_seed
from diambra.arena.stable_baselines3.sb3_utils import AutoSave
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from diambra.arena import SpaceTypes
from settings import load_settings

# diambra run -s _ python train.py --cfg _ --policy_path _ --deterministic

def main(cfg: str, policy_path: str, deterministic: bool):
    configs: dict = load_settings(cfg)
    if deterministic:
        # Due to DIAMBRA's implementation, selecting the same action each frame corresponds to holding down tthe button for that action.
        # This means that an attack the policy is trying to spam each frame won't come out after the first button press, as holding
        # an attack button leads to nothing happening. To allow the policy to play the game as it intends, this wrapper helps set any attack
        # which is the same as the previous frame's to 0, which stops the env interpreting that attack button as held down and allows the
        # policy to send out another attack. This is particularly important for deterministic policies, as they often converge to relying
        # on one or two attacks no matter the situation, and are particularly prone to this button hold "bug".
        configs["eval_wrappers"].wrappers.append([cw.NoOpWrapper, {
            "action_space_type": "discrete" if configs["eval_settings"].action_space == SpaceTypes.DISCRETE else "multi_discrete",
            "no_attack": 0,
        }])
    
    # train_env, eval_env = train_eval_split(
    #     game_id=configs["train_settings"].game_id,
    #     num_train_envs=configs["misc"]["num_train_envs"],
    #     num_eval_envs=configs["misc"]["num_eval_envs"],
    #     train_settings=configs["train_settings"],
    #     eval_settings=configs["eval_settings"],
    #     train_wrappers=configs["train_wrappers"],
    #     eval_wrappers=configs["eval_wrappers"],
    #     seed=configs["misc"]["seed"]
    # )
    # # Transpose the env's images so that they have shape (C,H,W) instead of (H,W,C) (stable_baselines3 requires channel first observations)
    # train_env, eval_env = VecTransposeImage(train_env), VecTransposeImage(eval_env)

    train_env, num_envs = make_sb3_env(
        game_id=configs["train_settings"].game_id,
        env_settings=configs["train_settings"],
        wrappers_settings=configs["train_wrappers"],
        seed=configs["misc"]["seed"]
    )
    train_env = VecTransposeImage(train_env)
    set_random_seed(configs["misc"]["seed"])
    
    # Load a policy
    model_checkpoint = configs["misc"]["model_checkpoint"]
    save_path = os.path.join(configs["folders"]["model_folder"], f"seed_{configs['misc']['seed']}")
    checkpoint_path = os.path.join(save_path, model_checkpoint) if not policy_path else policy_path
    agent = load_agent(settings_config=configs, env=train_env, policy_path=checkpoint_path)

    # Create callback list
    callbacks_config: dict = configs["callbacks_settings"]
    callback_list = CallbackList([])
    if "autosave" in callbacks_config.keys() and callbacks_config["autosave"]:
        auto_save_callback = AutoSave(
            check_freq=callbacks_config["check_freq"],
            num_envs=num_envs,
            save_path=save_path,
            filename_prefix=model_checkpoint + "_"
        )
        callback_list.callbacks.append(auto_save_callback)
    
    if "arcade_metrics" in callbacks_config.keys() and callbacks_config["arcade_metrics"]:
        callback_list.callbacks.append(cc.ArcadeMetricsTrainCallback(verbose=1))
    
    if "eval" in callbacks_config.keys() and callbacks_config["eval"]:
        if "stop_on_no_improvement" in callbacks_config.keys() and callbacks_config["stop_on_no_improvement"]:
            after_eval = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
        else:
            after_eval = None
        eval_callback = cc.ArcadeMetricsEvalCallback(
            eval_env=train_env,
            n_eval_episodes=callbacks_config["n_eval_episodes"],
            teachers=configs["teachers"],
            eval_freq=max(callbacks_config["eval_freq"] // num_envs, 1),
            log_path=save_path,
            best_model_save_path=save_path,
            deterministic=True,
            render=False,
            callback_after_eval=after_eval,
            verbose=1,
            episode_num=callbacks_config["episode_num"] if "episode_num" in callbacks_config.keys() else None,
        )
        callback_list.callbacks.append(eval_callback)

    if "student_teacher_similarity" in callbacks_config.keys() and callbacks_config["student_teacher_similarity"]:
        callback_list.callbacks.append(cc.StudentSimilarityCallback(verbose=1))

    if "unique_actions" in callbacks_config.keys() and callbacks_config["unique_actions"]:
        callback_list.callbacks.append(cc.UniqueActionsCallback(verbose=1))

    try:
        agent.learn(
            total_timesteps=configs["misc"]["timesteps"],
            callback=callback_list,
            reset_num_timesteps=True,
            progress_bar=True,
            log_interval=1,
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model before exiting.")


    # Save the agent
    model_checkpoint = str(int(model_checkpoint) + configs["misc"]["timesteps"])
    model_path = os.path.join(save_path, model_checkpoint)
    agent.save(model_path)

    train_env.close()
    # eval_env.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Whether to follow a deterministic or stochastic policy", default=True)
    parser.add_argument("--policy_path", type=str, required=False, help="Path to load pre-trained policy", default=None)
    parser.add_argument("--cfg", type=str, required=True, help="Path to settings config")
    opt = parser.parse_args()
    main(cfg=opt.cfg, policy_path=opt.policy_path, deterministic=opt.deterministic)
