import os
import argparse
import configs
import diambra.arena
import torch as th
import custom_wrappers as cw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from diambra.arena import SpaceTypes

# diambra run -g python watch.py --watch_id _ --num_players _ --policy_path _ --deterministic

def main(
    game_id: str,
    num_players: int,
    policy_path: str,
    deterministic: bool,
):
    assert game_id in configs.game_ids, f"Invalid game id ({game_id}), available ids: [{configs.game_ids}]"

    # Load configs
    settings_config = configs.env_settings
    ppo_config = configs.ppo_settings

    # Load envs
    assert num_players in [1,2]
    if num_players == 1:
        settings, _, wrappers, _ = configs.load_1p_settings(game_id=game_id)
    else:
        settings, _, wrappers, _ = configs.load_2p_settings(game_id=game_id)
    # step_ratio > 1 will make gameplay too fast to observe normally
    settings.step_ratio = 1
    if deterministic:
        wrappers.wrappers.append([cw.NoOpWrapper, {
            "action_space_type": "discrete" if settings.action_space == SpaceTypes.DISCRETE else "multi_discrete",
            "no_attack": 0,
        }])

    env = diambra.arena.make(
        game_id=game_id,
        env_settings=settings,
        wrappers_settings=wrappers,
    )
    # Transpose the env's images so that they have shape (C,H,W) instead of (H,W,C) (stable_baselines3 requires channel first observations)
    env = VecTransposeImage(DummyVecEnv([lambda: env]))
    
    model_checkpoint = ppo_config["model_checkpoint"]
    load_path = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}")
    checkpoint_path = os.path.join(load_path, model_checkpoint) if not policy_path else policy_path
    agent = PPO.load(
        path=checkpoint_path,
        env=env,
        policy_kwargs=configs.policy_kwargs,
        device=configs.ppo_settings["device"],
        custom_objects={
            "action_space" : env.action_space,
            "observation_space" : env.observation_space,
        }
    )
    obs = env.reset()
    while True:
        # dist = agent.policy.get_distribution(th.tensor(obs).float().to(agent.device))
        # move_logits = dist.distribution[0].logits
        # act_logits = dist.distribution[1].logits
        # print("Move Logits:", move_logits)
        # print("Act Logits:", act_logits)
        actions, _ = agent.predict(obs, deterministic=deterministic)
        # print(f"Actions: {actions}")
        obs, rew, done, info = env.step(actions)
        # print(f"Observation: {obs}")
        # print(f"Reward: {rew}")
        # print(f"Info: {info}")
        if done:
            break

    env.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=str, required=False, help="Specific game to train on", default="sfiii3n")
    parser.add_argument("--num_players", type=int, required=False, help="Number of players in the env", default=1)
    parser.add_argument("--policy_path", type=str, required=False, help="Path to load pre-trained policy", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Whether to follow a deterministic or stochastic policy", default=True)
    opt = parser.parse_args()

    main(
        game_id=opt.game_id,
        num_players=opt.num_players,
        policy_path=opt.policy_path,
        deterministic=opt.deterministic,
    )
