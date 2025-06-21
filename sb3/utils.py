import random
import torch
import numpy as np
import os
import time

import diambra.arena
from diambra.arena import EnvironmentSettings, EnvironmentSettingsMultiAgent, WrappersSettings, RecordingSettings, SpaceTypes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
import custom_wrappers
from typing import Union

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_sb3_envs(
    game_id: str,
    num_train_envs: int,
    num_eval_envs: int,
    train_characters: list[str] | None,
    eval_characters: list[str] | None,
    multi_agent: bool = False,
    train_env_settings: Union[EnvironmentSettings, EnvironmentSettingsMultiAgent]=EnvironmentSettings(),
    eval_env_settings: Union[EnvironmentSettings, EnvironmentSettingsMultiAgent]=EnvironmentSettings(),
    wrappers_settings: WrappersSettings=WrappersSettings(),
    episode_recording_settings: RecordingSettings=RecordingSettings(),
    render_mode: str="rgb_array", 
    seed: int=None, 
    start_index: int=0,
    allow_early_resets: bool=True, 
    start_method: str=None, 
    no_vec: bool=False,
    use_subprocess: bool=True, 
    log_dir_base: str="/tmp/DIAMBRALog/"
    ):
    def _make_multiagent_env(rank: int, seed: int, env_settings: EnvironmentSettingsMultiAgent, discrete: bool):
        # Seed management
        env_settings.seed = int(time.time()) if seed is None else seed
        env_settings.seed += rank

        def _init():
            env = diambra.arena.make(game_id, env_settings, wrappers_settings,
                                        episode_recording_settings, render_mode, rank=rank)

            # Create log dir
            log_dir = os.path.join(log_dir_base, str(rank))
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir, allow_early_resets=allow_early_resets)
            if discrete:
                env = custom_wrappers.MultiAgentDiscreteTransferWrapper(
                    env=env,
                    stack_frames=wrappers_settings.stack_frames,
                )
            else:
                env = custom_wrappers.MultiAgentMDTransferWrapper(
                    env=env,
                    stack_frames=wrappers_settings.stack_frames,
                )
            return env
        return _init

    def _make_sb3_env(rank: int, seed: int, env_settings: EnvironmentSettings, characters: list[str], discrete: bool):
        # Seed management
        env_settings.seed = int(time.time()) if seed is None else seed
        env_settings.seed += rank

        def _init():
            env = diambra.arena.make(game_id, env_settings, wrappers_settings,
                                        episode_recording_settings, render_mode, rank=rank)

            # Create log dir
            log_dir = os.path.join(log_dir_base, str(rank))
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir, allow_early_resets=allow_early_resets)
            if discrete:
                env = custom_wrappers.DiscreteTransferWrapper(
                    env=env,
                    stack_frames=wrappers_settings.stack_frames,
                    characters=characters,
                )
            else:
                env = custom_wrappers.MDTransferWrapper(
                    env=env,
                    stack_frames=wrappers_settings.stack_frames,
                    characters=characters,
                )
            return env
        return _init

    assert train_env_settings.action_space == eval_env_settings.action_space, "Train env and eval env action spaces not the same!"
    is_discrete = train_env_settings.action_space == SpaceTypes.DISCRETE

    if multi_agent:
        # If not wanting vectorized envs
        if no_vec and num_train_envs == 1 and num_eval_envs == 1:
            train_env = _make_multiagent_env(
                rank=0,
                seed=seed,
                env_settings=train_env_settings,
                discrete=is_discrete,
            )()
            eval_env = _make_multiagent_env(
                rank=1,
                seed=seed,
                env_settings=eval_env_settings,
                discrete=is_discrete,
            )()
        else:
            # When using one environment, no need to start subprocesses
            if (num_train_envs == 1 and num_eval_envs == 1) or not use_subprocess:
                train_env = DummyVecEnv([_make_multiagent_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=train_env_settings,
                    discrete=is_discrete,
                ) for i in range(num_train_envs)])
                start_index = num_train_envs
                eval_env = DummyVecEnv([_make_multiagent_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=eval_env_settings,
                    discrete=is_discrete,
                ) for i in range(num_eval_envs)])
            else:
                train_env = SubprocVecEnv([_make_multiagent_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=train_env_settings,
                    discrete=is_discrete,
                ) for i in range(num_train_envs)], start_method=start_method)
                start_index = num_train_envs
                eval_env = SubprocVecEnv([_make_multiagent_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=eval_env_settings,
                    discrete=is_discrete,
                ) for i in range(num_eval_envs)], start_method=start_method)
    else:
        # If not wanting vectorized envs
        if no_vec and num_train_envs == 1 and num_eval_envs == 1:
            train_env = _make_sb3_env(
                rank=0,
                seed=seed,
                env_settings=train_env_settings,
                characters=train_characters,
                discrete=is_discrete,
            )()
            eval_env = _make_sb3_env(
                rank=1,
                seed=seed,
                env_settings=eval_env_settings,
                characters=eval_characters,
                discrete=is_discrete,
            )()
        else:
            # When using one environment, no need to start subprocesses
            if (num_train_envs == 1 and num_eval_envs == 1) or not use_subprocess:
                train_env = DummyVecEnv([_make_sb3_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=train_env_settings,
                    characters=train_characters,
                    discrete=is_discrete,
                ) for i in range(num_train_envs)])
                start_index = num_train_envs
                eval_env = DummyVecEnv([_make_sb3_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=eval_env_settings,
                    characters=eval_characters,
                    discrete=is_discrete,
                ) for i in range(num_eval_envs)])
            else:
                train_env = SubprocVecEnv([_make_sb3_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=train_env_settings,
                    characters=train_characters,
                    discrete=is_discrete,
                ) for i in range(num_train_envs)], start_method=start_method)
                start_index = num_train_envs
                eval_env = SubprocVecEnv([_make_sb3_env(
                    rank=i + start_index,
                    seed=seed,
                    env_settings=eval_env_settings,
                    characters=eval_characters,
                    discrete=is_discrete,
                ) for i in range(num_eval_envs)], start_method=start_method)

    if not (no_vec and num_train_envs == 1 and num_eval_envs == 1):
        train_env = VecTransposeImage(train_env)
        eval_env = VecTransposeImage(eval_env)

    if seed:
        set_global_seed(seed) # Set global seed

    return train_env, eval_env