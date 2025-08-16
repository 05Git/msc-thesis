"""
utils.py: Various useful functions for loading policies and evaluating them.
"""
import os
import time
import warnings
import numpy as np
import gymnasium as gym
import torch as th
import diambra.arena

from stable_baselines3 import PPO
from RND import RNDPPO
from sb3_distill import StudentDistill, TeacherDistill, ProximalPolicyDistillation
from diambra.arena import EnvironmentSettings, WrappersSettings, RecordingSettings
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common import type_aliases, distributions
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from typing import Any, Callable, Optional, Union
from FusionNet import MultiExpertFusionPolicy, MultiExpertFusionNet
from collections import OrderedDict


def load_agent(settings_config: dict, env: gym.Env, policy_path: str, force_load: bool = False):
    """
    Load a PPO, RNDPPO, or MultiExpertFusionNet agent.

    :param settings_config: (dict) Dictionary containing necessary policy settings.
    :param env: (gym.Env) Environment for the agent to interact with.
    :param policy_path: (str) Path to load the policy.
    :param force_load: (bool) If False and the given checkpoint is invalid, a new policy will be created.
    Otherwise, this will return an error.
    """
    if policy_path[-4:] != ".zip":
        policy_path = policy_path + ".zip"

    policy_settings: dict = settings_config["policy_settings"]
    agent_type: PPO | RNDPPO | MultiExpertFusionNet = settings_config["agent_type"]

    if os.path.isfile(policy_path):
        print("\nCheckpoint found, loading policy.")
        if "fusion_settings" in settings_config.keys():
            if "custom_objects" in policy_settings.keys():
                custom_objects = policy_settings.pop("custom_objects")
            else:
                custom_objects = dict()
            agent = agent_type(env=env, **policy_settings)
            assert isinstance(agent.policy, MultiExpertFusionPolicy)

            fusion_settings: dict = settings_config["fusion_settings"]
            if "fixed_weights" in fusion_settings["expert_params"].keys() \
                and fusion_settings["expert_params"]["fixed_weights"] == "uniform":
                n_experts = len(fusion_settings["experts"])
                fusion_settings["expert_params"]["fixed_weights"] = [
                    1 / n_experts for _ in range(n_experts)
                ]
            agent.policy.set_experts(fusion_settings["experts"])
            agent.policy.set_expert_params(**fusion_settings["expert_params"])

            if hasattr(agent.policy, "weights_net"):
                # stable_baselines3's load() function builds a new model and sets the params for it after some preprocessing.
                # Since weights_net (for adaptive weights) is set after the model is loaded, this causes errors when loading
                # params from the state_dict, since the newly created model has no weights_net to load the saved weights into.
                # As such, for a fusion policy with adaptive weights, we skip the preprocessing steps and go straight to loading
                # the params directly. This works right now since the models are saved in line with what stable_baselines3 expects
                # out of its preprocessing stage, however this may cause issues if this is not the case.
                _, params, _ = load_from_zip_file(
                    policy_path,
                    device=policy_settings["device"],
                    custom_objects=custom_objects,
                    print_system_info=True,
                )
                try:
                    agent.set_parameters(params, exact_match=True, device=policy_settings["device"])
                except RuntimeError as e:
                    raise e
            else:
                agent.load(policy_path, custom_objects=custom_objects)
        else:
            agent = agent_type.load(path=policy_path, env=env, **policy_settings)
    elif not force_load:
        if "custom_objects" in policy_settings.keys():
            custom_objects = policy_settings.pop("custom_objects")
            del custom_objects
        agent = agent_type(env=env, **policy_settings)
        if "fusion_settings" in settings_config.keys():
            fusion_settings: dict = settings_config["fusion_settings"]
            agent.policy.set_experts(fusion_settings["experts"])
            agent.policy.set_expert_params(**fusion_settings["expert_params"])
    else:
        raise Exception("\nInvalid checkpoint, please check policy path provided.")

    if "distil_settings" in settings_config.keys():
        distil_settings: dict = settings_config["distil_settings"]
        # Get the type of distillation method
        distil_type: str = distil_settings["distil_type"]
        if distil_type == "student":
            student_type = StudentDistill
        elif distil_type == "teacher":
            student_type = TeacherDistill
        elif distil_type == "ppd":
            student_type = ProximalPolicyDistillation
        else:
            raise ValueError(f"""Invalid student type ({distil_type}). Please select from:
                             'student', 'teacher', 'ppd'.""")
        student_policy_settings: dict = distil_settings["policy_settings"]

        # Load policy path if specified, else create new student policy
        if "policy_path" in distil_settings.keys():
            student_policy_path = distil_settings["policy_path"]
            if student_policy_path[-4:] != ".zip":
                student_policy_path = student_policy_path + ".zip"
            if os.path.isfile(student_policy_path):
                student = student_type.load(path=student_policy_path, env=env, **student_policy_settings)
            else:
                raise Exception("\nInvalid student policy checkpoint, please check path provided.")
        elif not force_load:
            student = student_type(env=env, **student_policy_settings)
        else:
            raise Exception("\nInvalid student policy checkpoint, please check path provided.")
        
        # Check for specific PPD settings
        # TODO: Test if this check is necessary
        for param in agent.policy.parameters():
            param.requires_grad = False # Don't think this is absolutely necessary, but leaving it here just in case
        if student_type == ProximalPolicyDistillation:
            if "ppd_settings" not in distil_settings.keys():
                distil_settings["ppd_settings"] = dict()
            student.set_teacher(agent, **distil_settings["ppd_settings"])
        else:
            student.set_teacher(agent)

        # Print policy network architecture
        print("Student architecture:")
        print(student.policy)
        print("Teacher architecture:")
        print(student.teacher_model.policy)

        return student

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)
    
    return agent


def train_eval_split(game_id: str,
                     ####################### MODIFIED #########################
                     num_train_envs: int, num_eval_envs: int,
                     train_settings: EnvironmentSettings=EnvironmentSettings(),
                     train_wrappers: WrappersSettings=WrappersSettings(),
                     eval_settings: EnvironmentSettings=EnvironmentSettings(),
                     eval_wrappers: WrappersSettings=WrappersSettings(),
                     ##########################################################
                     episode_recording_settings: RecordingSettings=RecordingSettings(),
                     render_mode: str="rgb_array", seed: int=None, start_index: int=0,
                     allow_early_resets: bool=True, start_method: str=None, no_vec: bool=False,
                     use_subprocess: bool=True, log_dir_base: str="/tmp/DIAMBRALog/"):
    """
    Modified version of make_sb3_env to simplify making separate train and eval envs. All lines of code which have been changed
    or added have been marked with ## MODIFIED ##.
    Original code available at: https://github.com/diambra/arena/blob/main/diambra/arena/stable_baselines3/make_sb3_env.py#L11

    :param game_id: (str) the game environment ID
    :param num_train_envs: (int) Number of training environments
    :param num_eval_envs: (int) Number of evaluation environments
    :param train_env_settings: (EnvironmentSettings) parameters for train env
    :param train_wrappers: (WrappersSettings) parameters train env
    :param eval_env_settings: (EnvironmentSettings) parameters for eval env
    :param eval_wrappers: (WrappersSettings) parameters eval env
    :param episode_recording_settings: (RecordingSettings) parameters for environment recording wrapping function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses. See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv`
    :param no_vec: (bool) Whether to avoid usage of Vectorized Env or not. Default: False
    :return: (VecEnv) The diambra environment
    """
    def _make_sb3_env(rank, seed, env_settings, wrappers_settings):
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
            return env
        return _init
    
    ############################################################## MODIFIED #################################################################
    assert eval_settings.game_id == train_settings.game_id
    assert num_train_envs > 0 and num_eval_envs > 0
    
    # If not wanting vectorized envs
    if no_vec and (num_train_envs == 1 and num_eval_envs == 1):
        train_env = _make_sb3_env(0, seed, train_settings, train_wrappers)()
        eval_env = _make_sb3_env(1, seed, eval_settings, eval_wrappers)()
    else:
        # When using one environment, no need to start subprocesses
        if (num_train_envs == 1 and num_eval_envs == 1) or not use_subprocess:
            train_env = DummyVecEnv([_make_sb3_env(i + start_index, seed, train_settings, train_wrappers) for i in range(num_train_envs)])
            start_index += num_train_envs
            eval_env = DummyVecEnv([_make_sb3_env(i + start_index, seed, eval_settings, eval_wrappers) for i in range(num_eval_envs)])
        else:
            train_env = SubprocVecEnv([_make_sb3_env(i + start_index, seed, train_settings, train_wrappers) for i in range(num_train_envs)],
                                      start_method=start_method)
            start_index += num_train_envs
            eval_env = SubprocVecEnv([_make_sb3_env(i + start_index, seed, eval_settings, eval_wrappers) for i in range(num_eval_envs)],
                                      start_method=start_method)

    if seed is not None:
        set_random_seed(seed)
        
    return train_env, eval_env
    #########################################################################################################################################


def evaluate_policy_with_arcade_metrics(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    ################# MODIFIED ###################
    teachers: dict[str: OnPolicyAlgorithm] = None,
    ##############################################
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
):
    """
    Extended version of stable_baseline3's evaluate_poliy method to track custom metrics. All lines of code which have been changed
    or added have been marked with ## MODIFIED ##.
    Original code available at: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    ################################# MODIFIED ##################################
    stages_completed = []       # Number of arcade stages completed in each env
    arcade_runs_completed = []  # Number of full arcade runs completed across envs
    current_stages_completed = np.zeros(n_envs, dtype=np.float64)
    current_arcade_runs_completed = np.zeros(n_envs, dtype=np.float64)
    student_teacher_divergences = {}
    teacher_act_counts = dict()
    total_act_counts = np.zeros((env.action_space.shape[0], 1))
    if teachers is not None:
        for t_id in teachers.keys():
            student_teacher_divergences.update({t_id: []})
            teacher_act_counts.update({t_id: np.zeros((env.action_space.shape[0], 1))})
    #############################################################################

    with th.no_grad():
        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            new_observations, rewards, dones, infos = env.step(actions)
            current_rewards += rewards
            current_lengths += 1

            ############################################ MODIFIED ##########################################
            if teachers is not None:
                tensor_obs = obs_as_tensor(observations, model.device)
                student_act_distribution = model.policy.get_distribution(tensor_obs)
                if type(tensor_obs) == dict:
                    tensor_obs = tensor_obs["image"]
                for id, teacher in teachers.items():
                    teacher_act_distribution = teacher.policy.get_distribution(tensor_obs)
                    kl_div = distributions.kl_divergence(teacher_act_distribution, student_act_distribution)
                    if isinstance(
                        teacher_act_distribution,
                        (distributions.DiagGaussianDistribution,
                            distributions.StateDependentNoiseDistribution)
                    ):
                        kl_div = distributions.sum_independent_dims(kl_div)
                    kl_div = th.mean(kl_div).cpu().detach().numpy().tolist()
                    student_teacher_divergences[id].append(kl_div)
                
                teacher_obs = observations
                if type(teacher_obs) == OrderedDict:
                    teacher_obs = teacher_obs["image"]
                t_acts = {t_id: t_net.predict(teacher_obs, deterministic=deterministic)[0]
                        for t_id, t_net in teachers.items()}                
            ################################################################################################

            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    if callback is not None:
                        callback(locals(), globals())
                    
                    ############################### MODIFIED ##########################
                    if teachers is not None:
                        for t_id in teachers.keys():
                            for a_idx in range(actions[i].shape[0]):
                                t_act = t_acts[t_id][i, a_idx]
                                s_act = actions[i, a_idx]
                                if t_act == s_act:
                                    teacher_act_counts[t_id][a_idx, 0] += 1
                        total_act_counts[:, 0] += 1
                    
                    if info["stage_done"]:
                        current_stages_completed[i] += 1
                        # Increment evey time the final stage is successfully completed
                        if done:
                            current_arcade_runs_completed[i] += 1
                    ###################################################################

                    if dones[i]:
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1
                        ######################## MODIFIED ############################
                        stages_completed.append(current_stages_completed[i])
                        arcade_runs_completed.append(current_arcade_runs_completed[i])
                        current_stages_completed[i] = 0
                        current_arcade_runs_completed[i] = 0
                        ##############################################################
                        current_rewards[i] = 0
                        current_lengths[i] = 0

            observations = new_observations

            if render:
                env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    ####################################### MODIFIED ########################################
    mean_stages = np.mean(stages_completed)
    std_stages = np.std(stages_completed)
    mean_arcade_runs = np.mean(arcade_runs_completed)
    std_arcade_runs = np.std(arcade_runs_completed)
    avg_kl_divs = {id: np.mean(kl_div) for id, kl_div in student_teacher_divergences.items()}
    std_kl_divs = {id: np.std(kl_div) for id, kl_div in student_teacher_divergences.items()}

    # Bernoulli mean and std
    teacher_act_means = dict()
    teacher_act_stds = dict()
    if teachers is not None:
        teacher_act_means.update({
            teacher_id: teacher_act_counts[teacher_id] / total_act_counts
            for teacher_id in teachers.keys()
        })
        teacher_act_stds.update({
            teacher_id: np.sqrt(
                (teacher_act_means[teacher_id] * (1 - teacher_act_means[teacher_id])) / total_act_counts
            )
            for teacher_id in teachers.keys()
        })
        
        # Preprocess to lists for saving to json
        for teacher_id in teachers.keys():
            teacher_act_counts[teacher_id] = teacher_act_counts[teacher_id].tolist()
            teacher_act_means[teacher_id] = teacher_act_means[teacher_id].tolist()
            teacher_act_stds[teacher_id] = teacher_act_stds[teacher_id].tolist()
    #########################################################################################

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    
    if return_episode_rewards:
    #################################################### MODIFIED ###############################################################
        return episode_rewards, episode_lengths, stages_completed, arcade_runs_completed, \
            student_teacher_divergences, teacher_act_counts, teacher_act_means, teacher_act_stds
    
    return (mean_reward, std_reward), (mean_stages, std_stages), (mean_arcade_runs, std_arcade_runs), \
        (avg_kl_divs, std_kl_divs), (teacher_act_counts, teacher_act_means, teacher_act_stds)
    #############################################################################################################################


def eval_student_teacher_likelihood(
    student: PPO,
    env: gym.Env,
    teachers: dict[str: OnPolicyAlgorithm],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
):
    """
    Evaluate student-teacher similarity
    Utilises code from stable_baselines3's evaluate_policy method, modified code is highlighted with ## MODIFIED ##.

    :param student: Student policy.
    :param env: Environment for student to interact with.
    :param num_teachers: (int) Number of teachers
    :param n_eval_episodes: (int) Number of episodes to evaluate for
    :param deterministic: (bool) Whether to evaluate the deterministic or stochastic policy 
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    
    n_envs = env.num_envs
    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    ########################################### MODIFIED #######################################
    teacher_act_counts = {teacher_id: np.zeros((env.action_space.shape[0], 1))
                          for teacher_id in teachers.keys()}
    total_act_counts = np.zeros((env.action_space.shape[0], 1))
    
    s_states = None
    observations = env.reset()
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        s_acts, s_states = student.predict(
            observations,
            state=s_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        t_acts = {teacher_id: teacher_net.predict(observations, deterministic=deterministic)[0]
                  for teacher_id, teacher_net in teachers.items()}
        
        observations, rewards, dones, infos = env.step(s_acts)
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                for t_id in teachers.keys():
                    for a_idx in range(s_acts[i].shape[0]):
                        t_act = t_acts[t_id][i, a_idx]
                        s_act = s_acts[i, a_idx]
                        if t_act == s_act:
                            teacher_act_counts[t_id][a_idx, 0] += 1
                total_act_counts[:, 0] += 1

                if dones[i]:
                    episode_counts[i] += 1
    
    # Bernoulli mean and std
    teacher_act_means = {teacher_id: teacher_act_counts[teacher_id] / total_act_counts
                        for teacher_id in teachers.keys()}
    teacher_act_stds = {teacher_id: np.sqrt(
                            (teacher_act_means[teacher_id] * (1 - teacher_act_means[teacher_id])) / total_act_counts
                        ) for teacher_id in teachers.keys()}
    
    # Preprocess to lists for saving to json
    for teacher_id in teachers.keys():
        teacher_act_counts[teacher_id] = teacher_act_counts[teacher_id].tolist()
        teacher_act_means[teacher_id] = teacher_act_means[teacher_id].tolist()
        teacher_act_stds[teacher_id] = teacher_act_stds[teacher_id].tolist()

    return teacher_act_counts, teacher_act_means, teacher_act_stds
    ############################################################################################
