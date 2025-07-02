import os
import torch as th
import custom_wrappers as cw

from stable_baselines3 import PPO
from diambra.arena import EnvironmentSettings, EnvironmentSettingsMultiAgent, WrappersSettings, SpaceTypes, load_settings_flat_dict

env_settings = {
    "1_player": {
        "shared": {
            "step_ratio": 6,
            "frame_shape": (84, 84, 1),
            "continue_game": 0.,
            "action_space": SpaceTypes.MULTI_DISCRETE,
            "outfits": 1,
            "splash_screen": False,
        },
        "sfiii3n": {
            "train": {
                "game_id": "sfiii3n",
                "characters": "Ryu",
                "difficulty": 8,
                "super_art": 1,
            },
            "eval": {
                "game_id": "sfiii3n",
                "characters": "Ryu",
                "difficulty": 6,
                "super_art": 1,
            },
        },
    },
    "2_player": {
        "shared": {
            "step_ratio": 6,
            "frame_shape": (84, 84, 1),
            "continue_game": 0.,
            "action_space": (SpaceTypes.MULTI_DISCRETE, SpaceTypes.MULTI_DISCRETE),
            "outfits": (1, 1),
            "splash_screen": False,
        },
        "sfiii3n": {
            "train": {
                "characters": ("Ryu", None),
                "difficulty": 1,
                "super_art": (1, 1),
            },
            "eval": {
                "characters": [
                    ("Ryu", "Alex"),
                    ("Ryu", "Gouki"),
                    ("Ryu", "Hugo"),
                    ("Ryu", "Ibuki"),
                ],
                "difficulty": 1,
                "super_art": (1, 1),
            },
        },
    },
    "seed": 0,
    "num_train_envs": 8,
    "num_eval_envs": 1,
}

wrappers_settings = {
    "normalize_reward": True,
    "normalization_factor": 1.,
    "no_attack_buttons_combinations": False,
    "stack_frames": 4,
    "dilation": 1,
    "add_last_action": False,
    "stack_actions": 1,
    "repeat_action": 1,
    "scale": False,
    "exclude_image_scaling": True,
    "role_relative": False,
    "filter_keys": ["frame"],
    "flatten": True,
    "wrappers": [],
}

teacher_paths = [
    # "experts/ryu_vs_alex/attack_expert_rand/model/seed_0/500000.zip",
    # "experts/ryu_vs_alex/def_expert_rand/model/seed_0/500000.zip",
    # "experts/ryu_vs_alex/anti_air_expert/model/seed_0/500000.zip",
]

def load_teachers():
    teachers = []
    for path in teacher_paths:
        teacher = PPO.load(path=path, device=ppo_settings["device"])
        teachers.append(teacher)
    
    return teachers

wrappers_1p = [
    [cw.ActionWrapper1P, {
        "action_space": "discrete" if env_settings["1_player"]["shared"]["action_space"] == SpaceTypes.DISCRETE else "multi_discrete",
        "no_op": 0,
        "max_actions": [9,11],
    }],
    [cw.PixelObsWrapper, {"stack_frames": wrappers_settings["stack_frames"]}],
]

wrappers_2p = [
    [cw.ActionWrapper2P, {
        "action_space": "discrete" if env_settings["2_player"]["shared"]["action_space"][0] == SpaceTypes.DISCRETE else "multi_discrete",
        "no_op": 0,
        "max_actions": [9,11],
        "opp_type": "jump",
    }],
    [cw.AttTrainWrapper, {}],
    [cw.PixelObsWrapper, {"stack_frames": wrappers_settings["stack_frames"]}],
]

folders = {
    "parent_dir": "experts/ryu_vs_rand",
    "model_name": "aa_expert",
}

policy_kwargs = {}

n_steps = 128
nminibatches = 8
batch_lambda = 8
batch_size = ((n_steps * env_settings["num_train_envs"]) // nminibatches) * batch_lambda
assert (n_steps * env_settings["num_train_envs"]) % nminibatches == 0

ppo_settings = {
    "policy": "CnnPolicy",
    "model_checkpoint": "0",
    "time_steps": 500_000,
    "device": th.device("cuda" if th.cuda.is_available else "cpu"),
    "gamma": 0.99,
    "train_lr": (2.5e-5, 2.5e-6),
    "finetune_lr": (5.0e-5, 2.5e-6),
    "train_cr": (0.15, 0.025),
    "finetune_cr": (0.075, 0.025),
    "batch_size": batch_size,
    "n_epochs": 10,
    "n_steps": n_steps,
    "gae_lambda": 0.95,
    "ent_coef": 0.,
    "vf_coef": 0.5,
    "target_kl": None,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "normalize_advantage": True,
    "stats_window_size": 100,
}

callbacks_settings = {
    "autosave_freq": 200_000,
    "n_eval_episodes": 100,
    "eval_freq": 200_000,
    "evaluate_during_training": False,
    "stop_training_if_no_improvement": False,
}

imitation_settings = {
    "bc": {
        "n_epochs": 100
    },
    "dagger": {
        "n_steps": 100_000
    },
    "gail": {
        "n_steps": 100_000
    },
}


def load_1p_settings(game_id: str):
    general_settings = env_settings["1_player"]["shared"]
    game_settings = env_settings["1_player"][game_id]
    train_settings = game_settings["train"]
    eval_settings = game_settings["eval"]

    train_wrappers = wrappers_settings.copy()
    train_wrappers["wrappers"] = wrappers_1p
    if type(train_settings["characters"]) == list:
        train_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": train_settings["characters"],
            "one_p_env": True,
        }])
    
    eval_wrappers = wrappers_settings.copy()
    eval_wrappers["wrappers"] = wrappers_1p
    if type(eval_settings["characters"]) == list:
        eval_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": eval_settings["characters"],
            "one_p_env": True,
        }])
    
    if type(train_settings["characters"]) == list:
        train_settings["characters"] = train_settings["characters"][0]
    
    if type(eval_settings["characters"]) == list:
        eval_settings["characters"] = eval_settings["characters"][0]

    train_settings.update(general_settings)
    eval_settings.update(general_settings)
    train_settings = load_settings_flat_dict(EnvironmentSettings, train_settings)
    eval_settings = load_settings_flat_dict(EnvironmentSettings, eval_settings)
    train_wrappers = load_settings_flat_dict(WrappersSettings, train_wrappers)
    eval_wrappers = load_settings_flat_dict(WrappersSettings, eval_wrappers)
    
    if teacher_paths:
        teacher_wrapper = [cw.TeacherInputWrapper, {
            "teachers": load_teachers(),
            "timesteps": ppo_settings["time_steps"],
            "deterministic": True,
            "teacher_action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete",
            "use_teacher_actions": False,
            "initial_epsilon": 1.,
        }]
        train_wrappers.wrappers.append(teacher_wrapper)
        eval_wrappers.wrappers.append(teacher_wrapper)
    
    # train_wrappers.wrappers.append(
    #     [cw.AddLastActions, {"action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete"}]
    # )
    # eval_wrappers.wrappers.append(
    #     [cw.AddLastActions, {"action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete"}]
    # )
    
    return train_settings, eval_settings, train_wrappers, eval_wrappers


def load_2p_settings(game_id: str):
    general_settings = env_settings["2_player"]["shared"]
    game_settings = env_settings["2_player"][game_id]
    train_settings = game_settings["train"]
    eval_settings = game_settings["eval"]

    wrappers_settings["filter_keys"].append("P1_health")
    wrappers_settings["filter_keys"].append("P2_health")

    train_wrappers = wrappers_settings.copy()
    train_wrappers["wrappers"] = wrappers_2p
    if type(train_settings["characters"]) == list:
        train_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": train_settings["characters"],
            "one_p_env": False,
        }])
    
    eval_wrappers = wrappers_settings.copy()
    eval_wrappers["wrappers"] = wrappers_2p
    if type(eval_settings["characters"]) == list:
        eval_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": eval_settings["characters"],
            "one_p_env": False,
        }])

    if type(train_settings["characters"]) == list:
        train_settings["characters"] = train_settings["characters"][0]
        
    if type(eval_settings["characters"]) == list:
        eval_settings["characters"] = eval_settings["characters"][0]

    train_settings.update(general_settings)
    eval_settings.update(general_settings)
    train_settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, train_settings)
    eval_settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, eval_settings)
    train_wrappers = load_settings_flat_dict(WrappersSettings, train_wrappers)
    eval_wrappers = load_settings_flat_dict(WrappersSettings, eval_wrappers)

    if teacher_paths:
        teacher_wrapper = [cw.TeacherInputWrapper, {
            "teachers": load_teachers(),
            "timesteps": ppo_settings["time_steps"],
            "deterministic": True,
            "teacher_action_space": "discrete" if general_settings["action_space"][0] == SpaceTypes.DISCRETE else "multi_discrete",
            "use_teacher_actions": False,
            "initial_epsilon": 1.,
        }]
        train_wrappers.wrappers.append(teacher_wrapper)
        eval_wrappers.wrappers.append(teacher_wrapper)

    # train_wrappers.wrappers.append(
    #     [cw.AddLastActions, {"action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete"}]
    # )
    # eval_wrappers.wrappers.append(
    #     [cw.AddLastActions, {"action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete"}]
    # )
    
    return train_settings, eval_settings, train_wrappers, eval_wrappers


base_path = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(
    base_path,
    folders["parent_dir"],
    folders["model_name"],
    "model"
)
tensor_board_folder = os.path.join(
    base_path,
    folders["parent_dir"],
    folders["model_name"],
    "tb"
)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(tensor_board_folder, exist_ok=True)

game_ids = [
    "sfiii3n",
    # "samsh5sp",
    # "kof98umh",
    # "umk3",
]
