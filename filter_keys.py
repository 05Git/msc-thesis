filter_keys_1p = {
    "global": [
        # "P1_character",
        # "P2_character",
        # "P1_health",
        # "P2_health",
        # "P1_side",
        # "P2_side",
        # "timer",
        "frame"
    ],
    "sfiii3n": [],
}

filter_keys_2p = {
    "global": [
        # "P1_character",
        # "P2_character",
        "P1_health",
        "P2_health",
        # "P1_side",
        # "P2_side",
        "timer",
        "frame"
    ],
    "sfiii3n": [],
}

def get_filter_keys(game_id:str, num_players: int):
    assert num_players in [1,2]
    if num_players == 1:
        filter_keys = filter_keys_1p["global"]
        filter_keys.extend(filter_keys_1p[game_id])
    else:
        filter_keys = filter_keys_2p["global"]
        filter_keys.extend(filter_keys_2p[game_id])
    
    return filter_keys