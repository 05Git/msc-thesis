import gymnasium as gym

MB_VALID_MOVES_BY_GAME_ID = {
    "sfiii": {"moves": 9, "attacks": 7},
    "umk3": {"moves": 9, "attacks": 7},
    "kof98umh": {"moves": 9, "attacks": 5},
    "samsh5sp": {"moves": 9, "attacks": 5},
}

class MBTransferActionWrapper(gym.ActionWrapper):
    def __init__(self, game_id: str, env, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.valid_moves = MB_VALID_MOVES_BY_GAME_ID[game_id]["moves"]
        self.valid_attacks = MB_VALID_MOVES_BY_GAME_ID[game_id]["attacks"]
        self.action_space = gym.spaces.MultiDiscrete([self.valid_moves, self.valid_attacks])
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx

    def action(self, action):
        move_idx, attack_idx = action
        move = move_idx if move_idx < self.valid_moves else self.no_move_idx
        attack = attack_idx if attack_idx < self.valid_attacks else self.no_attack_idx
        return move, attack