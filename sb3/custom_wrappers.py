import gymnasium as gym

MB_VALID_MOVES_BY_GAME_ID = {
    "sfiii3n": {"moves": 9, "attacks": 7},
    "umk3": {"moves": 9, "attacks": 7},
    "kof98umh": {"moves": 9, "attacks": 5},
    "samsh5sp": {"moves": 9, "attacks": 5},
}

class MBTransferActionWrapper(gym.Wrapper):
    def __init__(self, game_id: str, env, no_move_idx: int = 0, no_attack_idx: int = 0):
        super().__init__(env)
        self.valid_moves = MB_VALID_MOVES_BY_GAME_ID[game_id]["moves"]
        self.valid_attacks = MB_VALID_MOVES_BY_GAME_ID[game_id]["attacks"]
        self.action_space = gym.spaces.MultiDiscrete([9, 7]) # 8 moves, 6 attacks, plus 1 no-op each
        self.env.action_space = self.action_space # Set action space of underlying env for consistency
        self.no_move_idx = no_move_idx
        self.no_attack_idx = no_attack_idx

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        '''
        Checks if a chosen action is within a game's valid move list.
        If not, replaces it with no-op.
        '''
        move_idx, attack_idx = action
        move = move_idx if move_idx < self.valid_moves else self.no_move_idx
        attack = attack_idx if attack_idx < self.valid_attacks else self.no_attack_idx
        return self.env.step([[move, attack]])