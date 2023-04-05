from open_mtg_env import mcts_env as mcts
from open_mtg_env.deck import *
from open_mtg_env.game import *
from open_mtg_env.player import *
from open_mtg_env.phases import Phases

from open_mtg_env.env import MtgEnv

# Testing MCTS over new environment class

env = MtgEnv(get_8ed_core_gold_deck, 'Gold0', get_8ed_core_gold_deck, 'Gold1')
n_games = 10
depth = 5
for i in range(n_games):
    done = False
    possible_moves, state, active_player_index = env.reset()
    while not done:
        #print(f"Player {active_player_index} turn")
        #print(state)
        #print(possible_moves)
        # input action index
        if len(possible_moves) == 1:
            action = possible_moves[0]
        else:
            # player 0 is MCTS
            if active_player_index == 0:
                action = mcts.uct(env, itermax=depth)
                if type(action) == int:
                    action = possible_moves[action]
                elif type(action) == tuple:
                    action = action[0]
                #import pdb; pdb.set_trace()
            else:
                # random action
                action = random.choice(possible_moves)
        state, possible_moves, active_player_index, done, info = env.step(action)
    print(f"Player {info['winning_player']} wins")
