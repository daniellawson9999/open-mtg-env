import argparse
import random
import os
import pickle

from open_mtg_env import mcts_env as mcts
from open_mtg_env.phases import Phases
from open_mtg_env.deck import *


from open_mtg_env.env import MtgEnv

# Collect data using MCTS
def main(args):
    # create environment
    env = MtgEnv(get_8ed_core_gold_deck, 'Gold0', get_8ed_core_gold_deck, 'Gold1')
    # collect data
    states, legal_actions, actions, rewards = collect_data(env, args)
    # save data
    dataset = {
        'states': states,
        'legal_actions': legal_actions,
        'actions': actions,
        'rewards': rewards
    }
    data_id = random.randint(int(1e4), int(1e5) - 1)
    data_name = f'data_{args.mode}_g{args.n_games}_d{args.depth}_{data_id}.pkl'
    data_path = os.path.join('data', data_name)
    with open(data_path, 'wb') as f:
        pickle.dump(dataset, f)
    


def get_action(active_player_index, policies, env, possible_moves, args):
    policy = policies[active_player_index]
    if len(possible_moves) == 1:
        action = possible_moves[0]
    else:
        if policy == 'mcts':
            action = mcts.uct(env, itermax=args.depth)
            if type(action) == int:
                action = possible_moves[action]
            elif type(action) == tuple:
                action = action[0]
        else:
            action = random.choice(possible_moves)
    return action

def collect_data(env, args):
    states = []
    legal_actions = []
    actions = []
    rewards = []
    for i in range(args.n_games):
        if args.mode == 'mcts_2':
            policies = ['mcts', 'mcts']
        else:
            policies = ['mcts', 'random']
        random.shuffle(policies)

        player_states = [[], []]
        player_actions = [[], []]
        player_legal_actions = [[], []]

        done = False
        possible_moves, state, active_player_index = env.reset()
        while not done:
            action = get_action(active_player_index, policies, env, possible_moves, args)
            # store transition
            if len(possible_moves) > 1:
                player_states[active_player_index].append(state)
                player_legal_actions[active_player_index].append(possible_moves)
                player_actions[active_player_index].append(action)
            state, possible_moves, active_player_index, done, info = env.step(action)

        print(f"Player {info['winning_player']} wins w/ policy policy {policies[info['winning_player']]}")

        player_rewards = []
        # Add rewards (just game outcome)
        for i in range(2):
            if info['winning_player'] == i:
                reward = 1
            else:
                reward = 0
            player_rewards.append([reward] * len(player_states[i]))

        # add player data to overall data
        for i in range(2):
            states.append(player_states[i])
            legal_actions.append(player_legal_actions[i])
            actions.append(player_actions[i])
            rewards.append(player_rewards[i])
    
    return states, legal_actions, actions, rewards
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=10)
    parser.add_argument('--depth', type=int, default=5)
    # choice for mcts vs mcsts or mcts vs random
    parser.add_argument('--mode', type=str, default='mcts_2', choices=['mcts_2', 'mcts_random'])
    args = parser.parse_args()
    main(args)