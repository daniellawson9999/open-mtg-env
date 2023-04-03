import random

import torch
import numpy as np


def score_response(model, tokenizer, query, response, device):
    # encode a query
    input_txt = query + response
    input_tensor = tokenizer.encode(input_txt, return_tensors="pt").to(device)

    # get model output, and then logits
    output = model(input_tensor)
    logits = output[0][0]
    input_tensor = input_tensor[0]
    score = 0
    for i in range(len(input_tensor)):
        score += logits[i][input_tensor[i]].item()
    return score

def get_action_lm(env, model, tokenizer, state, possible_moves, device, sample_action=False):
    query = env.state_action_to_query(state, possible_moves)
    scores = []
    for action in possible_moves:
        action = env.format_action(action)
        score = score_response(model, tokenizer, query, action, device)
        scores.append(score)
    if not sample_action:
        predicted_action = possible_moves[np.argmax(scores)]
    else:
        predicted_action = np.random.choice(possible_moves, p=torch.softmax(torch.tensor(scores),dim=-1).numpy())
    return predicted_action

def get_action(env, model, tokenizer, state, possible_moves, mode, device, sample_action=False):
    if len(possible_moves) == 1:
            action = possible_moves[0] 
    elif mode == 'random':
        action = random.sample(range(len(possible_moves)), 1)[0]
        action = possible_moves[action]
    elif mode == 'lm':
        action = get_action_lm(env, model, tokenizer, state, possible_moves, device, sample_action)
    elif mode == 'human':
        action_index = input("Enter action: ")
        action = possible_moves[int(action_index)]
    else:
        raise Exception("invalid model type")
    return action


def run_games(env, model, tokenizer, device, n_games=1, mode_0='random', mode_1='random', lm_sample=False, print_state=False):
    assert mode_0 in ['random', 'lm', 'human'] and mode_1 in ['random', 'lm', 'human'], "invalid model type"
    modes = [mode_0, mode_1]

    queries = []
    actions = []
    rewards = []

    player_0_wins = 0
    player_1_wins = 0

    for i in range(n_games):
        player_0_queries = []
        player_1_queries = []
        player_0_actions = []
        player_1_actions = []

        done = False
        possible_moves, state, active_player_index = env.reset()
        while not done:
            query_str = env.state_action_to_query(state, possible_moves)
            if print_state:
                print(query_str)

            action = get_action(env, model, tokenizer, state, possible_moves, modes[active_player_index], device, sample_action=lm_sample)

            # Store transition
            if len(possible_moves) > 1:
                query = tokenizer.encode(query_str, return_tensors="pt")[0]
                if active_player_index == 0:
                    player_0_queries.append(query)
                    player_0_actions.append(tokenizer.encode(env.format_action(action), return_tensors="pt")[0])
                else:
                    player_1_queries.append(query)
                    player_1_actions.append(tokenizer.encode(env.format_action(action), return_tensors="pt")[0])

            state, possible_moves, active_player_index, done, info = env.step(action)

        # get reward for each player from game outcome
        win_reward = 1.0
        lose_rewad = 0.1
        if info['winning_player'] == 0:
            player_0_wins += 1
            reward_0 = win_reward
            reward_1 = lose_rewad
            #reward_1 = 0
        else:  
            player_1_wins += 1
            reward_0 = lose_rewad
            #reward_0 = 0
            reward_1 = win_reward
        # TODO, remove
        # reward_0 =1.0
        # reward_1 = 1.0
        player_0_rewards = [torch.tensor([reward_0]) for i in range(len(player_0_queries))]
        player_1_rewards = [torch.tensor([reward_1]) for i in range(len(player_1_queries))]
        queries.extend(player_0_queries + player_1_queries)
        actions.extend(player_0_actions + player_1_actions)
        rewards.extend(player_0_rewards + player_1_rewards)
    return queries, actions, rewards, player_0_wins, player_1_wins