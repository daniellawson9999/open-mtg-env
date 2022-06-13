# %%
from deck import *
from game import *
from player import *
from phases import Phases
import random
import torch
import time
from decision_transformer.decision_transformer import DecisionTransformer

# %%
data = {
    "Silver": [],
    "Gold": []
}
# Tuple State, Legal_actions, action, reward

# %%

policies = [None,None]
wins=[0,0]
n_games = 500


def get_move(index, state, move_indexes=None, move_strings=None):
    assert(move_indexes != None or move_strings != None), "at least one list must not be None {} {}".format(move_indexes,move_strings)

    if (move_indexes != None):
        length = len(move_indexes)
    else:
        length = len(move_strings)

    if policies[index] == None or length == 1:
        if move_indexes != None:
            move = random.sample(move_indexes,1)[0]
        else:
            move = random.sample(move_strings,1)[0]
    else:
        assert (False), "TODO"
        pass
    return move

# %%
# Simulate n_names times
for i in range(n_games):
    print("Starting game %d" % (i + 1))
    # Create game
    #game = Game([Player(get_8ed_core_gold_deck(), name="Gold"), Player(get_8ed_core_silver_deck(), name="Silver")])
    game = Game([Player(get_8ed_core_gold_deck(), name="Player1", deck_name="Gold"), Player(get_8ed_core_gold_deck(), name="Player2", deck_name="Gold")])
    game.start_game()

    game_data = {
        game.players[0].name: [],
        game.players[1].name: []
    }
    while not game.is_over():
        # Get player, valid moves
        player = game.player_with_priority
        other_player = game.players[1 - player.index]

        active_player_life = player.life
        other_player_life = other_player.life
        
        move_indexes, move_strings = game.get_moves()
        state = game.get_board_string()

        life_total_1 =  game.players[0].life
        life_total_2 = game.players[0].life
        # Skip COMBAT_DAMAGE_STEP_510_1c, no choices in damage assignment
        if game.current_phase_index == Phases.COMBAT_DAMAGE_STEP_510_1c:
            assert(len(move_indexes) == 1), "Should only be 1 value move"
            game.make_move(move_indexes[0],False)            
        else:
            # Get move, may have to unroll action
            if isinstance(move_strings, ActionUnroller):
                unroller = move_strings
                while not unroller.is_done():
                    moves = unroller.get_legal_moves()
                    move_string = get_move(player.index,state, move_indexes=None, move_strings=moves)

                    # Record Move
                    if len(moves) > 1:
                        game_data[player.name].append([state, moves, move_string, 0])

                    #print(move_string)
                    state = unroller.register_move(move_string)
                # Apply unrolled actions to game
                unroller.make_move()
            else:
                #print(state)
                move = get_move(player.index, state, move_indexes, move_strings)
                move_string = move_strings[move_indexes.index(move)]

                # Record Move
                if len(move_indexes) > 1:
                    game_data[player.name].append([state, move_strings, move_string, 0])

                #print(move_string)
                game.make_move(move, False)
        #move = mcts.uct(game, itermax=5)
        final_active_player_life = player.life
        final_other_player_life = other_player.life

        # Add reward for decreasing opponent, penality for decreasing your own
        if final_other_player_life != other_player_life:
            game_data[player.name][-1][-1] += other_player_life - final_other_player_life
        if final_active_player_life != active_player_life:
            game_data[player.name][-1][-1] += final_active_player_life - active_player_life

    if game.players[1].has_lost:
        wins[0] += 1
        reward_0 = 1
        reward_1 = -1
        #print("State", state)
    elif game.players[0].has_lost:
        wins[1] += 1
        reward_0 = -1
        reward_1 = 1
        #print("State", state)
    # Add reward info
    name_0 = game.players[0].name
    name_1 = game.players[1].name
    len_0 = len(game_data[name_0])
    len_1 = len(game_data[name_1])
    # for i in range(max(len_0, len_1)):
    #     if i <= len_0 - 1:
    #         game_data[name_0][i].append(reward_0)
    #     if i <= len_1 - 1:
    #         game_data[name_1][i].append(reward_1)
    # Add winning and losing rewards
    outcome_constant = 100
    game_data[name_0][-1][-1] += reward_0 * outcome_constant 
    game_data[name_1][-1][-1] += reward_1 * outcome_constant
    # Add game_data to overall data
    for player in game.players:
        data[player.deck_name].append(game_data[player.name])
    
print("player 0 wins: %i, player 1 wins: %i" % (wins[0], wins[1]))


# %%
data['Gold'][0]

# %%
len(data['Silver'])

# %%
data['Gold'][0][0]

# %%
wins

# %%
max_per_game = -1
max_state_len = -1
# Count total amount of transitions
total_transitions = 0
for deck_name, deck_data in data.items():
    for game_data in deck_data:
        game_len = len(game_data)
        max_per_game = max(max_per_game, game_len)
        max_state_len = max(max_state_len, max([len(transition[0]) for transition in game_data]))
        total_transitions += game_len
print(total_transitions)
print(max_per_game)
print(max_state_len)

# %%
flat_data = []
for deck_type, deck_data in data.items():
    for game in deck_data:
        flat_data.extend(game)

# %%
gold_data = data['Gold']

# %%
flat_array = np.array(flat_data, dtype=object)

# %%
#np.savez_compressed('./data/test_data', flat_array)

# %%
actions = set(flat_array[:, 2])

# %%
print(len(actions))

# %%
print(data['Gold'][0][0])

# %%
import torchtext

# %%
def customer_tokenizer(text):
    return text.split("$")

# %%
def customer_tokenizer_whole(text):
    return [text]

# %%
import collections

# %%
counter_obj_actions = collections.Counter()

# %%
# for action in actions:
#     sub_actions = action.split("$")
#     counter_obj_actions.update(sub_actions)

# %%
for action in actions:
    counter_obj_actions.update([action])

# %%
counter_obj_actions

# %%
vocab_actions = torchtext.vocab.Vocab(counter_obj_actions, min_freq=1, specials=["<unk>"])

# %%
for i in range(len(vocab_actions)):
    print(i, vocab_actions.itos[i])

# %%
counter_obj_states = collections.Counter()

# %%
def process_state(state):
    return state.replace(' ', '').replace('\n', '').split('$')

# %%
print(flat_data[0][0].replace(' ', '').replace('\n', '').split('$'))

# %%
max_state_token_len = -1
for transition in flat_data:
    state = process_state(transition[0])
    max_state_token_len = max(max_state_token_len, len(state))
    counter_obj_states.update(state)


# %%
max_state_token_len

# %%
vocab_states = torchtext.vocab.Vocab(counter_obj_states, min_freq=1, specials=["<unk>"])

# %%
type(vocab_states)

# %%
len(vocab_states)

# %%
for i in range(len(vocab_states)):
    print(i, vocab_states.itos[i])

# %%
trajectories = []

# %%
for game in data['Gold']:
    game_data = {'states': [], 'actions': [], 'rewards':[], 'timesteps':[], 'state_masks':[]}
    timestep = 0
    for trajectory in game:
        state = [vocab_states[word] for word in process_state(trajectory[0])]
        mask = [1] * len(state)
        padding_len = max_state_token_len - len(state)
        mask.extend([0] * padding_len)
        state.extend([0] * padding_len)

        game_data['state_masks'].append(mask)
        game_data['states'].append(state)
        game_data['actions'].append(vocab_actions[trajectory[2]])
        game_data['rewards'].append(trajectory[3])
        game_data['timesteps'].append(timestep)
        timestep +=1
    for key,value in game_data.items():
        if not isinstance(value, np.ndarray):
            game_data[key] = np.array(value)
    trajectories.append(game_data)

# %%
embed_dim=128
embed_dim=256
dropout=.1

hidden_size=embed_dim
n_layer=3
n_head=1
n_head=4
n_inner=4*embed_dim
activation_function='relu'
n_positions=1024
resid_pdrop=dropout
attn_pdrop=dropout

device = 'cuda'
num_trajectories = len(trajectories)
K=20
max_ep_len = max_per_game
act_dim = 1

# %%
max_state_token_len

# %%
returns = []
for traj in trajectories:
    returns.append(sum(traj['rewards']))

# %%
print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
print("Number of transitions", total_transitions)

# %%
scale = np.mean(returns)
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
        )

        s, a, r, rtg, timesteps, mask, state_masks = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[batch_inds[i]]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            state_dim = max_state_token_len
            state_masks.append(traj['state_masks'][si:si + max_len].reshape(1, -1, state_dim))
            s.append(traj['states'][si:si + max_len].reshape(1, -1, max_state_token_len))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            tlen = a[-1].shape[1]

            
            timesteps.append(np.arange(si, si + tlen).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            state_masks[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), state_masks[-1]], axis=1)
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.int32, device=device)
        state_masks = torch.from_numpy(np.concatenate(state_masks, axis=0)).to(dtype=torch.int32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.int32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, state_masks, a, r, rtg, timesteps, mask

# %%
s, state_masks, a, r, rtg, timesteps, mask = get_batch()

# %%
r.shape

# %%
len(vocab_states)

# %%
learning_rate = 4e-4
warmup_steps = 10000
loss_fn = torch.nn.CrossEntropyLoss()
weight_decay = 1e-4


# %%
len(vocab_states)

# %%
model = DecisionTransformer(
    state_vocab_size=len(vocab_states),
    action_vocab_size=len(vocab_actions),
    act_dim=act_dim,
    hidden_size=embed_dim,
    max_length=K,
    max_ep_len=max_ep_len,
    n_layer=n_layer,
    n_head=n_head,
    n_inner=4*embed_dim,
    activation_function=activation_function,
    n_positions=1024,
    resid_pdrop=resid_pdrop,
    attn_pdrop=attn_pdrop
)
model = model.to(device=device)

# %%
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

# %%
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
params

# %%
def train_step(batch_size=256):
    states, state_masks, actions, rewards, rtg, timesteps, mask = get_batch(batch_size)
    action_target = torch.clone(actions)

    state_preds, action_preds, reward_preds = model.forward(
        states, actions, rtg[:,:-1], timesteps, state_masks, attention_mask=mask,
    )
    act_dim = action_preds.shape[2]
    action_preds = action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
    action_target = action_target.reshape(-1)[mask.reshape(-1) > 0]
    
    loss = loss_fn(action_preds,action_target.to(dtype=torch.long,device=device))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
    optimizer.step()

    # with torch.no_grad():
    #     diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

    return loss.detach().cpu().item()

# %%
def train_iteration(num_steps, iter_num=0, print_logs=False, eval_fns=[], batch_size=256):

        train_losses = []
        logs = dict()

        train_start = time.time()

        model.train()
        for _ in range(num_steps):
            train_loss = train_step(batch_size)
            train_losses.append(train_loss)
            if scheduler is not None:
                scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        model.eval()
        for eval_fn in eval_fns:
            outputs = eval_fn(model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        # logs['time/total'] = time.time() - self.start_time
        # logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

# %%
max_iters = 10
num_steps_per_iter = 10000

# %%
for i in range(max_iters):
    output = train_iteration(num_steps=num_steps_per_iter, iter_num=i+1, print_logs=True,batch_size=4)

# %%
output

# %%
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of params", params)

# %%
policies = [None,None]
wins=[0,0]
n_games = 1000

# %%
model.eval()
model.to(device=device)

# %%
def get_move_dt(state, actions, target_return, timesteps, state_mask, move_list,move_indexes=None):
    action_probs = model.get_action(
        state,
        actions, 
        target_return, 
        timesteps, 
        state_mask
    )
    action_probs = softmax(action_probs)
    probs = []
    for move in move_list:
        probs.append(action_probs[vocab_actions[move]].item())
    probs = [prob/sum(probs) for prob in probs]
    if move_indexes is None:
        final_move = np.random.choice(move_list, p=probs)
    else:
        final_move = np.random.choice(move_indexes, p=probs)
    return final_move

# %%
def state_list_to_tensor(state_list):
    padding_value = -1
    padded_states = torch.nn.utils.rnn.pad_sequence(state_list, batch_first=True, padding_value=-1)
    pad_mask = ~(padding_value == padded_states)
    padded_states[padded_states == padding_value] = 0
    return padded_states, pad_mask


# %%
eval_horizon = 20
target_return_value = np.max(returns) / scale
softmax = torch.nn.Softmax()
# Simulate n_names times
for i in range(n_games):
    # Create tensors
    processed_state_strings = []
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.long)
    
    target_return = torch.tensor(target_return_value, device=device, dtype=torch.float32).reshape(1,1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    action_steps = 0 

    state_tensors = []

    print("Starting game %d" % (i + 1))
    # Create game
    #game = Game([Player(get_8ed_core_gold_deck(), name="Gold"), Player(get_8ed_core_silver_deck(), name="Silver")])
    game = Game([Player(get_8ed_core_gold_deck(), name="Player1", deck_name="Gold"), Player(get_8ed_core_gold_deck(), name="Player2", deck_name="Gold")])
    game.start_game()

    game_data = {
        game.players[0].name: [],
        game.players[1].name: []
    }

    while not game.is_over():
        # Get player, valid moves
        player = game.player_with_priority
        move_indexes, move_strings = game.get_moves()
        state_string = game.get_board_string()
        state = np.array([vocab_states[word] for word in process_state(state_string)])
        #state_mask = torch.ones(len(state)).reshape(1,len(state)).to(device=device,dtype=torch.long)
        #state = torch.from_numpy(state).reshape(1, len(state)).to(device=device,dtype=torch.long)
        state = torch.from_numpy(state).to(device=device,dtype=torch.long)             
        

        # Skip COMBAT_DAMAGE_STEP_510_1c, no choices in damage assignment
        if game.current_phase_index == Phases.COMBAT_DAMAGE_STEP_510_1c:
            assert(len(move_indexes) == 1), "Should only be 1 value move"
            game.make_move(move_indexes[0],False)
            continue

        # Get move, may have to unroll action
        if isinstance(move_strings, ActionUnroller):
            unroller = move_strings
            while not unroller.is_done():
                moves = unroller.get_legal_moves()
                if player.index == 0 or len(moves) == 1:
                    move_string = get_move(player.index,state_string, move_indexes=None, move_strings=moves)
                else:
                    state = np.array([vocab_states[word] for word in process_state(state_string)])
                    #state_mask = torch.ones(len(state)).reshape(1,len(state)).to(device=device,dtype=torch.long)
                    #state = torch.from_numpy(state).reshape(1, len(state)).to(device=device,dtype=torch.long)
                    state = torch.from_numpy(state).to(device=device,dtype=torch.long)   
                    
                    #move_string = get_move_dt(state, actions, target_return, timesteps, state_mask, move_list=moves)
                    state_tensors.append(state)
                    states, states_mask = state_list_to_tensor(state_tensors)
                    
                    move_string = get_move_dt(states, actions, target_return, timesteps, states_mask, move_list=moves)
                    # Append to data structures
                    action_steps += 1
                    action_value = vocab_actions[move_string]
                    actions = torch.cat([actions, torch.tensor(action_value).reshape(1,1).to(device=device)], dim=0)
                    target_return = torch.cat([target_return, torch.tensor(target_return_value).reshape(1,1).to(device=device)], dim=1)
                    timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (action_steps)], dim=1)

                    # Truncate
                    timesteps = timesteps[:, -eval_horizon: ]
                    target_return = target_return[:, -eval_horizon:]
                    state_tensors = state_tensors[-eval_horizon:]
                    actions = actions[-eval_horizon:, :]



                # Record Move
                if len(moves) > 1:
                    game_data[player.name].append([state_string, moves, move_string])

                #print(move_string)
                state_string = unroller.register_move(move_string)


            # Apply unrolled actions to game
            unroller.make_move()
        else:
            #print(state)
            if player.index == 0 or len(move_strings) == 1:
                move = get_move(player.index, state_string, move_indexes, move_strings)
                move_string = move_strings[move_indexes.index(move)]
            else:
                state = np.array([vocab_states[word] for word in process_state(state_string)])
                #state_mask = torch.ones(len(state)).reshape(1,len(state)).to(device=device,dtype=torch.long)
                #state = torch.from_numpy(state).reshape(1, len(state)).to(device=device,dtype=torch.long)
                state = torch.from_numpy(state).to(device=device,dtype=torch.long) 
                state_tensors.append(state)
                states, states_mask = state_list_to_tensor(state_tensors)

                move = get_move_dt(states, actions, target_return, timesteps, states_mask, move_list=move_strings, move_indexes=move_indexes)
                lookup_move = int(move) if ( (type(move) == str or type(move) == np.str_) and move.isdigit()) else move
                move_string = move_strings[move_indexes.index(lookup_move)]

                # Append to data structures
                action_steps += 1
                action_value = vocab_actions[move_string]
                actions = torch.cat([actions, torch.tensor(action_value).reshape(1,1).to(device=device)], dim=0)
                target_return = torch.cat([target_return, torch.tensor(target_return_value).reshape(1,1).to(device=device)], dim=1)
                timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (action_steps)], dim=1)

                # Truncate
                timesteps = timesteps[:, -eval_horizon: ]
                target_return = target_return[:, -eval_horizon:]
                state_tensors = state_tensors[-eval_horizon:]
                actions = actions[-eval_horizon:, :]
            # Record Move
            if len(move_indexes) > 1:
                game_data[player.name].append([state_string, move_strings, move_string])

            #print(move_string)
            game.make_move(move, False)

            
        #move = mcts.uct(game, itermax=5)

    if game.players[1].has_lost:
        wins[0] += 1
        reward_0 = 1
        reward_1 = -1
        #print("State", state)
    elif game.players[0].has_lost:
        wins[1] += 1
        reward_0 = -1
        reward_1 = 1
        #print("State", state)
    # Add reward info
    name_0 = game.players[0].name
    name_1 = game.players[1].name
    len_0 = len(game_data[name_0])
    len_1 = len(game_data[name_1])
    for i in range(max(len_0, len_1)):
        if i <= len_0 - 1:
            game_data[name_0][i].append(reward_0)
        if i <= len_1 - 1:
            game_data[name_1][i].append(reward_1)
    # Add game_data to overall data
    for player in game.players:
        data[player.deck_name].append(game_data[player.name])
    
    print("player 0 wins: %i, player 1 wins: %i" % (wins[0], wins[1]))


# %%
# target_return = torch.tensor(1, device=device, dtype=torch.long).reshape(1,1)
# timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
# actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)

# # Simulate n_names times
# for i in range(n_games):
#     print("Starting game %d" % (i + 1))
#     # Create game
#     #game = Game([Player(get_8ed_core_gold_deck(), name="Gold"), Player(get_8ed_core_silver_deck(), name="Silver")])
#     game = Game([Player(get_8ed_core_gold_deck(), name="Player1", deck_name="Gold"), Player(get_8ed_core_gold_deck(), name="Player2", deck_name="Gold")])
#     game.start_game()

#     game_data = {
#         game.players[0].name: [],
#         game.players[1].name: []
#     }
#     while not game.is_over():
#         # Get player, valid moves
#         player = game.player_with_priority
#         move_indexes, move_strings = game.get_moves()
#         state_string = game.get_board_string()
#         state = np.array([vocab_states[word] for word in process_state(state_string)])
#         state_mask = torch.ones(len(state)).reshape(1,len(state)).to(device=device,dtype=torch.long)
#         state = torch.from_numpy(state).reshape(1, len(state)).to(device=device,dtype=torch.long)


#         # Skip COMBAT_DAMAGE_STEP_510_1c, no choices in damage assignment
#         if game.current_phase_index == Phases.COMBAT_DAMAGE_STEP_510_1c:
#             assert(len(move_indexes) == 1), "Should only be 1 value move"
#             game.make_move(move_indexes[0],False)
#             continue

#         # Get move, may have to unroll action
#         if isinstance(move_strings, ActionUnroller):
#             unroller = move_strings
#             while not unroller.is_done():
#                 moves = unroller.get_legal_moves()
#                 if player.index == 0:
#                     move_string = get_move(player.index,state_string, move_indexes=None, move_strings=moves)
#                 else:
#                     state = np.array([vocab_states[word] for word in process_state(state_string)])
#                     state_mask = torch.ones(len(state)).reshape(1,len(state)).to(device=device,dtype=torch.long)
#                     state = torch.from_numpy(state).reshape(1, len(state)).to(device=device,dtype=torch.long)
#                     action_probs = model.get_action(
#                         state,
#                         actions, 
#                         target_return, 
#                         timesteps, 
#                         state_mask
#                     )
#                     best_move_string = None
#                     best_move_val = None
#                     for move in moves:
#                         move_id = vocab_actions[move]
#                         if best_move_val is None or best_move_val < action_probs[move_id]:
#                             best_move_val = action_probs[move_id]
#                             best_move_string = move
#                     move_string = best_move_string

#                 # Record Move
#                 if len(moves) > 1:
#                     game_data[player.name].append([state_string, moves, move_string])

#                 #print(move_string)
#                 state_string = unroller.register_move(move_string)
#             # Apply unrolled actions to game
#             unroller.make_move()
#         else:
#             #print(state)
#             if player.index == 0:
#                 move = get_move(player.index, state_string, move_indexes, move_strings)
#                 move_string = move_strings[move_indexes.index(move)]
#             else:
#                 state = np.array([vocab_states[word] for word in process_state(state_string)])
#                 state_mask = torch.ones(len(state)).reshape(1,len(state)).to(device=device,dtype=torch.long)
#                 state = torch.from_numpy(state).reshape(1, len(state)).to(device=device,dtype=torch.long)
#                 action_probs = model.get_action(
#                     state,
#                     actions, 
#                     target_return, 
#                     timesteps, 
#                     state_mask
#                 )
#                 best_move_string = None
#                 best_move_val = None
#                 for move in moves:
#                     move_id = vocab_actions[move]
#                     if best_move_val is None or best_move_val < action_probs[move_id]:
#                         best_move_val = action_probs[move_id]
#                         best_move_string = move
#                 move = move_indexes[move_strings.index(best_move_string)]

#             # Record Move
#             if len(move_indexes) > 1:
#                 game_data[player.name].append([state_string, move_strings, move_string])

#             #print(move_string)
#             game.make_move(int(move), False)
#         #move = mcts.uct(game, itermax=5)

#     if game.players[1].has_lost:
#         wins[0] += 1
#         reward_0 = 1
#         reward_1 = -1
#         #print("State", state)
#     elif game.players[0].has_lost:
#         wins[1] += 1
#         reward_0 = -1
#         reward_1 = 1
#         #print("State", state)
#     # Add reward info
#     name_0 = game.players[0].name
#     name_1 = game.players[1].name
#     len_0 = len(game_data[name_0])
#     len_1 = len(game_data[name_1])
#     for i in range(max(len_0, len_1)):
#         if i <= len_0 - 1:
#             game_data[name_0][i].append(reward_0)
#         if i <= len_1 - 1:
#             game_data[name_1][i].append(reward_1)
#     # Add game_data to overall data
#     for player in game.players:
#         data[player.deck_name].append(game_data[player.name])
    
# print("player 0 wins: %i, player 1 wins: %i" % (wins[0], wins[1]))



