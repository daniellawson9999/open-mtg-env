import random
import os
import pickle
import copy
import argparse

from open_mtg_env.deck import *
from open_mtg_env.game import *
from open_mtg_env.player import *
from open_mtg_env.phases import Phases
from open_mtg_env.env import MtgEnv

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, Trainer, TrainingArguments, logging, AutoModelForCausalLM,TrainerCallback

#from trl import AutoModelForCausalLM, set_seed
from trl.core import LengthSampler, respond_to_batch

from lm_utils import run_games

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class MtgDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        #item['labels'] = item['input_ids'].clone()
        return item
    def __len__(self):
        return len(self.encodings)

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #trust_remote_code=True,
        #use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map='auto',
        #device_map={"": Accelerator().process_index},
    )
   # model = prepare_model_for_int8_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    # prepare custom dataset
    data_name = 'data_mcts_2_g250_d10_74432' + '.pkl'
    with open(os.path.join('data', data_name),'rb') as f:
        data = pickle.load(f)
    
    env = MtgEnv(get_8ed_core_gold_deck, 'Gold0', get_8ed_core_gold_deck, 'Gold1') # TODO

    input_strs = []
    # iterate over trajectories
    max_traj_len = -1
    input_lens = []
    n_trajectories = 0
    for i in range(len(data['states'])):
        # check if winning trajectory
        if data['rewards'][i][0] != 1:
            continue
        else:
            n_trajectories += 1
            
        traj_len = len(data['states'][i])
        if traj_len > max_traj_len:
            max_traj_len = traj_len

        for j in range(traj_len):
            state = data['states'][i][j]
            possible_moves = data['legal_actions'][i][j]
            action = data['actions'][i][j]
            query = env.state_action_to_query(state, possible_moves)
            response = env.format_action(action)
            input_str = query + response
            input_len = len(input_str)
            input_lens.append(input_len)
            input_strs.append(input_str)
    # create dataset
    print(f'Creating dataset with {n_trajectories}, max trajectory len {max_traj_len}, max input len {max(input_lens)}, mean input len: {np.mean(input_lens)}')

    input_encodings = tokenizer(input_strs, padding=True, return_offsets_mapping=True) #truncation=True, maybe adjust
    labels = []
    for i in range(len(input_encodings['input_ids'])):
        offset_map = input_encodings['offset_mapping'][i]
        first_index, last_index = zip(*offset_map)
        last_index = np.array(last_index)
        target_str = "best:" # make sure this is the same as in env
        # TODO
        # get start of response
        #response_index = input_strs[i].index(target_str)
        start_of_output = input_strs[0].index(target_str) + len(target_str)
        np_ids = input_encodings['input_ids'][i]
        label = np.where(last_index < start_of_output,-100,np_ids)
        labels.append(label)

    train_dataset = MtgDataset(input_encodings, labels)

    #import pdb; pdb.set_trace()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        #run_name="gpt-test",
        #report_to="wandb",
        report_to= "none",
        ddp_find_unused_parameters=False,
    )
    # create trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    # add custom eval callback for running games every eval_freq steps
    eval_freq = 2 #1
    games_per_eval = 10
    # eval_freq = 1
    # games_per_eval = 1

    class EvaluateGameCallback(TrainerCallback):
        def __init__(self, model, tokenizer, env, args, eval_freq=10, games_per_eval=5):
            self.env = env
            self.args = args
            self.eval_freq = eval_freq
            self.games_per_eval = games_per_eval
            self.model = model
            self.tokenizer = tokenizer
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.eval_freq == 0:
                queries, actions, rewards, player_0_wins, player_1_wins = run_games(env, self.model, self.tokenizer, args.device, n_games=self.games_per_eval, mode_0='lm', mode_1='mcts', lm_sample=False, print_state=False, depth=self.args.depth)
                print(f"LM vs MCTS depth {self.args.depth}")
                print("Player 0 wins: ", player_0_wins)
                print("Player 1 wins: ", player_1_wins)



    #trainer.add_callback(EvaluateGameCallback(model, tokenizer, env, args, eval_freq=eval_freq, games_per_eval=games_per_eval))

    trainer.train()
    import pdb; pdb.set_trace()

    # eval model
    queries, actions, rewards, player_0_wins, player_1_wins = run_games(env, model, tokenizer, args.device, n_games=args.eval_games, mode_0='lm', mode_1='mcts', lm_sample=False, print_state=False, depth=args.depth)
    print(f"LM vs MCTS depth {args.depth}")
    print("Player 0 wins: ", player_0_wins)
    print("Player 1 wins: ", player_1_wins)

    queries, actions, rewards, player_0_wins, player_1_wins = run_games(env, model, tokenizer, args.device, n_games=args.eval_games, mode_0='lm', mode_1='random', lm_sample=False, print_state=False, depth=args.depth)
    print("LM vs Random")
    print("Player 0 wins: ", player_0_wins)
    print("Player 1 wins: ", player_1_wins)



if __name__ == '__main__':
    # @dataclass
    # class ScriptArguments:
    #     """
    #     The name of the Casual LM model we wish to fine with PPO
    #     """

    #     # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    #     # models like gpt-neo* models are more suitable.
    #     #model_name: Optional[str] = field(default="edbeeching/gpt-neo-125M-imdb", metadata={"help": "the model name"})
    #     model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125M", metadata={"help": "the model name"})
    #     #model_name: Optional[str] = field(default="EleutherAI/gpt-neo-1.3B", metadata={"help": "the model name"})
    #     log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    #     learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    #     mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    #     batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"}) # 256
    #     gradient_accumulation_steps: Optional[int] = field(
    #         default=1, metadata={"help": "the number of gradient accumulation steps"}
    #     )

    # parser = HfArgumentParser(ScriptArguments)
    # script_args = parser.parse_args_into_dataclasses()[0]

    # instead use argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', type=str, default='EleutherAI/gpt-neo-125M')
    parser.add_argument('--device', type=str, default='cuda:0'),
    parser.add_argument('--model_name', type=str, default='EleutherAI/Pythia-410m') #1b, 410m worked..Pythia-410m
    #parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--output_dir", default="./checkpoints", type=str)

    parser.add_argument("--eval_games", default='5', type=int)
    parser.add_argument('--depth', type=int, default=5)

    args = parser.parse_args()
    # TODO mask idea, use this, return_offsets_mapping=True
    # https://stackoverflow.com/questions/63413414/is-there-a-way-to-get-the-location-of-the-substring-from-which-a-certain-token-h
    main(args)