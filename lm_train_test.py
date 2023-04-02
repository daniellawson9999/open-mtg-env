from deck import *
from game import *
from player import *
from phases import Phases
import copy
from env import MtgEnv
import random

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler, respond_to_batch

from lm_utils import run_games 

tqdm.pandas()

def main(script_args):
    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        init_kl_coef=script_args.init_kl_coef,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_8bit=True,
        peft_config=lora_config,
        layer_norm_names=[],
    )

    # Apply LoRA
    # Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
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


    print_trainable_parameters(model)

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tokenizer)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = model.current_device if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    env = MtgEnv(get_8ed_core_gold_deck, 'Gold0', get_8ed_core_gold_deck, 'Gold1')

    n_iterations = 50
    train_games = 3
    eval_games = 3

    # train_games = 3
    # eval_games = 10

    for iteration in range(n_iterations):
        # collect training data from self-play
        print(f"Starting iteration {iteration}")
        #queries, actions, rewards, player_0_wins, player_1_wins = run_games(env, model, tokenizer, device, n_games=train_games, mode_0='lm', mode_1='lm', lm_sample=True)
        queries, actions, rewards, player_0_wins, player_1_wins = run_games(env, model, tokenizer, device, n_games=train_games, mode_0='random', mode_1='random', lm_sample=True)
        #import pdb; pdb.set_trace()
        # shuffle order of queries, actions, rewards
        bundle = list(zip(queries, actions, rewards))
        random.shuffle(bundle)
        queries, actions, rewards = zip(*bundle)
        queries = list(queries)
        actions = list(actions)
        rewards = list(rewards)

        for i in range(0, len(queries), ppo_trainer.config.batch_size):
            # train model for one step with ppo
            queries_batch = queries[i:i+ppo_trainer.config.batch_size]
            actions_batch = actions[i:i+ppo_trainer.config.batch_size]
            rewards_batch = rewards[i:i+ppo_trainer.config.batch_size]
            if len(queries_batch) < ppo_trainer.config.batch_size:
                break
            model.gradient_checkpointing_enable()
            model.pretrained_model.config.use_cache = False

            train_stats = ppo_trainer.step(queries_batch, actions_batch, rewards_batch)
            #import pdb; pdb.set_trace()

        # evaluate the model against random
        _, _, _, player_0_wins, player_1_wins = run_games(env, model, tokenizer, device, n_games=eval_games, mode_0='lm', mode_1='random', lm_sample=False)
        print(f"Player 0 wins (lm): {player_0_wins}, Player 1 wins (random): {player_1_wins}")

    # import pdb; pdb.set_trace()
    #_, _, _, player_0_wins, player_1_wins = run_games(env, model, tokenizer, device, n_games=eval_games, mode_0='lm', mode_1='human', lm_sample=False,print_state=True)
if __name__ == '__main__':
    @dataclass
    class ScriptArguments:
        """
        The name of the Casual LM model we wish to fine with PPO
        """

        # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
        # models like gpt-neo* models are more suitable.
        #model_name: Optional[str] = field(default="edbeeching/gpt-neo-125M-imdb", metadata={"help": "the model name"})
        model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125M", metadata={"help": "the model name"})
        #model_name: Optional[str] = field(default="EleutherAI/gpt-neo-1.3B", metadata={"help": "the model name"})
        log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
        learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
        mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
        batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"}) # 256
        gradient_accumulation_steps: Optional[int] = field(
            default=1, metadata={"help": "the number of gradient accumulation steps"}
        )
        adap_kl_ctrl: Optional[bool] = field(default=False, metadata={"help": "whether to use adaptive KL control"})
        #init_kl_coef: Optional[float] = field(default=0.2, metadata={"help": "the initial KL coefficient"}),
        init_kl_coef: Optional[float] = field(default=0.0, metadata={"help": "the initial KL coefficient"}) # set to negative temp


    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)