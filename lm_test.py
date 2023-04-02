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

tqdm.pandas()

# Define and parse arguments.
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
    mini_batch_size: Optional[int] = field(default=16, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.mini_batch_size}

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

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

def respond(query, txt_len=20, top_k=0, top_p=1.0):
    # encode a query
    query_tensor = tokenizer.encode(query, return_tensors="pt").to(device)

    # get model response
    response_tensor  = respond_to_batch(model, query_tensor, txt_len, top_k, top_p)
    return tokenizer.decode(response_tensor[0])


env = MtgEnv(get_8ed_core_gold_deck, 'Gold0', get_8ed_core_gold_deck, 'Gold1')
n_games = 250

queries = []
actions = []
rewards = []

for i in range(n_games):
    player_0_queries = []
    player_1_queries = []
    player_0_actions = []
    player_1_actions = []

    done = False
    possible_moves, state, active_player_index = env.reset()
    while not done:
        # print(f"Player {active_player_index} turn")
        # print(state)
        # print(possible_moves)
        # input action index
        if len(possible_moves) == 1:
            action = 0
        else:
            #action = int(input("Enter action: "))
            action = random.sample(range(len(possible_moves)), 1)[0]
        action = possible_moves[action]

        # Store transition
        if len(possible_moves) > 1:
            if active_player_index == 0:
                player_0_queries.append(tokenizer.encode(env.state_action_to_query(state, possible_moves), return_tensors="pt")[0])
                player_0_actions.append(tokenizer.encode(env.format_action(action), return_tensors="pt")[0])
            else:
                player_1_queries.append(tokenizer.encode(env.state_action_to_query(state, possible_moves), return_tensors="pt")[0])
                player_1_actions.append(tokenizer.encode(env.format_action(action), return_tensors="pt")[0])

        state, possible_moves, active_player_index, done, info = env.step(action)

    # get reward for each player from game outcome
    if info['winning_player'] == 0:
        reward_0 = 1.0
        reward_1 = -1.0
    else:
        reward_0 = -1.0
        reward_1 = 1.0
    player_0_rewards = [torch.tensor([reward_0]) for i in range(len(player_0_queries))]
    player_1_rewards = [torch.tensor([reward_1]) for i in range(len(player_1_queries))]
    queries.extend(player_0_queries + player_1_queries)
    actions.extend(player_0_actions + player_1_actions)
    rewards.extend(player_0_rewards + player_1_rewards)

# train model for one step with ppo
# adjusted_size = 2 ** int(math.log(len(queries),2))
# queries = queries[:adjusted_size]
# actions = actions[:adjusted_size]
# rewards = rewards[:adjusted_size]
# ppo_trainer.config.batch_size = adjusted_size
#ppo_trainer.config.mini_batch_size = 4
ppo_trainer.config.mini_batch_size = 4
# iterate over batch_size chunks
for i in range(0, len(queries), ppo_trainer.config.batch_size):
    # train model for one step with ppo
    queries_batch = queries[i:i+ppo_trainer.config.batch_size]
    actions_batch = actions[i:i+ppo_trainer.config.batch_size]
    rewards_batch = rewards[i:i+ppo_trainer.config.batch_size]
    if len(queries_batch) < ppo_trainer.config.batch_size:
        break
    train_stats = ppo_trainer.step(queries_batch, actions_batch, rewards_batch)

import pdb; pdb.set_trace()