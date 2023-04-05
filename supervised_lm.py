from open_mtg_env.deck import *
from open_mtg_env.game import *
from open_mtg_env.player import *
from open_mtg_env.phases import Phases
import copy
from open_mtg_env.env import MtgEnv
import random
import os
import pickle

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, Trainer, TrainingArguments, logging, AutoModelForCausalLM

#from trl import AutoModelForCausalLM, set_seed
from trl.core import LengthSampler, respond_to_batch

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
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return {'text': self.data[idx]}
    def __len__(self):
        return len(self.data)

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #trust_remote_code=True,
        #use_cache=not args.no_gradient_checkpointing,
        #load_in_8bit=True,
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
    data_name = 'data_mcts_random_g3_d8_34386'
    with open(os.path.join('data', data_name, 'rb')) as f:
        data = pickle.load(f)
    
    env = MtgEnv() # TODO

    input_strs = []
    # iterate over trajectories
    for i in range(len(data['states'])):
        # check if winning trajectory
        if data['rewards'][i][0] != 1:
            continue
        for j in range(len(data['states'][i])):
            state = data['states'][i][j]
            possible_moves = data['possible_moves'][i][j]
            action = data['actions'][i][j]
            query = env.state_action_to_query(state, possible_moves)
            response = env.format_action(action)
            input_str = query + response
            input_strs.append(input_str)
    # create dataset
    train_dataset = MtgDataset(input_strs)

    # tokenize, TODO, could change padding, etc
    train_dataset = train_dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True), batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="llama-7b-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )
    # create trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    trainer.train()


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

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)