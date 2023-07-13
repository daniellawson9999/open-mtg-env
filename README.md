# OpenMTG - Env

This is a fork of Hlynur Davíð Hlynsson's [repository](https://github.com/hlynurd/open-mtg/issues), "An experimental framework for writing, testing and evaluating agents for the card game Magic: The Gathering."

Modifications are being made to support natural language based reinforcement learning or training transformers to simplified mtg. Example usage can be found in env.py. Current basic procedure is collecting "expert" data with MCTS, finetune a pretrained LM, can see an example of this in: https://github.com/daniellawson9999/open-mtg-env/blob/master/lm_train_test.py
