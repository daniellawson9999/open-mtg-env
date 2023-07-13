# OpenMTG - Env

This is a fork of Hlynur Davíð Hlynsson's [repository](https://github.com/hlynurd/open-mtg/issues), "An experimental framework for writing, testing and evaluating agents for the card game Magic: The Gathering."

Modifications are being made to support gym-style API, and natural language-based reinforcement learning with transformers to play a simplified form of mtg. Can create gym-style environment: https://github.com/daniellawson9999/open-mtg-env/blob/master/open_mtg_env/env.py. 


Example usage can be found in env.py. The current basic procedure is collecting "expert" data with MCTS, finetune a pretrained LM, can see an example of this in: https://github.com/daniellawson9999/open-mtg-env/blob/master/lm_train_test.py
