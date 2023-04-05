import pickle
import os

data_name = 'data_mcts_random_g3_d8_34386.pkl'
data_path = os.path.join('data', data_name)
with open(data_path, 'rb') as f:
    dataset = pickle.load(f)
import pdb; pdb.set_trace()