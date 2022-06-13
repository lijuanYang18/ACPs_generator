#Utilities to prepare the SMILES dataset to be input to the model

import csv
import pandas as pd
import numpy as np
import pickle
import torch

from torch.utils import data
from io import open


char_to_int_dict = {'<start>': 0, 'M': 1, 'V': 2, 'K': 3, 'I': 4, 'F': 5, 'L': 6, 'W': 7, 'T': 8, 'P': 9, 'R': 10, 'Y': 11, 'S': 12, 
                    'H': 13, 'C': 14, 'A': 15, 'G': 16, 'D': 17, 'N': 18, 'E': 19, 'Q': 20, '<eos>': 21, '<unk>': 22, '<pad>': 23}
int_to_char_dict = {'0': '<start>', '1': 'M', '2': 'V', '3': 'K', '4': 'I', '5': 'F', '6': 'L', '7': 'W', '8': 'T', '9': 'P', '10': 'R', 
                    '11': 'Y', '12': 'S', '13': 'H', '14': 'C', '15': 'A', '16': 'G', '17': 'D', '18': 'N', '19': 'E', '20': 'Q', '21': '<eos>', '22': '<unk>', '23': '<pad>'}
#Given a tokenized SMILES dataset split it into X and Y one-hot encoded numpy tensors and return them
def get_petide(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(line.strip())
    return np.array(data)
        

def prepare_dataset(smiles, embed):
	one_hot =  np.zeros((smiles.shape[0], embed), dtype=np.int64)
	for i,smile in enumerate(smiles):
		one_hot[i,0] = char_to_int_dict['<start>']
		for j,c in enumerate(smile):
			one_hot[i,j+1] = char_to_int_dict[c]
		one_hot[i,len(smile)+1] = char_to_int_dict['<eos>']
		one_hot[i,len(smile)+2:] = char_to_int_dict['<pad>']
	return one_hot

def anneal(start_val, end_val, start_iter, end_iter, it):
    if it < start_iter:
        return start_val
    elif it >= end_iter:
        return end_val
    else:
        return start_val + (end_val - start_val) * (it - start_iter) / (end_iter - start_iter)


class Dataset(data.Dataset):
	def __init__(self, smiles_dataset, max_len):
		self.data = get_petide(smiles_dataset)
		self.inputs = prepare_dataset(self.data, max_len)
		self.vocab_size = len(char_to_int_dict)

	def __len__(self):
		return len(self.data)
    
	def idx2sentence(self, idx_list):
		idx_list = idx_list.detach().cpu().numpy().tolist()
		seq = []
		for idx in idx_list:
			seq.append(int_to_char_dict[str(idx)])
		return ''.join(seq)

	def idx2sentences(self, idx_list):
		seq_all = []
		idx_list = idx_list.cpu().numpy().tolist()
		for each in idx_list:
			seq = []
			for idx in each:
				seq.append(int_to_char_dict[str(idx)])
			seq_all.append(''.join(seq))
		return seq_all            

	def __getitem__(self, index):
		X_out = torch.from_numpy(self.inputs[index,:])
		return X_out

class Dataset_classifier(data.Dataset):
	def __init__(self, smiles_dataset, max_len):
		self.data = np.array(smiles_dataset)
		self.data, self.label = self.data[:,0], self.data[:,1]
		self.inputs = prepare_dataset(self.data, max_len)
		self.vocab_size = len(char_to_int_dict)

	def __len__(self):
		return len(self.data)           

	def __getitem__(self, index):
		X_out = torch.from_numpy(self.inputs[index,:])
		Y_out = torch.tensor(int(self.label[index]))
		return X_out, Y_out

