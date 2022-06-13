# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:59:04 2021

@author: Windows10
"""

class Parameters():
    def __init__(self):
        self.model = {
                    'z_dim': 100,
                    'c_dim': 2,
                    'emb_dim': 150,
                    'Encoder_args':{'h_dim': 80, 'biGRU': True, 'layers': 1, 'p_dropout': 0.0},
                    'Decoder_args':{'p_word_dropout': 0.3, 'biGRU': False, 'layers': 1}
                    }
        self.vae = {'batch_size': 32,
                     'lr': 0.001,
                     's_iter': 0,
                     'n_iter': 200000,
                     'beta': {'start': {'val': 1.0, 'iter': 0},
                      'end': {'val': 2.0, 'iter': 10000}},
                     'lambda_logvar_L1': 0.0,
                     'lambda_logvar_KL': 0.001,
                     'z_regu_loss': 'mmdrf',
                     'cheaplog_every': 500,
                     'expsvlog_every': 20000,
                     'clip_grad': 5.0,
                     'chkpt_path': 'output\\default\\model_{}.pt',
                     'gen_samples_path': 'output\\default\\vae_gen.txt',
                     'eval_path': 'output\\default\\vae_eval.txt',
                     'fasta_gen_samples_path': 'output\\default\\vae_gen.fasta'
                    }
        self.sample_size = 2000
        self.max_seq_len = 25
        self.vocab_size = 24

paramters = Parameters()