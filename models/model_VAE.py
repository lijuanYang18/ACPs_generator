import numpy as np
import torch
import torch.nn as nn


unk = 22
pad = 23
start = 0
eos = 21
class Maskedwords(nn.Module):
    def __init__(self, p_word_dropout):
        super(Maskedwords, self).__init__()
        self.p = p_word_dropout

    def forward(self, x):
        data = x.clone().detach()
        mask = torch.from_numpy(np.random.binomial(1, p=self.p, size=tuple(data.size())).astype('uint8')).to(x.device)
        mask = mask.bool()
        data[mask] = unk
        return data

class MyEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, z_dim, bidir, n_layers, dropout):
        super(MyEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.bidir = bidir
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, bidirectional=bidir, dropout=dropout, batch_first=True)
        
        if bidir:
            self.mu_layer = nn.Linear(2*hid_dim, z_dim)
            self.logvar_layer = nn.Linear(2*hid_dim, z_dim)
        else:
            self.mu_layer = nn.Linear(hid_dim, z_dim)
            self.logvar_layer = nn.Linear(hid_dim, z_dim)
    def forward(self, x):
        _, hidden = self.gru(x, None)
        if self.bidir:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = hidden.view(-1, hidden.size()[-1])
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar


class MyDecoder(nn.Module):
    def __init__(self, embedding_layer, emb_dim, hid_dim, output_dim, bidir, n_layers, masked_p):
        super(MyDecoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.bidir = bidir
        self.n_layers = n_layers
        self.masked_p = masked_p
        
        self.embedding = embedding_layer
        self.word_dropout = Maskedwords(self.masked_p)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, bidirectional=bidir, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(self.masked_p), nn.Linear(hid_dim, output_dim))
                

    def forward(self, x, z, c):
        [batch_size, seq_lens] = x.size()
        z_vector = torch.cat([z,c], dim = 1)
        inputs = self.embedding(self.word_dropout(x))
        z_vector_expand = z_vector.unsqueeze(1).expand(-1, seq_lens, -1)
        
        inputs = torch.cat([inputs, z_vector_expand], dim=2)
        output, _ = self.gru(inputs, z_vector.unsqueeze(0))
        y = self.fc(output)
        
        return y
        
        
    def samples(self, start_token, z, c, h):
        inputs = self.embedding(start_token)
        inputs = torch.cat([inputs, z, c], 1).unsqueeze(1)
        output, h = self.gru(inputs, h)
        output = output.squeeze(1)
        logits = self.fc(output)        
        return logits, h



class RNN_VAE(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 device,
                 z_dim,
                 c_dim,
                 emb_dim,
                 Encoder_args,
                 Decoder_args
                 ):
        super(RNN_VAE, self).__init__()
        self.MAX_SEQ_LEN = max_seq_len
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.device = torch.device('cuda')
        self.emb_dim = emb_dim
        
        self.word_emb = nn.Embedding(vocab_size, self.emb_dim, pad)
        self.encoder = MyEncoder(emb_dim=self.emb_dim, hid_dim=Encoder_args['h_dim'], z_dim=z_dim, 
                                 bidir=Encoder_args['biGRU'], n_layers=Encoder_args['layers'], dropout = Encoder_args['p_dropout'])
        
        self.decoder = MyDecoder(embedding_layer=self.word_emb, emb_dim=self.emb_dim + z_dim + c_dim, hid_dim=z_dim + c_dim, 
                                 output_dim=self.vocab_size, bidir=False, n_layers=1, masked_p=Decoder_args['p_word_dropout'])


    def sample_z(self, mu, logvar):
        """
        z = mu + std*eps; eps ~ N(0, I)
        """
        eps = torch.randn(mu.size(0), self.z_dim).to(self.device)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, mbsize):
        """
        z ~ p(z) = N(0, I)
        """
        z = torch.randn(mbsize, self.z_dim).to(self.device)
        return z

    def sample_c_prior(self, mbsize):
        """
        Sample c ~ p(c) = Cat([0.5, 0.5])
        """
        c = torch.from_numpy(np.random.multinomial(1, [0.5, 0.5], mbsize).astype('float32')).to(self.device)
        return c


    def forward(self, sequences):

        mbsize = sequences.size(0)
        #encoder: seq to z
        inputs = self.word_emb(sequences)        
        mu, logvar = self.encoder(inputs)
        
        assert mu.size(0) == logvar.size(0) == mbsize 
        z = self.sample_z(mu, logvar)
        c = self.sample_c_prior(mbsize)
        
        #decoder: z to seq
        dec_logits = self.decoder(sequences, z, c)
        
        return (mu, logvar), (z, c), dec_logits    
        
    def sample(self, mbsize, z=None, c=None):  
        if z is None:
            z = self.sample_z_prior(mbsize)  # Sample Z from the prior distribution      
            c = self.sample_c_prior(mbsize)  # onehots
        else:
            z = z
            c = c
        h = torch.cat([z, c], dim=1).unsqueeze(0)
        
        self.eval()
        seqs = []
        finished = torch.zeros(mbsize, dtype=torch.bool).to(self.device)
        prefix = torch.LongTensor(mbsize).to(self.device).fill_(start)
        seqs.append(prefix)
        for i in range(self.MAX_SEQ_LEN):
            logits, h = self.decoder.samples(prefix, z, c, h)
            prefix = torch.distributions.Categorical(logits=logits).sample()
            prefix.masked_fill_(finished, pad)
            finished[prefix == eos] = True  # new EOS reached, mask out in the future.
            seqs.append(prefix)
            
            if finished.sum() == mbsize:
                break
        seqs = torch.stack(seqs, dim=1)
        self.train()
        
        return seqs

