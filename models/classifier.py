import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_VAE import RNN_VAE
from params import paramters
import numpy as np
from itertools import chain


class Classifier(nn.Module):

    def __init__(self,
                 emb_dim,
                 channel,
                 dropout):
        super(Classifier, self).__init__()
        self.conv = nn.ModuleList([nn.Conv2d(1, channel, (3, emb_dim)),
                                   nn.Conv2d(1, channel, (4, emb_dim)),
                                   nn.Conv2d(1, channel, (5, emb_dim))])
        self.fc = nn.Linear(channel * len(self.conv), 2)
        self.drop = nn.Dropout(dropout)


    def forward(self, x):

        x = x.unsqueeze(1)  # batch_size x 1 x max_seq_len x emb_dim 64*1*35*150

        conv_out = []
        for i in range(len(self.conv)):
            one_out = self.conv[i](x)
            one_out = F.relu(one_out).squeeze(3)
            pool_out = F.max_pool1d(one_out, one_out.size(2)).squeeze(2)
            conv_out.append(pool_out)            
            
        conv_out = torch.cat(conv_out, dim=1) 
        drop_out = self.drop(conv_out)
        fc_out = self.fc(drop_out)

        return fc_out



class Model_classifier(nn.Module):
    def __init__(self, device, generator_path):
        super(Model_classifier, self).__init__()
        model = RNN_VAE(vocab_size=paramters.vocab_size, max_seq_len=25, device=device, 
                **paramters.model).to(device)
        state = torch.load(generator_path, map_location=torch.device(device))
        model.load_state_dict(state)
        model.word_emb.weight.requires_grad = False
        
        self.embedding = model.word_emb
        self.classifier = Classifier(150, 100, 0.3)
    
    def classifier_params(self):
        params = [self.embedding.parameters(), self.classifier.parameters()]
        return filter(lambda p: p.requires_grad, chain(*params))
        
    
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        out = self.classifier(inputs)
        out = F.softmax(out, dim = 1)
        return out

class FocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)-0.75
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        #logit = F.softmax(input, dim=1)
        logit = input
 
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss  
