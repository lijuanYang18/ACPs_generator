import sys
import torch
import torch.nn as nn
import torch.optim as optim
import models.losses as losses
from torch.utils import data
from params import paramters
from models.model_VAE import RNN_VAE
from data_processing.utils import Dataset, anneal
import os

def train(model_name, model, loss_fun, train_batches, optimizer, device, it):    
    model.train()
    
    loss_all = 0.0
    recon_mean = 0.0
    kl_mean = 0.0
    mmd_mean = 0.0
    
    for idx, inputs in enumerate(train_batches):    
        inputs = inputs.to(device)
        (z_mu, z_logvar), (z, c), dec_logits = model(inputs)
            
        
        recon_loss = losses.recon_dec(inputs, dec_logits)
        kl_loss = losses.kl_gaussianprior(z_mu, z_logvar)
        wae_mmd_loss = losses.wae_mmd_gaussianprior(z, method='full_kernel', sigma = 7)
        wae_mmdrf_loss = losses.wae_mmd_gaussianprior(z, method='rf', sigma = 7)
        z_logvar_KL_penalty = losses.kl_gaussian_sharedmu(z_mu, z_logvar)
        
        if model_name == 'wae':
            beta = anneal(start_val=1.0, end_val=2.0, start_iter=0, end_iter=40000, it=it)
            loss = recon_loss + beta*wae_mmdrf_loss + 0.001*z_logvar_KL_penalty
        else:
            beta = 0.03
            loss = recon_loss + beta*kl_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()       
        it += 1
        loss_all += loss.item()
        recon_mean += recon_loss.item()
        kl_mean += kl_loss.item()
        mmd_mean += wae_mmdrf_loss.item()
        
        
    return loss_all/len(train_batches), recon_mean/len(train_batches), kl_mean/len(train_batches), mmd_mean/len(train_batches), it


# DATA
model_name = 'wae' #wae
batch_size = 32
lr = 0.001
epochs = 100
save_path = './output/wae/'
dataset = Dataset('./data_processing/data/train.txt', 35)
train_batches = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
device = 'cuda:0'
model = RNN_VAE(vocab_size=paramters.vocab_size, max_seq_len=25, device=device, 
                **paramters.model).to(device)
loss_fun = losses
optimizer = optim.Adam(model.parameters(), lr=lr)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

it = 0
print('Training base wae ...')
for epoch in range(1, epochs+1):

    loss, loss_recon, loss_kl, loss_mmd, it =  train(model_name, model, loss_fun, train_batches, optimizer, device, it)
    log_sent = model.sample(1)  
    with open('training_process_wae','a') as f:
        f.write(
            'Epoch {}. loss_vae: {:.4f}; loss_recon: {:.4f}; loss_kl: {:.4f}; loss_mmd: {:.4f}; \n'
                .format(epoch, loss, loss_recon, loss_kl, loss_mmd))                      
        f.write('Sample (cat T=1.0): "{}" \n'.format(dataset.idx2sentence(log_sent.squeeze())))
    
    torch.save(model.state_dict(), os.path.join(save_path, 'model_epoch_%d.pt'%epoch)) 
    print(epoch, it)
    