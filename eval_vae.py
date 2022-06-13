import torch
import numpy as np
from params import paramters
from models.model_VAE import RNN_VAE
from data_processing.utils import Dataset
from torch.utils import data
import torch.nn.functional as F
import math
from numpy import mean




dataset = Dataset('./data_processing/data/test.txt', 35)
device = 'cuda:0'
model = RNN_VAE(vocab_size=paramters.vocab_size, max_seq_len=25, device=device, 
                **paramters.model).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
evaluate = True
if evaluate:
    state = torch.load('./output/model_epoch_80.pt', map_location=torch.device(device))
    model.load_state_dict(state)
batches = data.DataLoader(dataset, batch_size=40, shuffle=False, drop_last=True, num_workers=0)



def get_bleu(prediction, target):    
    target = target[1:target.index(21)]
    if 21 in prediction:
        prediction = prediction[:prediction.index(21)]
    elif 23 in prediction:
        prediction = prediction[:prediction.index(23)]
    else:
        prediction = prediction[:]
    
    n_1 = sum([each in target for each in prediction])/len(prediction)
    
    n_2_p = [(prediction[i], prediction[i+1]) for i in range(len(prediction)-1)]
    n_2_t = [(target[i], target[i+1]) for i in range(len(target)-1)]
    n_3_p = [(prediction[i], prediction[i+1],prediction[i+2]) for i in range(len(prediction)-2)]
    n_3_t = [(target[i], target[i+1], target[i+2]) for i in range(len(target)-2)]
    n_4_p = [(prediction[i], prediction[i+1],prediction[i+2]) for i in range(len(prediction)-2)]
    n_4_t = [(target[i], target[i+1], target[i+2]) for i in range(len(target)-2)]
    
    n_2 = sum([each in n_2_t for each in n_2_p])/len(n_2_p)
    n_3 = sum([each in n_3_t for each in n_3_p])/len(n_3_p)
    n_4 = sum([each in n_4_t for each in n_4_p])/len(n_4_p)
    
    if len(prediction) > len(target):
        BP = 1
    else:
        BP = math.exp(1-len(target)/len(prediction))
    #return BP*math.exp(0.25*(math.log(n_1) + math.log(n_2) + math.log(n_3) + math.log(n_4)))
    return BP*0.25*(n_1 + n_2 + n_3 + n_4)

def get_ppl(target,prediction):
    target = target[1:target.index(21)]
    ppl = 1
    count = 0
    for i in range(len(target)):
        ppl = ppl*prediction[i][target[i]]
        count += 1
    return math.pow(1/ppl, 1/count)

######################     Evaluating vae PPL BLEU    #################################

inputs_all_ = []
output_all_bleu = []
output_all_ppl = []
with torch.no_grad():
    for idx, inputs in enumerate(batches):
        inputs_all_.append(inputs.numpy().tolist())
        inputs = inputs.to(device)
        (mu, logvar), (z, c), dec_logits = model(inputs)
        output_all_bleu.append(torch.argmax(F.softmax(dec_logits, dim=-1),dim=-1).detach().cpu().numpy().tolist())
        output_all_ppl.append(F.softmax(dec_logits, dim=-1).detach().cpu().numpy().tolist())



    
bleu = []
for i in range(len(inputs_all_)):
    for j in range(len(inputs_all_[i])):
        try:
            a = get_bleu(output_all_bleu[i][j], inputs_all_[i][j])
            bleu.append(a)
        except:
            pass
print('The value of BLEU is %s'%str(mean(bleu)))


ppl = []
for i in range(len(inputs_all_)):
    for j in range(len(inputs_all_[i])):
        ppl.append(get_ppl(inputs_all_[i][j], output_all_ppl[i][j]))
print('The value of PPL is %s'%str(mean(ppl)))












