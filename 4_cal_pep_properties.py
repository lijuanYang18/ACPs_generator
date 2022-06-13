import json
import os
import json
import numpy as np
from peptide_evals import PeptideEvaluator
import argparse


file = './output/PSO_samples/output.json'
cluster_threshold = 18

result = []
with open(file) as f:
    result = json.load(f)



petide = PeptideEvaluator()

def cluster(seq_list, seq):
    simi = []
    for each in seq_list:
        _, a = petide.similarity(each, seq)
        simi.append(a)
    short_1 = []
    for i in range(len(simi)):
        if simi[i] <= cluster_threshold:
            short_1.append(seq_list[i])
    return short_1


for i in range(0,len(result)):
    try:
        result = cluster(result, result[i])
        print(i, len(result))
    except:
        break

def get_charge(seq_list):
    charge_list = []
    hydrophobicity = []
    hydrophobicity_moment = []
    for each in seq_list:
        charge_list.append(petide.calculate_charge(each))
        hy = petide.assign_hydrophobicity(each)
        if len(hy) > 0:
            hy_m = petide.calculate_moment(hy)
            hy = sum(hy)/len(hy)
            hydrophobicity.append(hy)
            hydrophobicity_moment.append(hy_m)
        else:
            hydrophobicity.append('None')
            hydrophobicity_moment.append('None')
            
    return charge_list, hydrophobicity, hydrophobicity_moment

charge_list, hydrophobicity, hydrophobicity_moment = get_charge(result)

with open('./output/PSO_samples/clustered_ACPs.csv','w') as f:
    f.write('Name, Sequences, Net charge, Hydrophobicity, Hydrophobicity_moment\n')
    for i in range(len(result)):
        f.write('Peptide_%d'%i + ',' + result[i] +  ',' + str(charge_list[i]) + ',' 
                + str(round(hydrophobicity[i],2)) + ',' + str(round(hydrophobicity_moment[i],2)) + '\n')
    




