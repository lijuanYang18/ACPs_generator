import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from data_processing.utils import Dataset_classifier
from models.classifier import FocalLoss, Model_classifier
from torch import optim


device = 'cuda:0'
##data
with open('./data_processing/data/ACPs_lung_cancer.json') as f:
    data = json.load(f)
train_dataset = Dataset_classifier(data['train'], 35)
test_dataset = Dataset_classifier(data['test'], 35)
train_batches = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=1)
test_batches = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=1)


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import sklearn.metrics

def auc_and_pr(prediction,label):

    auc = roc_auc_score(label, prediction)
    precision, recall, thresholds = precision_recall_curve(label, prediction)
    area = sklearn.metrics.auc(recall, precision)
    
    return auc,area


def train(model_class, loss_fun, trainset, optimizer, device='cuda:0'):
    total_loss = 0.
    model_class.train()
    for i, (x, y) in enumerate(trainset):     
        optimizer.zero_grad()
        x = x.to(device)
        y_ = y.to(device)
        output = model_class(x)
        loss = loss_fun(output, y_)
        loss.backward()
        nn.utils.clip_grad_norm_(model_class.parameters(), 3)
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(trainset)

def test(model_class, loss_fun, testset, device='cuda:0'):
    outputs = []
    targets = []
    test_loss = 0
    with torch.no_grad():
        model_class.eval()        
        for x_test, y_test in testset:
            y_test = y_test.to(device)
            x_test = x_test.to(device)
            y_hat = model_class(x_test)
            test_loss += loss_fun(y_hat, y_test).item()
            outputs.append(y_hat[:,1].cpu().numpy().reshape(-1))
            targets.append(y_test.cpu().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    test_loss /= len(testset.dataset)
    
    return auc_and_pr(outputs, targets), test_loss

model_class = Model_classifier(device, './output/wae/model_epoch_80.pt').to(device)
loss_fun = nn.CrossEntropyLoss()
#loss_fun = FocalLoss(2)
optimizer = optim.Adam(model_class.classifier_params(), lr=0.0001)

for epoch in range(1, 100+1):
    print(epoch)
    train_loss = train(model_class, loss_fun, train_batches, optimizer, 'cuda:0')
    (train_auc, train_aupr), train_losses = test(model_class, loss_fun, train_batches, 'cuda:0')
    (test_auc, test_aupr), test_losses = test(model_class, loss_fun, test_batches, 'cuda:0')
    
    with open('train_classifier_process__','a') as f:
        f.write('Epoch = ' + '\t' + str(epoch) + '\t' + 'Train_auc' + '\t' + str(round(train_auc,3)) + '\t' 'Train_aupr' + '\t' + str(round(train_aupr,3)) + '\t' + 'Train_loss' + '\t' + str(round(train_losses, 4)) + '\n')
        f.write('Epoch = ' + '\t' + str(epoch) + '\t' + 'Test_auc' + '\t' + str(round(test_auc,3)) + '\t' 'Test_aupr' + '\t' + str(round(test_aupr,3)) + '\t' + 'Test_loss' + '\t' + str(round(test_losses, 4)) + '\n')

    if epoch % 5 == 0:
        torch.save(model_class.state_dict(), './output/classifier/' + 'classifier_lung_%s.tar'%str(epoch))