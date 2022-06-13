import numpy as np
import torch
from models.model_VAE import RNN_VAE
from models.classifier import  Model_classifier
from params import paramters
from data_processing.utils import prepare_dataset
from data_processing.utils import Dataset
from tqdm import tqdm
from threading import Thread
import json

device = 'cuda:0'
dataset = Dataset('./data_processing/data/train.txt', 35)
model_generator = RNN_VAE(vocab_size=paramters.vocab_size, max_seq_len=20, device=device, **paramters.model).to(device)
model_generator.load_state_dict(torch.load('./output/wae/model_epoch_80.pt', map_location=torch.device(device)))


model_classifier = Model_classifier(device, './output/wae/model_epoch_80.pt').to(device)
state = torch.load('./output/classifier/classifier_lung.tar', map_location=torch.device(device))
model_classifier.load_state_dict(state)

def fit_fun(z):
    z = torch.from_numpy(np.array(z).astype('float32')).to(device)
    pop = z.size()[0]
    c = torch.from_numpy(np.array([[1., 0.]]*pop).astype('float32')).to(device)
    with torch.no_grad():
        model_generator.eval()
        seq = model_generator.sample(pop, z=z, c=c)

    seq = dataset.idx2sentences(seq)
    result = []
    for i in range(len(seq)):        
        if '<eos>' in seq[i]:
            one = [seq[i].split('<start>')[1].split('<eos>')[0]]
        else:
            one = [seq[i].split('<start>')[1]]        
        if '<pad>' in seq[i] or '<unk>' in seq[i]:
            result.append(['0',one[0]])
        else:
            data = prepare_dataset(np.array(one), 35)
            inputs = torch.from_numpy(data)
            with torch.no_grad():
                model_classifier.eval()
                output = model_classifier(inputs.to(device))
                output = output.cpu().detach().numpy().reshape(-1).tolist()[1]
            result.append([str(output),one[0]])
    result = np.array(result)
    return result

class PSO():
    def __init__(self, n_dim=None, pop=40, max_vel=1, max_iter=150, w=1, c1=2, c2=2):

        self.w = w  # inertia
        self.c1 = c1
        self.c2 = c2
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.max_vel = max_vel

        self.is_feasible = np.array([True] * pop)
        self.X = torch.randn(pop, n_dim).numpy()
        self.V = np.random.uniform(low=-self.max_vel, high=self.max_vel, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[0,'aaa']] * pop)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = 0  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        self.collected = []

    def check_constraint(self, x):
        # gather all unequal constraint functions
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.c1 * r1 * (self.pbest_x - self.X) + \
                 self.c2 * r2 * (self.gbest_x - self.X)
        self.V = np.clip(self.V, -self.max_vel, self.max_vel)

    def update_X(self):
        self.X = self.X + self.V


    def cal_y(self):
        # calculate y for every x in X  
        self.Y = fit_fun(self.X)
        #print(self.Y)
        
        return np.array(self.Y) #np.array([[0,'aaa']] * pop)

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y[:,0].reshape(-1,1) < self.Y[:,0].reshape(-1,1)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_max = self.pbest_y.argmax()
        idx_max = np.array([eval(each) for each in self.pbest_y[:,0].tolist()]).argmax()
        if self.gbest_y < eval(self.pbest_y[idx_max][0]):
            self.gbest_x = self.X[idx_max, :].copy()
            self.gbest_y = eval(self.pbest_y[idx_max][0])


    def run(self, max_iter, N):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        pbar = tqdm(range(self.max_iter))
        for iter_num in pbar:
            self.update_V()
            self.update_X()
            self.cal_y()
            #print(self.Y[:,0])
            coll = self.Y[:,0] >= np.array(['0.5']*self.pop)
            self.collected = self.collected + self.Y[:,1][coll].tolist()
            self.collected = list(set(self.collected))
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.collected, self.gbest_y_hist

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.collected, _ = self.func(*self.args)

    def get_result(self):
        try:
            return self.collected
        except Exception:
            return None

max_pools = 5
threads = []
for i in range(max_pools):
    pso = PSO(n_dim=100, pop=40, max_vel=1, max_iter=500, w=1, c1=2, c2=2)
    thread = MyThread(pso.run, args=(500,20))
    thread.start()   
    threads.append(thread)

for thread in threads:
    thread.join()

result_all = []
for thread in threads:
    res = thread.get_result()
    result_all += res


with open('./output/PSO_samples/output.json','w') as f:
    json.dump(list(set(result_all)),f)