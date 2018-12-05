from train import *
import copy

import torch
import torch.utils.data


class DatasetClass:
    """Characterizes a dataset for PyTorch"""

    def __init__(self, P, d, main_dir):
        self.P = P
        self.d = d
        self.current = 'D0'
        self.main_dir = main_dir
        self.batch_size = int(2**21)

        self.big = self.P > self.batch_size

    def new(self):
        self.labels = ((torch.empty((self.P, 1)).random_(0, 2) - .5) * 2)
        self.D = torch.empty((self.P, self.d)).normal_()

        if not self.big:
            self.labels = self.labels.cuda()
            self.D = self.D.cuda()

        torch.save(self.D, self.main_dir + '/data/D0.pt')
        torch.save(self.labels, self.main_dir + '/data/labels.pt')

    def build_dataset(self, N):

        model = load_model(N, self.main_dir)

        if self.big:
            model.cpu()

        # Read dimension of first layer
        for name, param in model.named_parameters():
            d = param.shape[1]
            break

        D_list = [self.D[:,:d]]

        for layer in model:
            D_list.append(layer(D_list[-1]))

        self.length = len(D_list)

        for layer in range(1, len(D_list)):
            torch.save(D_list[layer].detach(), self.main_dir + f'/data/D{layer}.pt')

        del self.D, self.labels, model

    def load(self, layer):
        self.D = torch.load(self.main_dir + f'/data/D{layer}.pt')
        self.labels = torch.load(self.main_dir + f'/data/labels.pt')
        self.d = self.D.shape[1]

    def __len__(self):
        return self.length

    def __iter__(self):
        """Iterator to process batches and move them to GPU"""
        if not self.big:
            yield self.D[:self.P, :self.d], self.labels
        else:
            for i in range(int(self.P / self.batch_size)):
                yield self.D[i*self.batch_size:min(self.P, (i+1)*self.batch_size), :self.d].cuda(), \
                      self.labels[i*self.batch_size:min(self.P, (i+1)*self.batch_size), :].cuda()


class ModelClass:
    """Example class for models"""

    def new(self):

        self.define_layers()
        self.model = nn.Sequential(*self.layers)
        self.model.cuda()
        self.model = orthogonal_init(self.model)

        # Re-count N
        self.count_params()

    def reduce(self, steps=1):
        raise NotImplemented

    def lr(self):
        raise NotImplemented

    def save(self, main_dir):
        with open(f'{main_dir}/models/model_N{self.N}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def count_params(self):
        self.N = 0
        for name, param in self.model.named_parameters():
            self.N += param.numel()
        self.N_list.append(self.N)

    def print_structure(self):
        for name, param in self.model.named_parameters():
            if '0.bias' in name:
                print(f'L{int(name[0])+1}: {param.numel()} | ', end="")

    def __call__(self, X):
        return self.model(X)


class Triangular(ModelClass):
    def __init__(self, P, r, d):
        super().__init__()

        self.N_list = []

        N = P / r
        self.d = d
        self.h = find_h_triangular_net(N)

        self.reduce_step = 0
        self.delta_h = 20
        self.l_size = self.layers_sizes()

    def reduce(self, steps=1):
        self.h -= steps
        self.l_size = self.layers_sizes()

    def lr(self):
        return 0.1 / self.h

    def layers_sizes(self):
        return np.asarray([self.delta_h*l for l in range(int(self.h/self.delta_h)-1)])

    def define_layers(self):
        if self.d != 0:
            self.layers = [nn.Sequential(nn.Linear(self.d, self.h),
                                         nn.ReLU(), )]  # nn.BatchNorm1d(self.h, affine=False))]
        else:
            self.layers = []
        for l in self.l_size:
            self.layers.append(nn.Sequential(nn.Linear(self.h - l, self.h - l - self.delta_h),
                                             nn.ReLU(), ))  # nn.BatchNorm1d( self.h - l - self.delta_h, affine=False)))
        self.layers.append(nn.Sequential(nn.Linear(self.h - l - self.delta_h, 1), nn.ReLU()))


class Rectangular(ModelClass):
    def __init__(self, P, r, d, L):
        super().__init__()
        self.N_list = []
        self.L = L
        self.h = find_h(d, L, P, r)

        if d != 0:
            self.d = d
        else:
            self.d = self.h

    def lr(self):
        return .1 / self.h ** 1.5

    def define_layers(self):
        self.layers = [nn.Sequential(nn.Linear(self.d, self.h), nn.ReLU(), )]
        for l in range(1, self.L):
            self.layers.append(nn.Sequential(nn.Linear(self.h, self.h), nn.ReLU(), ))

        self.layers.append(nn.Linear(self.h, 1))

    def reduce(self, steps=1):
        self.h -= steps


class ResultsClass:
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.results = {}
        self.dict = {'loss': [],
                     'N': [],
                     'N_delta': [],
                     'r': [],
                     'gen_error': []}

        self.results['D0'] = copy.deepcopy(self.dict)

    def new_results(self, name):
        """Instanciate new results entry with key = name and return it"""
        self.results[name] = copy.deepcopy(self.dict)
        return self.results[name]

    def save(self):
        pickle_save(self.results, 'results', self.main_dir)

    def load(self):
        self.results = pickle_load('results', self.main_dir)