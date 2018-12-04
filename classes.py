from alice import *
from train import *
import copy

import torch
import torch.utils.data

class DatasetClass(torch.utils.data.Dataset):
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
            print(param.shape)
            d = param.shape[1]
            break

        D_list = [self.D[:,:d]]

        for layer in model:
            print(layer)
            D_list.append(layer(D_list[-1]))

        self.length = len(D_list)

        for layer in range(1, len(D_list)):
            torch.save(D_list[layer].detach(), self.main_dir + f'/data/D{layer}.pt')

        del self.D, model

    def load(self, layer):
        self.D = torch.load(self.main_dir + f'/data/D{layer}.pt')
        self.d = self.D.shape[1]

    def __len__(self):
        return self.length

    def __iter__(self):
        """Iterator to process batches and move them to GPU"""
        if not self.big:
            yield self.D[:,:self.d], self.labels
        else:
            for i in range(int(self.P / self.batch_size)):
                yield self.D[i*self.batch_size:(i+1)*self.batch_size, :self.d].cuda(), self.labels[i*self.batch_size:(i+1)*self.batch_size, :].cuda()


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


class Experiment:

    def __init__(self, P, datatype, networktype, r_min,
                 L = 0, new_data=True):

        self.P = P
        self.L = L
        self.datatype = datatype

        if 'ria' in networktype:
            self.networktype = 'Triangular'
        else:
            self.networktype = 'Rectangular'

        self.r_min = r_min

        self.main_dir = f'{networktype}Net_{datatype}L{L:02}P{P:05}r{r_min:0.1f}'
        make_dir(self.main_dir)

        if 'ria' in networktype:
            d_max = find_h_triangular_net(P/r_min) + 1
        else:
            d_max = find_h(0, self.L, self.P, self.r_min) + 1

        self.dataset = DatasetClass(P, d_max, self.main_dir)

        if new_data:
            self.dataset.new()

        self.results = ResultsClass(self.main_dir)

    def save(self):
        del self.dataset.D, self.dataset.labels
        pickle_save(self.__dict__, 'experiment', self.main_dir)

    def load(self):
        self.__dict__ = pickle_load('experiment', self.main_dir)

    def find_transition(self,r_min,
                        P=0,
                        L=0,
                        layer_to_train=0,
                        jamming_margin=2,
                        gen_error_flag=False):

        # For rectangular net define L
        if L == 0:
            L = self.L - layer_to_train

        print(f'\n\n>>> FINDING TRANSITION FOR D{layer_to_train} WITH {L} LAYERS <<<')

        self.dataset.load(layer_to_train)

        results_label = f'D{layer_to_train}'

        # If P is varying from initial, keep track in results
        if P:
            results_label += f'P{P}'
        else:
            P = self.P
        results_dict = self.results.new_results(results_label)

        if layer_to_train == 0:
            d = 0
        else:
            d = self.dataset.D.shape[1]

        if 'ria' in self.networktype:
            model = Triangular(P, r_min, d)
        else:
            model = Rectangular(P, r_min, d, L)

        UNSATFlag = 0

        while UNSATFlag < jamming_margin:

            model.new()

            if d == 0:
                self.dataset.d = model.h

            r = P / model.N
            print(f'\n{self.networktype} Net - ', end="")
            model.print_structure()
            print(f'\n  - N = {model.N}, P = {P}, r = {r:0.4f}')

            loss, N_delta, model, gen_error = train_model(model,
                                                          self.dataset,
                                                          learning_rate= model.lr(),
                                                          epochs_number=int(1e6),
                                                          gen_error_flag=gen_error_flag)

            plt.savefig(f'{self.main_dir}/figures/loss/L{L}N{model.N}P{self.P}.png', format='png')
            plt.close('all')

            # Continue increasing r until it finds transition. Take 3 points after and stop
            if N_delta > 0.0021 * P:
                UNSATFlag += 1
            else:
                UNSATFlag = 0

            if not d:
                model.save(self.main_dir)

            if gen_error_flag:
                results_dict['gen_error'].append(gen_error)

            results_dict['loss'].append(loss)
            results_dict['N_delta'].append(N_delta)
            results_dict['N'].append(model.N)
            results_dict['r'].append(r)

            if not d and self.datatype == 'ov':
                # Dont need to find jamming, no jamming_margin needed
                jamming_margin = -1
                break

            model.reduce(steps=1)

        self.results.save()

        # If training the first layer construct dataset from activations
        if d == 0:
            self.dataset.build_dataset(model.N_list[-jamming_margin-1])


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
        print(self.h, self.l_size)

    def reduce(self, steps=1):
        self.h -= steps
        self.l_size = self.layers_sizes()

    def lr(self):
        return 0.5 * self.h ** 1.5

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
