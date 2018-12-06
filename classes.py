from train import *
import copy
import torch
import torch.utils.data
from abc import ABC, abstractmethod


class DatasetClass:
    """Class to create and manage data-sets. In particular it builds a data-set, D0,
    of P random points on the d-sphere with random labels in {-1,+1}.

    When P is too large (> batch_size) that it could give memory error (on GPU) data
    is given in batches when the __iter__ method is called.


    """

    def __init__(self, P, d, main_dir):
        self.P = P
        self.d = d
        self.current = 'D0'
        self.main_dir = main_dir
        self.batch_size = int(2**16)

        self.big = self.P > self.batch_size

    def new(self):
        """Create a new D0 of points on d-sphere and {-1,+1} labels.
        Move to GPU if it can fit (NVIDIA GTX 1080 with 11Gb memory) and saves on file."""
        self.labels = ((torch.empty((self.P, 1)).random_(0, 2) - .5) * 2)
        self.D = torch.empty((self.P, self.d)).normal_()

        if not self.big:
            self.labels = self.labels.cuda()
            self.D = self.D.cuda()

        torch.save(self.D, self.main_dir + '/data/D0.pt')
        torch.save(self.labels, self.main_dir + '/data/labels.pt')

    def build_dataset(self, N):
        """After training a model with total number of parameters N,
        builds new data-set from hidden layers' activations."""

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

        del self.D, self.labels

        return manual_count_params(model)

    def load(self, layer):
        """Load data-set for specified layer in the net."""
        self.D = torch.load(self.main_dir + f'/data/D{layer}.pt')
        self.labels = torch.load(self.main_dir + f'/data/labels.pt')
        self.d = self.D.shape[1]

    def __len__(self):
        """Outputs the number of data-sets."""
        return self.length

    def __iter__(self):
        """Iterator to process batches and move them to GPU"""
        if not self.big:
            yield self.D[:self.P, :self.d], self.labels
        else:
            for i in range(int(self.P / self.batch_size)):
                yield self.D[i*self.batch_size:min(self.P, (i+1)*self.batch_size), :self.d].cuda(), \
                      self.labels[i*self.batch_size:min(self.P, (i+1)*self.batch_size), :].cuda()


class ModelClass(ABC):
    """Abstract class for networks models."""
    def __init__(self):
        self.N_list = []
        self.layers = []
        self.model = None

    @abstractmethod
    def define_layers(self):
        raise NotImplemented

    @abstractmethod
    def lr(self):
        raise NotImplemented

    def reduce(self, steps=1):
        """Reduce network dimension.
        Is used to find the jamming starting from the SAT phase, r increases."""
        self.h -= steps

    def new(self):
        """Initialize new network and move it to GPU"""
        self.define_layers()
        self.model = nn.Sequential(*self.layers)
        self.model.cuda()
        self.model = orthogonal_init(self.model)

        # Re-count N
        self.count_params()

    def save(self, main_dir):
        """Save model in main_dir/models/ ."""
        with open(f'{main_dir}/models/model_N{self.N}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def count_params(self):
        """Count the number of parameters of the current network and add it
        to a list to keep track of the size of the networks previously used."""
        self.N = 0
        for name, param in self.model.named_parameters():
            self.N += param.numel()
        self.N_list.append(self.N)

    def print_structure(self):
        """Print the structure of the network with layer: number of units |

            e.g. L1: 10 | L2: 20 | L3: 30 | ...

        """
        for name, param in self.model.named_parameters():
            if '0.bias' in name:
                print(f'L{int(name[0])+1}: {param.numel()} | ', end="")

    def __call__(self, X):
        """If the instance is called it feeds the input to the network."""
        return self.model(X)


class Triangular(ModelClass):
    """Define a triangular neural network, i.e. the hidden layers' dimension
    decreases with depth. Inherits from the abstract ModelClass."""
    def __init__(self, P, r, d):
        super().__init__()

        # Input dimension
        self.d = d

        # Compute the expected first layer size give set total number of params
        self.h = find_h_triangular_net(P / r)

        # Define number of units difference between successive layers
        self.delta_h = 10

    def lr(self):
        """Return the learning rate for the network depending on the size"""
        return 0.1 / self.h

    def layers_sizes(self):
        """Define layers sizes give the size of the first and the steps."""
        return iter([self.delta_h*l for l in range(int(self.h/self.delta_h)-1)])

    def define_layers(self):
        """Define the network structure with PyTorch nn.sequential modules."""
        if self.d != 0:
            # If we have a fixed input size we use it do define the first layer
            self.layers = [nn.Sequential(nn.Linear(self.d, self.h),
                                         nn.ReLU(), )]  # nn.BatchNorm1d(self.h, affine=False))]
        else:
            self.layers = [nn.Sequential(nn.Linear(self.h, self.h),
                                         nn.ReLU(), )]

        l = 0
        for l in self.layers_sizes():
            self.layers.append(nn.Sequential(nn.Linear(self.h - l, self.h - l - self.delta_h),
                                             nn.ReLU(), ))  # nn.BatchNorm1d( self.h - l - self.delta_h, affine=False)))
        self.layers.append(nn.Sequential(nn.Linear(self.h - l - self.delta_h, 1), nn.ReLU()))


class Rectangular(ModelClass):
    """Define a triangular neural network, i.e. the hidden layers' dimension
    is constant with depth. Inherits from the abstract ModelClass."""
    def __init__(self, P, r, d, L):
        super().__init__()

        self.L = L
        self.h = find_h_rectangular_net(d, L, P, r)

        if d != 0:
            # If input dimension d is fixed we use it for the first layer
            self.d = d
        else:
            self.d = self.h

    def lr(self):
        """Return the learning rate for the network depending on the size"""
        return .1 / self.h ** 1.5

    def define_layers(self):
        """Define the network structure with PyTorch nn.sequential modules."""
        self.layers = [nn.Sequential(nn.Linear(self.d, self.h), nn.ReLU(), )]
        for l in range(1, self.L):
            self.layers.append(nn.Sequential(nn.Linear(self.h, self.h), nn.ReLU(), ))

        self.layers.append(nn.Linear(self.h, 1))


class ResultsClass:
    """Store save and load results for each experiment / trained layer. """

    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.results = {}

        # General structure for a dictionary of relevant quantities to save
        self.dict = {'loss': [],
                     'N': [],
                     'N_delta': [],
                     'r': [],
                     'gen_error': []}

        # Initialize only entry for initial dataset D0
        self.results['D0'] = copy.deepcopy(self.dict)

    def new_results(self, name):
        """Instanciate new results entry with key = name and return it"""
        self.results[name] = copy.deepcopy(self.dict)
        return self.results[name]

    def save(self):
        """Save results in main_dir/results.pkl ."""
        pickle_save(self.results, 'results', self.main_dir)

    def load(self):
        """Load results from main_dir/results.pkl ."""
        self.results = pickle_load('results', self.main_dir)