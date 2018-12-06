import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn-muted')  # muted

import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
import pickle
import sklearn.manifold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from time import gmtime, strftime, time
from functools import update_wrapper


print('Starting... @ ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))


def count_params_rectangular(d, h, L):
    """ Count parameters in a fully-connected rectangular network"""
    return h * ((L - 1) * (h + 1) + d + 2) + 1


def manual_count_params(model):
    """Takes as input a torch model.
        Returns list where each element list[i] is the number of parameters present
    in the residual network, i.e. the net constructed starting from layer i on."""

    params_per_layer = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            params_per_layer.append(param.numel())
        else:
            params_per_layer[-1] += param.numel()
    return np.cumsum(params_per_layer[::-1])[::-1]


def find_h_rectangular_net(d, L, P, r, exact=True):
    """Find the layers size for given set of params and a rectangular net.
    If exact == False, return an approximate value for h."""
    if exact:
        if L == 0:
            return
        if L == 1:
            return int(P / (r * (d + 2)))
        if d == 0:
            return int(np.sqrt(P / (L * r)))
        else:
            return int((-(L + d + 1) + np.sqrt((L + d + 1) ** 2 - 4 * (L - 1) * (1 - P / r))) / (2 * (L - 1)))
    else:
        return int((-d + np.sqrt(d ** 2 + 4 * P / r * (L - 1))) / (2 * (L - 1)))


def find_h_triangular_net(N):
    """Approximately find the size of the first layer for a triangular net for
    a given number of paramentes"""
    return int(4 * np.cbrt(N/2)) + 1


def make_dir(main_dir_name):
    """Makes directories tree for each experiment:

        main_dir/
                /figures/
                        /loss
                /models
                /data
    """

    for sub_dir in ['/figures/loss', '/models', '/data']:
        try:
            os.makedirs(main_dir_name + sub_dir)
            print("Directory ", main_dir_name + sub_dir, " Created ")
        except FileExistsError:
            print("Directory ", main_dir_name + sub_dir, " already exists")


def pickle_save(dictionary, name, directory):
    """Save a dictionary name.pkl in /directory. """
    with open(directory + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f)


def pickle_load(name, directory):
    """Load directory/name.pkl ."""
    with open(directory + '/' + name + '.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary


def load_model(N, main_dir):
    """ Load the model with total number of parameter == N from
            main_dir/models/..
    """
    with open(f'{main_dir}/models/model_N{N}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def hinge_loss(y_true, y_pred):
    """Returns mean Hinge loss and N_delta, number of samples with non-zero loss"""
    hinge_ = F.relu(1 - y_true * y_pred) ** 2
    return torch.mean(hinge_), torch.sum(hinge_ > 0)


def orthogonal_init(model):
    """Orthogonal initializarion of a PyTorch model parameters.
    Biases are initialized at zero."""
    for name, param in model.named_parameters():
        if '0.weight' in name:
            nn.init.orthogonal_(param)
        if '0.bias' in name:
            nn.init.zeros_(param)
    return model


# Define the timeit decorator to time methods' execution
def decorator(d):
    """Make function d a decorator: d wraps a function fn."""
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d


@decorator
def timeit(f):
    """time a function, used as decorator"""
    def new_f(*args, **kwargs):
        bt = time()
        r = f(*args, **kwargs)
        et = time()
        print("Time spent on {0}: {1:.2f}s".format(f.__name__, et - bt))
        return r
    return new_f


# >>>>> PLOT FUNCTIONS <<<<< #


def plot_loss_N_delta(running_loss, running_N_delta):
    """Plot running loss and N_delta as function of epoch number on different axis."""

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.loglog(running_loss, color=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('$N_{\Delta}$', color=color)  # we already handled the x-label with ax1
    ax2.loglog(running_N_delta, color=color, lw=.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.grid()


def plot_results(datatype, L, D='D_', version=0):
    with open('results/{}/results/L{}.pkl'.format(datatype, L), 'rb') as f:
        results = pickle.load(f)

    N_delta_over_N = np.asarray(results['N_delta']) / results['N']

    plt.figure(1, figsize=(9, 6))
    plt.xlabel('$r = P/N$')
    plt.ylabel("$N_{\Delta}/N$")
    plt.plot(results['r'], N_delta_over_N, '--o', markerSize=10, label='{}, L{}, {}'.format(datatype, L, D))
    plt.grid()
    plt.legend()
    plt.savefig('./figures/Ndelta_N__vs__r_{}.png'.format(version), format='png')

    plt.figure(2, figsize=(9, 6))
    plt.xlabel('loss')
    plt.ylabel("$N_{\Delta}/N$")
    plt.semilogy(results['loss'], N_delta_over_N, '--o', markerSize=10, label='{}, L{}, {}'.format(datatype, L, D))
    plt.legend()
    plt.grid()
    plt.savefig('./figures/Ndelta_N__vs__loss_{}.png'.format(version), format='png')
