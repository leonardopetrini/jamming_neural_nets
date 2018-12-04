import os
import numpy as np
import matplotlib


import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl


from mpl_toolkits.mplot3d import Axes3D

import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
import pickle
import sklearn.manifold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.optimize import fsolve

from time import gmtime, strftime, time

mpl.style.use('seaborn-muted')  # muted
matplotlib.use('Agg')

print('Starting... @ ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))


def countParams(d, h, L):
    """ Count parameters in a fully-connected rectangular network"""
    return h * ((L - 1) * (h + 1) + d + 2) + 1


def find_h(d, L, P, r):
    """Find the h-layers size for given set of params"""
    if L == 0:
        return
    if L == 1:
        return int(P / (r * (d + 2)))
    if d == 0:
        return int(np.sqrt(P / (L * r)))
    else:
        return int((-(L + d + 1) + np.sqrt((L + d + 1) ** 2 - 4 * (L - 1) * (1 - P / r))) / (2 * (L - 1)))


def find_h_triangular_net(N):
    return int(4 * np.cbrt(N/2)) + 1


def find_h_approx(d, L, P, r):
    return int((-d + np.sqrt(d ** 2 + 4 * P / r * (L - 1))) / (2 * (L - 1)))


def generate_D0(P, d, distribution='normal'):
    '''Generate points on the d-sphere and labels in {-1,+1}'''
    if distribution == 'normal':
        D0 = torch.empty((P, d)).normal_().cuda()
        labels = ((torch.empty((P, 1)).random_(0, 2) - .5) * 2).cuda()
    elif 'unif' in distribution:
        D0 = (torch.empty((P, d)).uniform_() * 2 - 1).cuda()
        labels = ((torch.empty((P, 1)).random_(0, 2) - .5) * 2).cuda()
    else:
        raise Exception('Insert either normal or uniform distribution')

    return D0, labels


def make_dir(main_dir_name):
    for sub_dir in ['/figures/loss', '/models', '/data']:
        try:
            os.makedirs(main_dir_name + sub_dir)
            print("Directory ", main_dir_name + sub_dir, " Created ")
        except FileExistsError:
            print("Directory ", main_dir_name + sub_dir, " already exists")


def pickle_save(dictionary, name, directory):
    with open(directory + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f)


def pickle_load(name, directory):
    with open(directory + '/' + name + '.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary

def load_model(N, main_dir):
    with open(f'{main_dir}/models/model_N{N}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def Net(d, h, L, gpuFlag=True):
    '''Build a NN with L hidden layers of hight h and input dimension d.
    If L = 0, Net is a perceptron.'''
    if L > 0:
        layers = [nn.Sequential(nn.Linear(d, h), nn.ReLU(), )]  # nn.BatchNorm1d(h, affine=False))]
        for l in range(1, L):
            layers.append(nn.Sequential(nn.Linear(h, h), nn.ReLU(), ))  # nn.BatchNorm1d(h, affine=False)))
        layers.append(nn.Linear(h, 1))
    elif L == 0:
        layers = [nn.Linear(d, 1)]

    model = nn.Sequential(*layers)

    if gpuFlag:
        model.cuda()

    model = orthogonal_init(model)

    return model


def hinge_loss(y_true, y_pred):
    '''Returns mean Hinge loss and N_delta, number of non-zero losses'''
    hinge_ = F.relu(1 - y_true * y_pred) ** 2
    return torch.mean(hinge_), torch.sum(hinge_ > 0)


def orthogonal_init(model):
    for name, param in model.named_parameters():
        if '0.weight' in name:
            nn.init.orthogonal_(param)
        if '0.bias' in name:
            nn.init.zeros_(param)
    return model


# >>>>> PLOT FUNCTIONS <<<<< #

def plot_loss_N_delta(running_loss, running_N_delta):
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


def plot_low_dimension(D, labels, data_type, model='PCA'):
    dataset = D.detach().cpu().numpy()

    if model == 'PCA':
        dataset = dataset - dataset.mean(axis=0)
        X_embedded = PCA(n_components=2).fit_transform(dataset)
    elif model == 'TSNE':
        X_embedded = TSNE(n_components=2, perplexity=30).fit_transform(dataset)
    elif model == 'LLE':
        X_embedded = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=10).fit_transform(dataset)
    plt.figure()
    cdict = {1: 'red', -1: 'blue'}
    plt.title("Dataset embedded in 2 dimensions")

    for i, g in enumerate(labels.detach().cpu().numpy()):
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], s=4, c=cdict[g[0]], label=g[0])

        # plt.text(X_embedded[:,0].max()-6, X_embedded[:,1].max()-6, "t-SNE\nP = {}\nN = {}\nr = {:0.2f}".format(P, totParams, P/totParams))
    # plt.savefig("{}/figures/{}_second_last_layer.png".format(data_type, model), format='png')


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
