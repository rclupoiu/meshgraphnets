import torch
print(torch.__version__)

import os

import torch_geometric

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional

from torch.nn import Parameter, Linear, Sequential, LayerNorm, ReLU
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

import random


import mesh_model





import torch.optim as optim
def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer















import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt


def train(dataset, device, args):

    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2 # the dynamic variables have the shape of 2 (velocity)

    model = mesh_model.MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops=0
        for batch in loader:
            batch=batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            #pred = pred[batch.train_mask]
            #label = label[batch.train_mask]
            loss = model.loss(pred, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            num_loops+=1
        total_loss /= len(loader)
        losses.append(total_loss)

        if epoch % 10 == 0:
          test_acc = test(test_loader, model)
          test_accs.append(test_acc.item())
          if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])

        if(epoch%100==0):
            print("train loss", str(round(total_loss,2)), "test loss", str(round(test_acc.item(),2)))

    return test_accs, losses, best_model, best_acc, test_loader

def test(loader, test_model, is_validation=False, save_model_preds=False, model_type=None):

    loss=0
    num_loops=0
    for data in loader:
        data=data.to(device)
        with torch.no_grad():
            pred = test_model(data)
            loss += test_model.loss(pred, data) * data.num_graphs
        num_loops+=1

    return loss/len(loader)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d









# Get the right path of dataset
DATA_FOLDER_NAME = 'torch_dataset'
root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, '01_dataset/cylinder_flow', DATA_FOLDER_NAME)



device = 'cuda' if torch.cuda.is_available() else 'cpu'




for args in [
    {'model_type': 'meshgraphnet', 'dataset': 'mini10', 'num_layers': 10,
      'batch_size': 16, 'hidden_dim': 10, 'epochs': 5000,
      'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-4, 'lr': 0.001,
      'train_size': 45, 'shuffle': True},
]:
    args = objectview(args)

    args.model_type = 'meshgraphnet'

    if args.dataset == 'mini10':
        file_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj.pt')
        stats_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_ms.pt')
        dataset = torch.load(file_path)[:50] #, batch_size = args['batch_size'])
        if(args.shuffle):
            random.shuffle(dataset)
        dataset_stats=torch.load(stats_path)
        #import pdb; pdb.set_trace()
    else:
        raise NotImplementedError("Unknown dataset")

    test_accs, losses, best_model, best_acc, test_loader = train(dataset, device, args)

    print("Min test set loss: {0}".format(min(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))

    # Run test for our best model to save the predictions!
    #test(test_loader, best_model, is_validation=False, save_model_preds=True, model_type=model)
    #print()

    plt.title(args.dataset)
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_accs, label="test loss" + " - " + args.model_type)

    plt.legend()
    plt.show()
