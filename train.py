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

import pandas as pd


import mesh_model
import stats





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


def train(dataset, device, stats_list, args):

    df = pd.DataFrame(columns=['epoch','train_loss','test_loss'])

    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_False'

    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)

    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2 # the dynamic variables have the shape of 2 (velocity)

    model = mesh_model.MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_losses = []
    best_test_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops=0
        for batch in loader:
            #Note that normalization must be done before it's called. The unnormalized
            #data needs to be preserved in order to correctly calculate the loss
            batch=batch.to(device)
            opt.zero_grad()
            pred = model(batch,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            #pred = pred[batch.train_mask]
            #label = label[batch.train_mask]
            loss = model.loss(pred,batch,mean_vec_y,std_vec_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_loops+=1
        total_loss /= num_loops
        losses.append(total_loss)

        if epoch % 10 == 0:
            test_loss = test(test_loader, model,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)
            test_losses.append(test_loss.item())

            PATH = os.path.join(args.checkpoint_dir, model_name+'.csv')
            df.to_csv(PATH,index=False)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            test_losses.append(test_losses[-1])

        df = df.append({'epoch': epoch,'train_loss': losses[-1],'test_loss':test_losses[-1]},ignore_index=True)

        if(epoch%100==0):
            print("train loss", str(round(total_loss,2)), "test loss", str(round(test_loss.item(),2)))

            if(args.save_best_model):
                # saving model
                if not os.path.isdir( args.checkpoint_dir ):
                    os.mkdir(args.checkpoint_dir)

                PATH = os.path.join(args.checkpoint_dir, model_name+'.pt')
                torch.save(best_model.state_dict(), PATH )

    return test_losses, losses, best_model, best_test_loss, test_loader

def test(loader, test_model,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y, is_validation=False, save_model_preds=False, model_type=None):

    loss=0
    num_loops=0
    for data in loader:
        data=data.to(device)
        with torch.no_grad():
            pred = test_model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            loss += test_model.loss(pred, data,mean_vec_y,std_vec_y)
        num_loops+=1

    return loss/num_loops


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d









# Get the right path of dataset
DATA_FOLDER_NAME = 'torch_dataset'
root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, '01_dataset/cylinder_flow', DATA_FOLDER_NAME)



device = 'cuda' if torch.cuda.is_available() else 'cpu'




for args in [
    {'model_type': 'meshgraphnet', 'dataset': 'mini10', 'num_layers': 1,
      'batch_size': 16, 'hidden_dim': 4, 'epochs': 200,
      'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-4, 'lr': 0.001,
      'train_size': 2, 'shuffle': False, 'save_best_model': False, 'checkpoint_dir': './best_models/'},
]:
    args = objectview(args)

    args.model_type = 'meshgraphnet'

    if args.dataset == 'mini10':
        file_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_vis.pt')
        #stats_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_ms.pt')
        dataset = torch.load(file_path)[:50] #, batch_size = args['batch_size'])
        if(args.shuffle):
            random.shuffle(dataset)
        #dataset_stats=torch.load(stats_path)
        #import pdb; pdb.set_trace()
    else:
        raise NotImplementedError("Unknown dataset")

    ## TODO: CHECK PERFORMANCE OF STAT CHANGES BY ITERATING THROUGH ALL DATASETS AND CHECKING
    ##       THE MEAN AND VAR OF NORMALIZED DATA
    stats_list = stats.get_stats(dataset)

    test_losses, losses, best_model, best_test_loss, test_loader = train(dataset, device, stats_list, args)

    print("Min test set loss: {0}".format(min(test_losses)))
    print("Minimum loss: {0}".format(min(losses)))

    # Run test for our best model to save the predictions!
    #test(test_loader, best_model, is_validation=False, save_model_preds=True, model_type=model)
    #print()

    plt.title(args.dataset)
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_losses, label="test loss" + " - " + args.model_type)

    plt.legend()
    plt.show()
