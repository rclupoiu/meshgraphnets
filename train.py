import torch
print(torch.__version__)

import os

#import torch_geometric

#import torch_scatter
#import torch.nn as nn
#import torch.nn.functional as F

#import torch_geometric.nn as pyg_nn
#import torch_geometric.utils as pyg_utils

#from torch import Tensor
#from typing import Union, Tuple, Optional

#from torch.nn import Parameter, Linear, Sequential, LayerNorm, ReLU
#from torch_sparse import SparseTensor, set_diag
#from torch_geometric.nn.conv import MessagePassing
#from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

import  argparse

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

#import networkx as nx
import numpy as np
#import torch
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy

#from torch_geometric.datasets import TUDataset
#from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

#import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt


def train(dataset, device, stats_list, args):

    df = pd.DataFrame(columns=['epoch','train_loss','test_loss', 'velo_val_loss'])

    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)

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
    velo_val_losses = []
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
            if (args.save_velo_val):
                # save velocity evaluation
                test_loss, velo_val_rmse = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_y,std_vec_y, args.save_velo_val)
                velo_val_losses.append(velo_val_rmse.item())
            else:
                test_loss, _ = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_y,std_vec_y, args.save_velo_val)

            test_losses.append(test_loss.item())

            # saving model
            if not os.path.isdir( args.checkpoint_dir ):
                os.mkdir(args.checkpoint_dir)

            PATH = os.path.join(args.checkpoint_dir, model_name+'.csv')
            df.to_csv(PATH,index=False)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            test_losses.append(test_losses[-1])
            velo_val_losses.append(velo_val_losses[-1])

        if (args.save_velo_val):
            df = df.append({'epoch': epoch,'train_loss': losses[-1],
                            'test_loss':test_losses[-1],
                           'velo_val_loss': velo_val_losses[-1]}, ignore_index=True)
        else:
            df = df.append({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]}, ignore_index=True)
        if(epoch%100==0):
            if (args.save_velo_val):
                print("train loss", str(round(total_loss, 2)),
                      "test loss", str(round(test_loss.item(), 2)),
                      "velo loss", str(round(velo_val_rmse.item(), 2)))
            else:
                print("train loss", str(round(total_loss,2)), "test loss", str(round(test_loss.item(),2)))


            if(args.save_best_model):

                PATH = os.path.join(args.checkpoint_dir, model_name+'.pt')
                torch.save(best_model.state_dict(), PATH )

    return test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader

def test(loader,device,test_model,
         mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y, is_validation,
          delta_t=0.01, save_model_preds=False, model_type=None):

    loss=0
    velo_rmse = 0
    num_loops=0

    for data in loader:
        data=data.to(device)
        with torch.no_grad():
            pred = test_model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            loss += test_model.loss(pred, data,mean_vec_y,std_vec_y)

            if (is_validation):

                normal = torch.tensor(0)
                outflow = torch.tensor(5)
                loss_mask = torch.logical_or((torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(0)),
                                             (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)))

                eval_velo = data.x[:, 0:2] + pred[:] * delta_t
                gs_velo = data.x[:, 0:2] + data.y[:] * delta_t

                error = torch.sum((eval_velo - gs_velo) ** 2, axis=1)
                velo_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops+=1
        # if velocity is evaluated, return velo_rmse as 0
    return loss/num_loops, velo_rmse/num_loops

def save_plots(args, losses, test_losses, velo_val_losses):
    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    PATH = os.path.join(args.postprocess_dir, model_name + '.pdf')

    f = plt.figure()
    plt.title(args.dataset)
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_losses, label="test loss" + " - " + args.model_type)
    if (args.save_velo_val):
        plt.plot(velo_val_losses, label="velocity loss" + " - " + args.model_type)

    plt.legend()
    plt.show()
    f.savefig(PATH, bbox_inches='tight')

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d



def main(args):

    # Get the right path of dataset
    DATA_FOLDER_NAME = '01_dataset\\cylinder_flow\\torch_dataset'
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, DATA_FOLDER_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Getting {}...'.format(device))
    '''for args in [
        {'model_type': 'meshgraphnet', 'dataset': 'mini10', 'num_layers': 10,
          'batch_size': 16, 'hidden_dim': 10, 'epochs': 5000,
          'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-4, 'lr': 0.001,
          'train_size': 45, 'test_size': 5, 'shuffle': True, 'save_best_model': False, 'checkpoint_dir': './best_models/'},
    ]:
        args = objectview(args)'''

    if args.dataset == 'mini10':
        file_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_vis.pt')
        #stats_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_ms.pt')
        dataset = torch.load(file_path)[:(args.train_size+args.test_size)] #, batch_size = args['batch_size'])
        if(args.shuffle):
            random.shuffle(dataset)
        #dataset_stats=torch.load(stats_path)
        #import pdb; pdb.set_trace()
    else:
        raise NotImplementedError("Unknown dataset")

    ## TODO: CHECK PERFORMANCE OF STAT CHANGES BY ITERATING THROUGH ALL DATASETS AND CHECKING
    ##       THE MEAN AND VAR OF NORMALIZED DATA
    stats_list = stats.get_stats(dataset)

    test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader = train(dataset, device, stats_list, args)

    print("Min test set loss: {0}".format(min(test_losses)))
    print("Minimum loss: {0}".format(min(losses)))
    if (args.save_velo_val):
        print("Minimum velocity validation loss: {0}".format(min(velo_val_losses)))

    # Run test for our best model to save the predictions!
    #test(test_loader, best_model, is_validation=False, save_model_preds=True, model_type=model)
    #print()
    save_plots(args, losses, test_losses, velo_val_losses)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_type', type=str, help='', default='meshgraphnet')
    argparser.add_argument('--dataset', type=str, help='', default='mini10')
    argparser.add_argument('--num_layers', type=int, help='', default=10)
    argparser.add_argument('--batch_size', type=int, help='', default=16)
    argparser.add_argument('--hidden_dim', type=int, help='', default=10)
    argparser.add_argument('--epochs', type=int, help='', default=5000)
    argparser.add_argument('--opt', type=str, help='', default='adam')
    argparser.add_argument('--opt_scheduler', type=str, help='', default='none')
    argparser.add_argument('--opt_restart', type=int, help='', default=0)
    argparser.add_argument('--weight_decay', type=float, help='', default=5e-4)
    argparser.add_argument('--lr', type=float, help='', default=0.001)
    argparser.add_argument('--train_size', type=int, help='', default=45)
    argparser.add_argument('--test_size', type=int, help='', default=5)
    argparser.add_argument('--shuffle', type=bool, help='', default=True)
    argparser.add_argument('--save_velo_val', type=bool, help='', default=True)
    argparser.add_argument('--save_best_model', type=bool, help='', default=True)
    argparser.add_argument('--checkpoint_dir', type=str, help='', default='./best_models/')
    argparser.add_argument('--postprocess_dir', type=str, help='', default='./2d_loss_plots/')

    args = argparser.parse_args()

    main(args)
