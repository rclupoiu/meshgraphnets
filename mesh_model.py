import torch

#import torch_geometric

#import torch
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

import stats

class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """

        self.num_layers = args.num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear( input_dim_edge , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim)
                              )


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear( hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )


        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        for _ in range(self.num_layers):
            edge_procssor_layer, node_processor_layer = self.build_processor_model(hidden_dim, hidden_dim)

            self.processor.append(edge_procssor_layer)
            self.processor.append(node_processor_layer)

    def build_processor_model(self, hidden_dim, output_dim):
        """
        Instantiate processor layer for both edges and nodes. Note edge_processor_layer needs to
        be called first to update edge embeddings.
        """
        edge_procssor_layer = EdgeProcessorLayer( hidden_dim, output_dim)
        node_processor_layer = NodeProcessorLayer( hidden_dim, output_dim)
        return edge_procssor_layer, node_processor_layer

    def forward(self, data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr, pressure = data.x, data.edge_index, data.edge_attr, data.p

        x = stats.normalize(x,mean_vec_x,std_vec_x)
        edge_attr=stats.normalize(edge_attr,mean_vec_edge,std_vec_edge)

        #print("x shape", x.shape)
        #print("edge_attr shape", edge_attr.shape)


        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape 128

        edge_attr = self.edge_encoder(edge_attr) # output shape 128
        #print('edge_attr shape {}'.format(edge_attr.size()))

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers*2):
            if (i % 2 == 1):
                # Step 2: update node embeddings
                x = self.processor[i](x, edge_index, edge_attr)
            else:
                # Step 1: update edge embeddings
                edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)

    def loss(self, pred, inputs,mean_vec_y,std_vec_y):
        # label: ground_truth dyanmic variables [1]
        normal=torch.tensor(0)
        outflow=torch.tensor(5)
        loss_mask=torch.logical_or((torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(0)),
                                   (torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(5)))

        labels = stats.normalize(inputs.y,mean_vec_y,std_vec_y)
        error=torch.sum((labels-pred)**2,axis=1)
        loss=torch.sqrt(torch.mean(error[loss_mask]))
        return loss



class EdgeProcessorLayer(MessagePassing):

    def __init__(self, in_channels, out_channels,  **kwargs):
        super(EdgeProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        EdgeProcessorLayer takes the node embeddings of two nodes connected to the current edge,
        and the edge feature of itself
        """
        self.mlp = Sequential(Linear( 3* in_channels , out_channels),
                              ReLU(),
                              Linear( out_channels, out_channels),
                              ReLU(),
                              Linear( out_channels, out_channels),
                              LayerNorm(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.mlp[0].reset_parameters()
        self.mlp[2].reset_parameters()
        #print("Rest MLP layer for EdgeProcessor")

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """
        # print('Running here...')
        out = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]
        #print('out shape {}'.format(out.size()))
        new_edge_attr = edge_attr + out # residual connection

        return new_edge_attr

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]
        """
        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
        #print('Shape of x_i {}'.format(x_i.size()))
        #print('Shape of x_j {}'.format(x_j.size()))
        #print('Shape of edge_attr {}'.format(edge_attr.size()))

        #print('Shape of tmp {}'.format(tmp.size()))
        return self.mlp(tmp)

    def aggregate(self, edge_msg):
        """
        Do not aggregate for edges
        """
        return edge_msg
















class NodeProcessorLayer(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(NodeProcessorLayer, self).__init__( **kwargs )
        """
        in_channels: dim of node feature [128], out_channels: dim of edge feature [128]

        NodeProcessorLayer aggregates embeddings from neighboring edges with sum
        aggregator with the self node embedding [self-implemented]
        """
        self.mlp = Sequential(Linear( 2* in_channels , out_channels),
                              ReLU(),
                              Linear( out_channels, out_channels),
                              ReLU(),
                              Linear( out_channels, out_channels),
                              LayerNorm(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.mlp[0].reset_parameters()
        self.mlp[2].reset_parameters()
        #print("Rest MLP layer for EdgeProcessor")

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        x has shpae [node_num , in_channels]
        edge_index: [2, edge_num]
        """

        out = self.propagate(edge_index, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]
                                                                                    # out is the aggregated neighboring edge features
        tmp = torch.cat([x, out], dim = 1) # tmp has the shape of [N, 2 * in_channels]
        new_node_feature = self.mlp(tmp) + x
        return new_node_feature


    def message(self, edge_attr ):
        """
        message comes from edge embeddings
        """
        return edge_attr

    def aggregate(self, edge_attr, edge_index, dim_size = None):
        """
        Perform sum aggregation of all edge attributes of a target nodes

        edge_index: Edge indices of shape [2, num_edges]
        edge_attr: Edge embeddings of shape [num_edges, num_features]
        """
        out = None # aggregated edge_embeddings of shape [num_nodes, num_embeddings]

        # The axis along which to index number of nodes.
        node_dim = 0 # The first axis.

        # print('edge_attr {}'.format(edge_attr.size()))
        # aggregate all neighboring edge attributes
        out = torch_scatter.scatter(edge_attr, edge_index[0, :], dim=node_dim, reduce = 'sum')

        return out
