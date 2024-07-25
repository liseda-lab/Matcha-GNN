from .models import *
import dgl
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
import dgl.nn.pytorch as dglpy


def load_checkpoint(self, file):
    """
    Load the model from input file of ".pt" extension
    """
    self.model = torch.load(file)

def save_checkpoint(self, file):
    """
    Save the model to input file of ".pt" extension
    """
    torch.save(self.model, file)


class GCN:
    """
    Graph Convolutional Network DGL implementation
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = dglpy.Sequential([
            dglnn.pytorch.conv.GraphConv(in_channels, hidden_channels),
            dglnn.pytorch.conv.GraphConv(in_channels, hidden_channels*2),
            dglnn.pytorch.conv.GraphConv(hidden_channels*2, out_channels)
        ])

    def forward(self, graph, x):

        for layer in self.module:
            x = F.relu(layer(graph, x))
        return x


class GAT:
    """
    Graph Attention Network DGL implementation
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=3):
        super().__init__()
        self.module = dglpy.Sequential([
            dglnn.pytorch.conv.GATConv(in_channels, hidden_channels, num_heads),
            dglnn.pytorch.conv.GATConv(in_channels, hidden_channels*2, num_heads),
            dglnn.pytorch.conv.GATConv(hidden_channels*2, out_channels, num_heads)
        ])

    def forward(self, graph, x):

        for layer in self.module:
            x = F.relu(layer(graph, x))
        return x


class DeepGCNLayer:
    """
    Deep Graph Convolutional Network Layer
    """
    def __init__(self, in_feats, hid_feats, out_feats):

        self.linear1 = dglnn.Linear(in_feats, hid_feats)
        self.linear2 = dglnn.Linear(hid_feats, out_feats)

    def forward(self, g, x):
        h = x
        with g.local_scope():
            for linear in [self.linear1, self.linear2]:
                g.ndata["h"] = h
                g.update_all(dgl.function.copy_u(u="h", out="m"), dgl.function.sum(msg="m", out="h"))
                h = g.ndata["h"]
                h = linear(h)
        return h



class DeepGCN():
    """
    Deep Graph Convolutional Network implementation
    """
    def __init__(self, in_channels, hidden_channels, out_channels):

        self.module = dglpy.Sequential([
            DeepGCNLayer(in_channels, in_channels*2, hidden_channels),
            DeepGCNLayer(hidden_channels, hidden_channels*2, hidden_channels),
            DeepGCNLayer(hidden_channels, hidden_channels*2, out_channels)
        ])

    def forward(self, graph, x):

        for layer in self.module:
            x = F.relu(layer(graph, x))
        return x


class GIN:
    """
    Graph Isomorphism Network DGL implementation
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = dglpy.Sequential([
            dglnn.pytorch.conv.GINConv(dglpy.Sequential([
                dglnn.Linear(in_channels, hidden_channels),
                dglnn.BatchNorm1d(hidden_channels),
                dglnn.ReLU(),
                dglnn.Linear(hidden_channels, hidden_channels)
            ])),
            dglnn.pytorch.conv.GINConv(dglpy.Sequential([
                dglnn.Linear(hidden_channels, hidden_channels*2),
                dglnn.BatchNorm1d(hidden_channels*2),
                dglnn.ReLU(),
                dglnn.Linear(hidden_channels*2, out_channels)
            ]))
        ])

    def forward(self, graph, x):

        for layer in self.module:
            x = F.relu(layer(graph, x))
        return x

