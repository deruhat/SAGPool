import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool





class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)

        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.conv5 = GCNConv(self.nhid, self.nhid)
        self.conv6 = GCNConv(self.nhid, self.nhid)
        self.conv7 = GCNConv(self.nhid, self.nhid)
        self.conv8 = GCNConv(self.nhid, self.nhid)
        self.conv9 = GCNConv(self.nhid, self.nhid)

        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.conv10 = GCNConv(self.nhid, self.nhid)
        self.conv11 = GCNConv(self.nhid, self.nhid)
        self.conv12 = GCNConv(self.nhid, self.nhid)
        self.conv13 = GCNConv(self.nhid, self.nhid)
        self.conv14 = GCNConv(self.nhid, self.nhid)

        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, edge_index))

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = x1 + x2

        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = x2 + x3

        x = F.relu(self.conv4(x3, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #x4 = x3 + x4

        x5 = F.relu(self.conv5(x, edge_index))
        x5 = x + x5

        x6 = F.relu(self.conv6(x5, edge_index))
        x6 = x5 + x6

        x7 = F.relu(self.conv7(x6, edge_index))
        x7 = x6 + x7

        x8 = F.relu(self.conv8(x7, edge_index))
        x8 = x7 + x8

        x = F.relu(self.conv9(x8, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x9 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #x9 = x8 + x9

        x10 = F.relu(self.conv10(x, edge_index))
        x10 = x + x10

        x11 = F.relu(self.conv11(x10, edge_index))
        x11 = x10 + x11

        x12 = F.relu(self.conv12(x11, edge_index))
        x12 = x11 + x12

        x13 = F.relu(self.conv13(x12, edge_index))
        x13 = x12 + x13 

        x = F.relu(self.conv14(x13, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x14 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #x14 = x13 + x14 

        x = x4 + x9 + x14 

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    