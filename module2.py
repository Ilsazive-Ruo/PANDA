import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import dgllife
from dgl.nn.pytorch import GATConv
from dgllife.model.gnn import mpnn
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout


class MyDataset(Dataset):
    def __init__(self, graph_pair, label):
        self.graph_pair = graph_pair
        self.label = label

    def __getitem__(self, item):
        return self.graph_pair[item][0], self.graph_pair[item][1], self.label[item]

    def __len__(self):
        return len(self.label)


class FSADataset(Dataset):
    def __init__(self, graph_pair, label, MPs):
        self.graph_pair = graph_pair
        self.label = label
        self.MPs = MPs

    def __getitem__(self, item):
        return self.graph_pair[item][0], self.graph_pair[item][1], self.label[item], self.MPs[item]

    def __len__(self):
        return len(self.label)


# GAT Layer
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(heads, in_features, out_features))  # weight for each head
        self.a = nn.Parameter(torch.empty(heads, 2 * out_features, 1))  # attention coefficient for each head
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        N = h.size(0)

        # Step 1: Linearly transform node features for each attention head
        h_prime = torch.matmul(h,
                               self.W.view(self.heads, self.in_features, self.out_features))  # [heads, N, out_features]

        # Step 2: Compute attention coefficients
        e = self.attention(h_prime, adj)  # [heads, N, N]

        # Step 3: Apply softmax to normalize attention scores
        attention = F.softmax(e, dim=-1)  # [heads, N, N]

        # Step 4: Apply attention to aggregate the node features
        h_out = torch.matmul(attention, h_prime)  # [heads, N, out_features]

        # Step 5: Concatenate all head outputs
        h_out = h_out.view(N, -1)  # [N, heads * out_features]

        return h_out

    def attention(self, h_prime, adj):
        N = h_prime.size(1)

        # Compute attention coefficients for each pair of nodes
        h_prime_concat = torch.cat([h_prime.repeat(N, 1, 1), h_prime.transpose(0, 1).repeat(N, 1, 1)],
                                   dim=-1)  # [heads, N, N, 2*out_features]
        e = torch.matmul(h_prime_concat, self.a.view(self.heads, 2 * self.out_features, 1))  # [heads, N, N, 1]

        e = e.squeeze(-1)  # [heads, N, N]

        return e


from dgl import function as fn
from dgl.nn.pytorch import edge_softmax


# Global Attention Mechanism (pooling or aggregation)
import torch
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import edge_softmax
from torch import nn


class GlobalAttentionLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GlobalAttentionLayer, self).__init__()
        self.W = nn.Linear(in_feats, out_feats)
        self.attn_w = nn.Parameter(torch.Tensor(out_feats * 2, 1))
        self.attn_b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_w)
        nn.init.zeros_(self.attn_b)

    def forward(self, g, features):
        # print('features', features.shape)
        transformed_features = self.W(features)
        # print(transformed_features.size())

        src, dst = g.edges()

        src_features = transformed_features[src]
        dst_features = transformed_features[dst]

        attention_scores = torch.cat([src_features, dst_features], dim=-1)
        # print('attention_scores.shape:', attention_scores.shape)
        # print(self.attn_w.shape)
        attention_scores = torch.matmul(attention_scores, self.attn_w).squeeze()
        # print('attention_scores.shape:', attention_scores.shape)
        attention_scores += self.attn_b

        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)

        attention_scores = edge_softmax(g, attention_scores)

        g.edata['attention'] = attention_scores
        # print(f'Number of nodes: {g.num_nodes()}')
        # print(f'Number of edges: {g.num_edges()}')
        # print('attention shape:', g.edata['attention'].shape)

        g.update_all(message_func=dgl.function.u_mul_e('h', 'attention', 'm'),
                     reduce_func=dgl.function.sum('m', 'h'))

        # 返回加权后的节点特征
        return g.ndata['h']


class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, f1, f2):
        Q1 = self.query(f1)  # (batch_size, D)
        K2 = self.key(f2)  # (batch_size, D)
        V2 = self.value(f2)  # (batch_size, D)

        attn_scores = torch.matmul(Q1, K2.transpose(-2, -1))
        attn_weights = F.softmax(attn_scores, dim=-1)

        fused_f2 = torch.matmul(attn_weights, V2)  # (batch_size, 1, D)
        fused_f1 = torch.matmul(attn_weights.transpose(-2, -1), self.value(f1))  # (batch_size, 1, D)

        fused_features = fused_f1 + fused_f2

        fused_features = fused_features.view(fused_features.size(0), -1)

        return fused_features


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Predictor, self).__init__()

        self.input = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        x = self.activation(self.input(in_feats))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def predict_prob(self, x):
        out = self.forward(x)
        return self.softmax(out)


# GAT with Global Attention (Three GAT layers followed by Three Global Attention layers)
class PANDA(nn.Module):
    def __init__(self, in_features, edge_dim, hidden_features, out_features, heads=1, dropout=0.1, alpha=0.2):
        super(ARCHIT, self).__init__()

        self.mpnn_layer = mpnn.MPNNGNN(node_in_feats=in_features,
                                       node_out_feats=in_features,
                                       edge_in_feats=edge_dim,
                                       edge_hidden_feats=12,
                                       num_step_message_passing=3)

        # Three GAT layers
        self.gat1 = GATConv(in_features, hidden_features * heads, heads, dropout)
        self.gat2 = GATConv(hidden_features * heads, hidden_features * heads, heads, dropout)
        self.gat3 = GATConv(hidden_features * heads, out_features, heads, dropout)

        # Three Global Attention layers
        self.global_att1 = GlobalAttentionLayer(out_features, hidden_features * heads)
        self.global_att2 = GlobalAttentionLayer(hidden_features * heads, hidden_features * heads)
        self.global_att3 = GlobalAttentionLayer(hidden_features * heads, out_features)

        self.readout_layer = AttentiveFPReadout(feat_size=out_features, num_timesteps=3, dropout=dropout)
        # self.readout_layer = dgllife.model.readout.mlp_readout.MLPNodeReadout(out_features, out_features, out_features)

    def forward(self, g):

        # g = dgl.add_self_loop(g)
        # print('looped:', g)

        feats = self.mpnn_layer(g, g.ndata['h'], g.edata['e'])

        # Apply first GAT layer
        h = self.gat1(g, feats)
        h = F.elu(h)  # Apply non-linearity
        # print('GAT1 output', h.size())

        # Apply second GAT layer
        h = self.gat2(g, h)
        h = F.elu(h)  # Apply non-linearity

        # Apply third GAT layer
        h = self.gat3(g, h)
        # print('GAT3 output', h.size())

        # Apply first Global Attention layer
        h = self.global_att1(g, h)  # Apply global attention
        # print('GATt1 output', h.size())

        # Apply second Global Attention layer
        h = self.global_att2(g, h)  # Apply global attention
        # print('GATt2 output', h.size())

        # Apply third Global Attention layer
        h = self.global_att3(g, h)  # Apply global attention
        # print('atted g', g)
        # g.edata.pop('attention', None)
        # print('del atted g', g)
        feats = self.readout_layer(g, h)

        return feats


class SYN_NC(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, num_heads=1):
        super(SYN_NC, self).__init__()

        self.archit1 = PANDA(node_size, edge_size, hidden_size, node_size, num_heads)
        self.archit2 = PANDA(node_size, edge_size, hidden_size, node_size, num_heads)
        # self.transformer_layers = nn.TransformerEncoderLayer(embed_size, embed_size)
        self.fusion_layer = AttentionFusion(node_size)
        self.predictor = Predictor(node_size, hidden_size, output_size=4)

    def forward(self, g1, g2):

        x1 = self.archit1(g1)
        x2 = self.archit2(g2)
        feats = self.fusion_layer(x1, x2)
        # print('x1.shape', x1.shape)
        # print('x2.shape', x2.shape)
        # print('feats.shape', feats.shape)
        output = self.predictor(feats)
        # print('output.shape', output.shape)

        return feats, output


class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_name):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.features = None
        self.hook = dict()
        def hook_fn(module, input, output):
            self.features = output
        layer = dict([*self.model.named_modules()])[layer_name]
        layer.register_forward_hook(hook_fn)

    def forward(self, x):
        self.model(x)
        return self.features


class SA(nn.Module):
    def __init__(self, pre_model, fusion_feature_size):
        super(SA, self).__init__()
        self.model = pre_model
        self.fc = Predictor(fusion_feature_size, fusion_feature_size * 4, output_size=2)

    def forward(self, g1, g2):
        features, _ = self.model(g1, g2)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

    def predict_prob(self, g1, g2):
        out = self.forward(g1, g2)
        return self.softmax(out)


class FSA(nn.Module):
    def __init__(self, pre_model, fusion_feature_size):
        super(FSA, self).__init__()
        self.model = pre_model
        self.fc = Predictor(fusion_feature_size + 4, fusion_feature_size * 4, output_size=2)

    def forward(self, g1, g2, MP):
        features, _ = self.model(g1, g2)
        features = torch.cat((features, MP), dim=1)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

    def predict_prob(self, g1, g2, MP):
        out = self.forward(g1, g2, MP)
        return self.softmax(out)
