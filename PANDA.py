import torch
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch import GATConv
from dgllife.model.gnn import mpnn
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
import dgl
from dgl.nn.pytorch import edge_softmax
from torch import nn
from dgllife.model.model_zoo import mlp_predictor
import copy


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

        attention_scores += self.attn_b

        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)

        attention_scores = edge_softmax(g, attention_scores)

        g.edata['attention'] = attention_scores

        g.update_all(message_func=dgl.function.u_mul_e('h', 'attention', 'm'),
                     reduce_func=dgl.function.sum('m', 'h'))

        return g.ndata['h']


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super(MultiHeadAttentionFusion, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Q, K, V for both inputs
        self.query1 = nn.Linear(input_dim, input_dim)
        self.key2 = nn.Linear(input_dim, input_dim)
        self.value2 = nn.Linear(input_dim, input_dim)

        self.query2 = nn.Linear(input_dim, input_dim)
        self.key1 = nn.Linear(input_dim, input_dim)
        self.value1 = nn.Linear(input_dim, input_dim)

        self.output_proj = nn.Linear(input_dim * 2, input_dim * 2)

    def split_heads(self, x):
        B, D = x.shape
        x = x.view(B, self.num_heads, self.head_dim)  # (B, H, D/H)
        return x

    def combine_heads(self, x):
        return x.view(x.size(0), -1)

    def forward(self, f1, f2):

        Q1 = self.split_heads(self.query1(f1))
        K2 = self.split_heads(self.key2(f2))
        V2 = self.split_heads(self.value2(f2))

        attn_scores_1 = torch.matmul(Q1, K2.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, 1)
        attn_weights_1 = F.softmax(attn_scores_1, dim=-1)
        fused_f1 = torch.matmul(attn_weights_1, V2)
        fused_f1 = self.combine_heads(fused_f1)

        Q2 = self.split_heads(self.query2(f2))
        K1 = self.split_heads(self.key1(f1))
        V1 = self.split_heads(self.value1(f1))

        attn_scores_2 = torch.matmul(Q2, K1.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights_2 = F.softmax(attn_scores_2, dim=-1)
        fused_f2 = torch.matmul(attn_weights_2, V1)
        fused_f2 = self.combine_heads(fused_f2)

        fused = torch.cat([fused_f1, fused_f2], dim=-1)  # (B, 2D)
        fused = self.output_proj(fused)

        return fused


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # 对应任务1
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # 对应任务2
        self.log_sigma3 = nn.Parameter(torch.tensor(0.0))  # 对应任务3

    def forward(self, loss1, loss2, loss3):
        total_loss = (
            torch.exp(-self.log_sigma1) * loss1 + self.log_sigma1 +
            torch.exp(-self.log_sigma2) * loss2 + self.log_sigma2 +
            torch.exp(-self.log_sigma3) * loss3 + self.log_sigma3
        )
        return total_loss


class ARCHIT(nn.Module):
    def __init__(self, in_features, edge_dim, hidden_features, out_features, heads=1, dropout=0.1):
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

    def forward(self, g):

        feats = self.mpnn_layer(g, g.ndata['h'], g.edata['e'])
        h = self.gat1(g, feats)
        h = F.elu(h)
        h = self.gat2(g, h)
        h = F.elu(h)
        h = self.gat3(g, h)
        h = self.global_att1(g, h)
        h = self.global_att2(g, h)
        h = self.global_att3(g, h)
        feats = self.readout_layer(g, h)

        return feats


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        super(Predictor, self).__init__()

        self.input = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(num_layers - 1)
        ])
        self.bn = nn.BatchNorm1d(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        x = self.activation(self.input(in_feats))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = self.bn(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + residual
        x = self.output_layer(x)
        return x


class Panda(nn.Module):
    def __init__(self, encoder, node_size, hidden_size):
        super(Panda, self).__init__()

        self.archit1 = encoder
        self.archit2 = encoder

        self.fusion_layer_syn = MultiHeadAttentionFusion(node_size)
        self.fusion_layer_sa = MultiHeadAttentionFusion(node_size)
        self.fusion_layer_fsa = MultiHeadAttentionFusion(node_size)

        self.predictor_syn = Predictor(node_size * 2, hidden_size, output_size=4, num_layers=4, dropout=0.1)
        self.predictor_sa = Predictor(node_size * 2, hidden_size, output_size=2, num_layers=5, dropout=0.1)
        self.predictor_fsa = Predictor(node_size * 2 + 4, hidden_size, output_size=2, num_layers=3, dropout=0.1)

        self.uncertainty_weighting_loss = MultiTaskUncertaintyLoss()

    def forward(self, g1, g2, mc):

        x1 = self.archit1(g1)
        x2 = self.archit2(g2)
        feats_syn = self.fusion_layer_syn(x1, x2)
        feats_sa = self.fusion_layer_sa(x1, x2)
        feats_fsa = self.fusion_layer_fsa(x1, x2)
        feats_fsa = torch.cat([feats_fsa, mc], dim=-1)
        output_syn = self.predictor_syn(feats_syn)
        output_sa = self.predictor_sa(feats_sa)
        output_fsa = self.predictor_fsa(feats_fsa)

        return output_syn.squeeze(), output_sa.squeeze(), output_fsa.squeeze()


class PandaPretrain(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, num_heads=1):
        super(PandaPretrain, self).__init__()

        self.archit = ARCHIT(node_size, edge_size, hidden_size, node_size, num_heads)

        self.predictor_logp = Predictor(node_size, hidden_size, output_size=1)
        self.predictor_qed = Predictor(node_size, hidden_size, output_size=1)
        self.predictor_sas = Predictor(node_size, hidden_size, output_size=1)

    def forward(self, g1):

        feats = self.archit(g1)

        logp = self.predictor_logp(feats)
        qed = self.predictor_qed(feats)
        sas = self.predictor_sas(feats)

        return feats, logp, qed, sas


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.005, path='checkpoint.pth'):
        """
        参数:
        - patience: 容忍多少个epoch内没有改善才终止训练
        - verbose: 是否打印每次提前终止的信息
        - delta: 如果验证损失没有变化大于delta，就认为没有改善
        - path: 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf  # 设定初始损失为正无穷
        self.early_stop = False
        self.best_model_wts = None
        self.metrix = None
        self.pr = None
        self.roc = None

    def __call__(self, val_loss, model, metrix=None, roc=None, pr=None):
        """
        检查是否满足提前终止的条件
        """
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.metrix = metrix
            self.pr = pr
            self.roc = roc
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best loss: {self.best_loss}")

        return self.early_stop

    def load_best_model(self, model):
        """
        恢复最佳模型权重
        """
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
        else:
            print('no weight saved')


class Panda_ST(nn.Module):
    def __init__(self, encoder, node_size, hidden_size, task):
        super(Panda_ST, self).__init__()

        self.archit1 = encoder
        self.archit2 = encoder
        self.task = task

        # self.transformer_layers = nn.TransformerEncoderLayer(embed_size, embed_size)
        self.fusion_layer_syn = MultiHeadAttentionFusion(node_size)
        self.fusion_layer_sa = MultiHeadAttentionFusion(node_size)
        self.fusion_layer_fsa = MultiHeadAttentionFusion(node_size)
        self.predictor_syn = Predictor(node_size * 2, hidden_size, output_size=4, num_layers=4, dropout=0.1)
        self.predictor_sa = Predictor(node_size * 2, hidden_size, output_size=2, num_layers=5, dropout=0.1)
        self.predictor_fsa = Predictor(node_size * 2 + 4, hidden_size, output_size=2, num_layers=3, dropout=0.1)

        self.uncertainty_weighting_loss = MultiTaskUncertaintyLoss()

    def forward(self, g1, g2, mc=None):

        x1 = self.archit1(g1)
        x2 = self.archit2(g2)
        if self.task == 'PV':
            feats = self.fusion_layer_syn(x1, x2)
            output = self.predictor_syn(feats)
        elif self.task == 'SA':
            feats = self.fusion_layer_sa(x1, x2)
            output = self.predictor_sa(feats)
        elif self.task == 'FSA':
            feats = self.fusion_layer_fsa(x1, x2)
            feats = torch.cat([feats, mc], dim=-1)
            output = self.predictor_fsa(feats)
        else:
            print('invalid task input')
            output = None

        return output.squeeze()


class GNN_ST(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, task):
        super(GNN_ST, self).__init__()
        self.mpnn_layer1 = mpnn.MPNNGNN(node_in_feats=node_size,
                                       node_out_feats=node_size,
                                       edge_in_feats=edge_size,
                                       edge_hidden_feats=12,
                                       num_step_message_passing=3)
        self.mpnn_layer2 = mpnn.MPNNGNN(node_in_feats=node_size,
                                        node_out_feats=node_size,
                                        edge_in_feats=edge_size,
                                        edge_hidden_feats=12,
                                        num_step_message_passing=3)
        self.readout1 = AttentiveFPReadout(feat_size=node_size,
                                          num_timesteps=6,
                                          dropout=0.2)
        self.readout2 = AttentiveFPReadout(feat_size=node_size,
                                           num_timesteps=6,
                                           dropout=0.2)
        self.predictor_sa = mlp_predictor.MLPPredictor(in_feats=node_size * 2,
                                                       hidden_feats=hidden_size,
                                                       n_tasks=2,
                                                       dropout=0.1)
        self.predictor_fsa = mlp_predictor.MLPPredictor(in_feats=node_size * 2 + 4,
                                                        hidden_feats=hidden_size,
                                                        n_tasks=2,
                                                        dropout=0.1)
        self.predictor_syn = mlp_predictor.MLPPredictor(in_feats=node_size * 2,
                                                        hidden_feats=hidden_size,
                                                        n_tasks=4,
                                                        dropout=0.1)

        self.task = task

    def forward(self, g1, g2, mc=None):

        x1 = self.mpnn_layer1(g1, g1.ndata['h'], g1.edata['e'])
        x1 = self.readout1(g1, x1)
        x2 = self.mpnn_layer2(g2, g2.ndata['h'], g2.edata['e'])
        x2 = self.readout2(g2, x2)
        if self.task == 'PV':
            feats = torch.cat([x1, x2], dim=-1)
            output = self.predictor_syn(feats)
        elif self.task == 'SA':
            feats = torch.cat([x1, x2], dim=-1)
            output = self.predictor_sa(feats)
        elif self.task == 'FSA':
            feats = torch.cat([x1, x2], dim=-1)
            feats = torch.cat([feats, mc], dim=-1)
            output = self.predictor_fsa(feats)
        else:
            print('invalid task input')
            output = None

        return output.squeeze()


class GNN_MT(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size):
        super(GNN_MT, self).__init__()
        self.mpnn_layer1 = mpnn.MPNNGNN(node_in_feats=node_size,
                                       node_out_feats=node_size,
                                       edge_in_feats=edge_size,
                                       edge_hidden_feats=12,
                                       num_step_message_passing=3)
        self.mpnn_layer2 = mpnn.MPNNGNN(node_in_feats=node_size,
                                        node_out_feats=node_size,
                                        edge_in_feats=edge_size,
                                        edge_hidden_feats=12,
                                        num_step_message_passing=3)
        self.readout1 = AttentiveFPReadout(feat_size=node_size,
                                          num_timesteps=6,
                                          dropout=0.2)
        self.readout2 = AttentiveFPReadout(feat_size=node_size,
                                           num_timesteps=6,
                                           dropout=0.2)
        self.predictor_sa = mlp_predictor.MLPPredictor(in_feats=node_size * 2,
                                                  hidden_feats=hidden_size,
                                                  n_tasks=2,
                                                  dropout=0.1)
        self.predictor_fsa = mlp_predictor.MLPPredictor(in_feats=node_size * 2 + 4,
                                                       hidden_feats=hidden_size,
                                                       n_tasks=2,
                                                       dropout=0.1)
        self.predictor_syn = mlp_predictor.MLPPredictor(in_feats=node_size * 2,
                                                       hidden_feats=hidden_size,
                                                       n_tasks=4,
                                                       dropout=0.1)


    def forward(self, g1, g2, mc=None):

        x1 = self.mpnn_layer1(g1, g1.ndata['h'], g1.edata['e'])
        x1 = self.readout1(g1, x1)
        x2 = self.mpnn_layer2(g2, g2.ndata['h'], g2.edata['e'])
        x2 = self.readout2(g2, x2)
        feats1 = torch.cat([x1, x2], dim=-1)

        output1 = self.predictor_syn(feats1)
        feats2 = torch.cat([x1, x2], dim=-1)

        output2 = self.predictor_sa(feats2)
        feats3 = torch.cat([x1, x2], dim=-1)
        feats3 = torch.cat([feats3, mc], dim=-1)
        output3 = self.predictor_fsa(feats3)

        return output1.squeeze(), output2.squeeze(), output3.squeeze()


