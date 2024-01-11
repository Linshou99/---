"""Meta Graph Transformer (MGT)"""
import math

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 时间属性嵌入后（映射到d_model维），再与位置编码拼接，然后再次映射到d_model维构成TemporalEmbedding
class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model, max_len):
        super(TemporalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.d_model = d_model
        #位置编码，shape(max_len, d_model),(4,16)
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer('pe', pe)
        # 2个nn.Embedding，[73,16]每一行对应一天内一个时间间隔的嵌入、[2,16]一行对应‘是休息日’、另一行对应‘非休息日’的嵌入
        self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in num_embeddings])
        # 3*16->16，这个+1是？
        self.linear = nn.Linear((len(num_embeddings) + 1) * d_model, d_model)

    def forward(self, extras):
        # 历史和未来交通序列的时间属性信息 extras = [inputs_time0, targets_time0, inputs_time1, targets_time1]
        assert len(extras) == 2 * len(self.num_embeddings)
        # print('len(extras)',len(extras))
        inputs_extras = extras[::2] # # 历史交通序列时间属性
        targets_extras = extras[1::2]   # # 未来交通序列时间属性

        # P: 历史交通序列长度, Q: 未来交通序列长度，inputs_extras[0]：inputs_time0，（B,P）
        B, P = inputs_extras[0].shape
        _, Q = targets_extras[0].shape
        # 输入的位置编码只用取前P步时间步
        # 使用 .expand 方法将这个部分矩阵扩展为 (B, P, d_model) 形状，
        # 其中 B 表示批处理大小， P 表示序列长度， d_model 表示嵌入维度。
        # 这将复制选定的位置编码矩阵 B 次，以适应当前批次的数据。
        inputs_pe = self.pe[:P, :].expand(B, P, self.d_model)
        targets_pe = self.pe[:Q, :].expand(B, Q, self.d_model)
        # print('inputs_extras_embedding',[self.embedding_modules[i](inputs_extras[i])
        #                                      for i in range(len(self.num_embeddings))])
        # print('targets_extras_embedding',[self.embedding_modules[i](targets_extras[i])
        #                                       for i in range(len(self.num_embeddings))])
        inputs_extras_embedding = torch.cat([self.embedding_modules[i](inputs_extras[i])
                                             for i in range(len(self.num_embeddings))] + [inputs_pe], dim=-1)
        targets_extras_embedding = torch.cat([self.embedding_modules[i](targets_extras[i])
                                              for i in range(len(self.num_embeddings))] + [targets_pe], dim=-1)
        # 拼接后的结果映射回 d_model 维
        inputs_extras_embedding = self.linear(inputs_extras_embedding)
        targets_extras_embedding = self.linear(targets_extras_embedding)

        return inputs_extras_embedding, targets_extras_embedding

# 一层线性变换eigenmaps生成空间属性嵌入，由（N, eigenmaps_k）-> (N, d_model) ，没有用TM
class SpatialEmbedding(nn.Module):
    def __init__(self, eigenmaps_k, d_model):
        super(SpatialEmbedding, self).__init__()
        self.linear = nn.Linear(eigenmaps_k, d_model)
    # 输入eigenmaps：statics['eigenmaps']，（N, eigenmaps_k）
    def forward(self, eigenmaps):
        spatial_embedding = self.linear(eigenmaps)

        return spatial_embedding

# 输出历史交通序列STE (B, P, N, d_model) 和未来交通序列STE (B, Q, N, d_model)
# 融合TE和SE，没有用TM
class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SpatialTemporalEmbedding, self).__init__()
        self.linear = nn.Linear(2 * d_model, d_model)

    def forward(self, z_inputs, z_targets, u):  # (B, P, d_model), (B, Q, d_model), (N, d_model)
        """
        z_inputs: 历史交通序列时间属性信息嵌入TE (B, P, d_model)
        z_targets: 未来交通序列时间属性信息嵌入TE (B, Q, d_model)
        u: 节点嵌入SE (N, d_model)
        输出：历史交通序列STE (B, P, N, d_model) 和未来交通序列STE (B, Q, N, d_model)
        """
        # (z_inputs, ) * len(u)：这部分使用 Python 中的元组扩展操作，len(u)=N
        # 将 z_inputs 复制 len(u) 次，形成一个元组
        # 使用 torch.stack 函数将这个元组中的副本在维度 2 上堆叠。
        # 这将扩展 z_inputs 的维度，使其包含多个相同的副本。
        z_inputs = torch.stack((z_inputs, ) * len(u), dim=2)    # (B, P, N, d_model)
        z_targets = torch.stack((z_targets, ) * len(u), dim=2)  # (B, Q, N, d_model)
        # 维度扩展操作，以确保相同的形状。
        # u_inputs (N, d_model) -> (B, P, N, d_model)
        # u_targets (N, d_model) -> (B, Q, N, d_model)
        u_inputs = u.expand_as(z_inputs)
        u_targets = u.expand_as(z_targets)
        # torch.cat使z_inputs和u_inputs在最后一维拼接，
        # (B, P, N, d_model) -> (B, P, N, 2*d_model)
        # 一层线性变换 (B, P, N, 2*d_model) -> (B, P, N, d_model)
        c_inputs = self.linear(torch.cat((z_inputs, u_inputs), dim=-1))
        c_targets = self.linear(torch.cat((z_targets, u_targets), dim=-1))
        # (B, P, N, d_model), (B, Q, N, d_model)
        return c_inputs, c_targets

# TSA时间多头自注意力，注意力分数采用缩放点积注意力，返回(B, P, N, d_model)
def multihead_temporal_attention(Q, K, V, causal=False):
    """
    Q: (B, P1, N, H, d_k) 在TEDA中的时间注意力机制的Q和K、V来自不同序列
    K: (B, P, N, H, d_k)
    V: (B, P, N, H, d_k)
    """
    B, P1, N, H, d_k = Q.shape
    P = K.shape[1]
    Q = Q.permute((0, 2, 3, 1, 4))  # (B, N, H, P1, d_k)
    # K做了个转置，为了缩放点积注意力中的Q * K^T
    K = K.permute((0, 2, 3, 4, 1))  # (B, N, H, d_k, P)

    scaled_dot_product = torch.matmul(Q, K) / math.sqrt(d_k)  # (B, N, H, P1, P)
    # 掩码（mask）操作，通常在自注意力（self-attention）机制中用于
    # 实现因果（causal）注意力，其中一个位置只能依赖于其前面的位置
    if causal is True:
        assert P1 == P
        # .new_full创建了一个形状为 (P, P) 的张量，其所有值都初始化为负无穷 -np.inf
        #  .triu 方法将张量转换为上三角矩阵，其中对角线以下的所有元素都被设置为负无穷。
        # 这样，掩码将确保每个位置只能关注其前面的位置。
        mask = scaled_dot_product.new_full((P, P), -np.inf).triu(diagonal=1)
        # 将这个因果掩码 mask 加到之前计算的缩放点积注意力 scaled_dot_product 上。
        # 这将在注意力计算中应用掩码，确保每个位置只依赖于其前面的位置，实现因果自注意力。
        scaled_dot_product += mask
    # 注意力权重，(B, N, H, P1, P)
    alpha = F.softmax(scaled_dot_product, dim=-1)

    V = V.permute((0, 2, 3, 1, 4))  # (B, N, H, P, d_k)
    out = torch.matmul(alpha, V)  # (B, N, H, P1, d_k)
    out = out.permute((0, 3, 1, 2, 4))  # (B, P1, N, H, d_k)
    # 把多个头的结果合起来，拼接得(B, P1, N, d_model)
    out = out.reshape((B, P1, N, H * d_k))  # (B, P1, N, H * d_k) i.e. (B, P1, N, d_model)

    return out

# 时间自注意力TSA
class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, noMeta, causal=False):
        super(TemporalSelfAttention, self).__init__()
        self.noMeta = noMeta
        self.causal = causal

        if self.noMeta:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.Linear(d_model, d_model, bias=False)
            self.linear_k = nn.Linear(d_model, d_model, bias=False)
            self.linear_v = nn.Linear(d_model, d_model, bias=False)
        # else:
        #     self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, inputs, c_inputs):
        """
        inputs: 历史交通状态序列 (B, P, N, d_model)
        c_inputs: 历史交通序列的STE (B, P, N, d_model)
        """
        if self.noMeta:
            B, P, N, _ = inputs.shape
            # Q <- X * W_Q，K <- X * W_K，V <- X * W_V
            # d_model分量分给不同的头
            # (B, P, N, d_model) -> (B, P, N, num_heads, d_k)
            # d_k * num_heads = d_model
            Q = self.linear_q(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v(inputs).reshape((B, P, N, self.num_heads, self.d_k))
        # else:
        #     W_q, W_k, W_v = self.meta_learner(c_inputs)  # (B, P, N, H, d_k, d_model)
        #     Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
        #     K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
        #     V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)

        out = multihead_temporal_attention(Q, K, V, causal=self.causal)  # (B, P, N, d_model)
        # 注意力后加一层线性映射到d_model维
        out = self.linear(out)  # (B, P, N, d_model)
        # 残差连接+层归一化
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model)

        return out

# (B, P, N, d_model)，SSA利用TM
def multihead_spatial_attention(Q, K, V, transition_matrix):
    B, P, N, H, d_k = Q.shape

    Q = Q.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    K = K.permute((0, 1, 3, 4, 2))  # (B, P, H, d_k, N)，转置
    V = V.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)

    scaled_dot_product = torch.matmul(Q, K) / math.sqrt(d_k)  # (B, P, H, N, N)
    mask = scaled_dot_product.new_full((N, N), -np.inf)  # (N, N)
    # 防止transition_matrix中的元素被忽略
    mask[transition_matrix != 0] = 0
    scaled_dot_product += mask
    alpha = F.softmax(scaled_dot_product, dim=-1)  # (B, P, H, N, N)
    # 注意力权重先和transition_matrix逐元素相乘，再与V矩阵相乘
    out = torch.matmul(alpha * transition_matrix, V)  # (B, P, H, N, d_k)

    out = out.permute((0, 1, 3, 2, 4))  # (B, P, N, H, d_k)
    out = out.reshape((B, P, N, H * d_k))  # (B, P, N, H * d_k) i.e. (B, P, N, d_model)

    return out

# 空间自注意力，SSA利用TM空间多头自注意力，transition_matrix用于掩码操作和注意力权重相乘，返回
class SpatialSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, noMeta):
        super(SpatialSelfAttention, self).__init__()
        self.which_transition_matrices = which_transition_matrices
        self.num_transition_matrices = sum(which_transition_matrices)   # 3
        assert self.num_transition_matrices > 0
        self.noMeta = noMeta

        if self.noMeta:
            self.num_heads = num_heads  # 4
            self.d_k = d_k  # 4
            self.linear_q = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(self.num_transition_matrices)])
            self.linear_k = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(self.num_transition_matrices)])
            self.linear_v = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(self.num_transition_matrices)])
        # else:
        #     self.meta_learners = nn.ModuleList([MetaLearner(
        #         d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)
        #         for _ in range(self.num_transition_matrices)])

        
        self.linear = nn.Linear(d_model * self.num_transition_matrices, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, inputs, c_inputs, transition_matrices):
        assert transition_matrices.shape[0] == len(self.which_transition_matrices)
        transition_matrices = transition_matrices[self.which_transition_matrices]

        out = []
        # 每种概率矩阵分别求自注意力
        for i in range(self.num_transition_matrices):
            if self.noMeta:
                B, P, N, _ = inputs.shape
                Q = self.linear_q[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
                K = self.linear_k[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
                V = self.linear_v[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            # else:
            #     W_q, W_k, W_v = self.meta_learners[i](c_inputs)  # (B, P, N, H, d_k, d_model)
            #     Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            #     K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            #     V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)

            out.append(multihead_spatial_attention(Q, K, V, transition_matrices[i]))  # (B, P, N, d_model)
        # 对应三种转移矩阵的空间自注意力在最后一维拼接，再线性变换映射回d_model维
        out = torch.cat(out, dim=-1)  # (B, P, N, d_model * num_transition_matrices)
        out = self.linear(out)  # (B, P, N, d_model)
        out = self.dropout(out)
        # 残差连接+层归一化
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model)

        return out

# FF+残差连接+层归一化，输入(B, P, N, d_model)，返回 (B, P, N, d_model)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden_ff, d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, inputs):
        out = self.relu(self.linear1(inputs))
        out = self.linear2(out)
        out = self.layer_norm(out + inputs)

        return out

# 先经过一层时间自注意力层，再经过空间自注意力层，最后FF
class EncoderLayer(nn.Module):
    def __init__(self, cfgs):
        super(EncoderLayer, self).__init__()
        d_model = cfgs['d_model']
        # 4，dimension of Q, K, V，应该是因为P=Q=4
        d_k = cfgs['d_k']
        # in meta learner
        d_hidden_mt = cfgs['d_hidden_mt']
        # in feed forward
        d_hidden_ff = cfgs['d_hidden_ff']
        num_heads = cfgs['num_heads'] # 16/4=4 ，d_k * num_heads = d_model
        # which_transition_matrices: [True, True, True]
        # [connectivity, similarity, correlation]
        which_transition_matrices = cfgs['which_transition_matrices']
        dropout = cfgs['dropout']   # 0.3
        self.noTSA = cfgs.get('noTSA', False)
        self.noSSA = cfgs.get('noSSA', False)
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)
        if self.noTE and self.noSE:
            self.noMeta = True
        # 时间自注意力
        if not self.noTSA:
            self.temporal_self_attention = TemporalSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, self.noMeta, causal=False)
        # 空间自注意力
        if not self.noSSA:
            self.spatial_self_attention = SpatialSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, self.noMeta)
        self.feed_forward = FeedForward(d_model, d_hidden_ff)

    def forward(self, inputs, c_inputs, transition_matrices):
        out = inputs
        # 先经过一层时间自注意力层，再经过空间自注意力层，最后FF
        if not self.noTSA:
            out = self.temporal_self_attention(out, c_inputs)
        if not self.noSSA:
            out = self.spatial_self_attention(out, c_inputs, transition_matrices)
        # 输入(B, P, N, d_model)，返回 (B, P, N, d_model)
        out = self.feed_forward(out)

        return out

class Encoder(nn.Module):
    def __init__(self, cfgs):
        super(Encoder, self).__init__()
        num_features = cfgs['num_features']
        d_model = cfgs['d_model']
        num_encoder_layers = cfgs['num_encoder_layers']
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)

        self.linear = nn.Linear(num_features, d_model)
        self.layer_stack = nn.ModuleList([EncoderLayer(cfgs) for _ in range(num_encoder_layers)])

    def forward(self, inputs, c_inputs, transition_matrices):
        """
        inputs: 历史交通状态序列 (B, P, N, C)
        c_inputs: 历史交通状态序列的STE (B, P, N, d_model)
        transition_matrices: 转移矩阵 (num_TMs, N, N)
        """
        # 一层线性+激活
        # 非元学习，存在时间嵌入或空间嵌入，
        # inputs(B, P, N, C)线性映射后(B, P, N, d_model)
        # 并add时空STE c_inputs，即out (B, P, N, d_model)
        if self.noMeta and ((not self.noTE) or (not self.noSE)):
            out = F.relu(self.linear(inputs) + c_inputs)    # # 将原始特征维度 C 映射到 d_model 维
        else:
            # 将原始特征维度 C 映射到 d_model 维
            out = F.relu(self.linear(inputs))
        skip = 0
        # encoder_layer返回值累加
        for encoder_layer in self.layer_stack:
            # 输入、STE、转移矩阵，输入out (B, P, N, d_model), 输出out (B, P, N, d_model)
            out = encoder_layer(out, c_inputs, transition_matrices)
            skip += out
        # 每层encoder_layer返回out累加，最终生成encoder_output (B, P, N, d_model)
        return skip

# TEDA时间encoder-decoder自注意力+线性层+残差连接+层归一化，输入输出均是(B, P1, N, d_model)
# TEDA是为了解码器创建的，键值来自编码器，查询来自解码器的上一步输出，沿时间维度执行TSA
class TemporalEncoderDecoderAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, noMeta):
        super(TemporalEncoderDecoderAttention, self).__init__()
        self.noMeta = noMeta

        if self.noMeta:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.Linear(d_model, d_model, bias=False)
        # else:
        #     self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=1)

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, inputs, enc_K, enc_V, c_targets):
        """
        inputs: (B, 1, N, d_model)， P1=1 
        enc_K: (B, P, N, H, d_k)
        enc_V: (B, P, N, H, d_k)
        c_targets: (B, 1, N, d_model)
        """
        if self.noMeta:
            B, P1, N, _ = inputs.shape  # P1=1，P1不等于P、Q
            Q = self.linear_q(inputs).reshape((B, P1, N, self.num_heads, self.d_k))
        # else:
        #     W_q, = self.meta_learner(c_targets)  # (B, P1, N, H, d_k, d_model)
        #     Q = multihead_linear_transform(W_q, inputs)  # (B, P1, N, H, d_k)

        out = multihead_temporal_attention(Q, enc_K, enc_V, causal=False)  # (B, P1, N, d_model)
        out = self.linear(out)  # (B, P1, N, d_model)
        out = self.layer_norm(out + inputs)  # (B, P1, N, d_model)

        return out

# 时间自注意力+空间自注意力+时间encoder-decoder自注意力+FF，输入输出均为(B, P1, N, d_model)
class DecoderLayer(nn.Module):
    def __init__(self, cfgs):
        super(DecoderLayer, self).__init__()
        d_model = cfgs['d_model']
        d_k = cfgs['d_k']
        d_hidden_mt = cfgs['d_hidden_mt']
        d_hidden_ff = cfgs['d_hidden_ff']
        num_heads = cfgs['num_heads']
        which_transition_matrices = cfgs['which_transition_matrices']
        dropout = cfgs['dropout']
        self.noTSA = cfgs.get('noTSA', False)
        self.noSSA = cfgs.get('noSSA', False)
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)
        if self.noTE and self.noSE:
            self.noMeta = True

        if not self.noTSA:
            self.temporal_self_attention = TemporalSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, self.noMeta, causal=True)
        if not self.noSSA:
            self.spatial_self_attention = SpatialSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, self.noMeta)
        self.temporal_encoder_decoder_attention = TemporalEncoderDecoderAttention(
            d_model, d_k, d_hidden_mt, num_heads, self.noMeta)
        self.feed_forward = FeedForward(d_model, d_hidden_ff)

    def forward(self, inputs, enc_K, enc_V, c_targets, transition_matrices):
        """
        inputs: (B, 1, N, d_model)
        enc_K: (B, P, N, H, d_k)
        enc_V: (B, P, N, H, d_k)
        c_targets: (B, 1, N, d_model)
        transition_matrices: (num_TMs, N, N)
        """
        out = inputs
        if not self.noTSA:
            out = self.temporal_self_attention(out, c_targets)
        if not self.noSSA:
            out = self.spatial_self_attention(out, c_targets, transition_matrices)
        out = self.temporal_encoder_decoder_attention(out, enc_K, enc_V, c_targets)
        out = self.feed_forward(out)

        return out

# 线性映射回最初的num_features，2
class Project(nn.Module):
    def __init__(self, d_model, num_features):
        super(Project, self).__init__()
        self.linear = nn.Linear(d_model, num_features)

    def forward(self, inputs):
        out = self.linear(inputs)

        return out

class Decoder(nn.Module):
    def __init__(self, cfgs):
        super(Decoder, self).__init__()
        d_model = cfgs['d_model']
        d_k = cfgs['d_k']
        d_hidden_mt = cfgs['d_hidden_mt']
        num_features = cfgs['num_features']
        num_heads = cfgs['num_heads']
        num_decoder_layers = cfgs['num_decoder_layers']
        self.out_len = cfgs['out_len']
        self.use_curriculum_learning = cfgs['use_curriculum_learning']
        self.cl_decay_steps = cfgs['cl_decay_steps']
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)

        if self.noMeta or (self.noTE and self.noSE):
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_k = nn.Linear(d_model, d_model, bias=False)
            self.linear_v = nn.Linear(d_model, d_model, bias=False)
        # else:
        #     self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=2)

        self.linear = nn.Linear(num_features, d_model)
        self.layer_stack = nn.ModuleList([DecoderLayer(cfgs) for _ in range(num_decoder_layers)])
        self.project = Project(d_model, num_features)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, inputs, targets, c_inputs, c_targets, enc_outputs, transition_matrices, batches_seen):
        """
        inputs: 历史交通状态序列 (B, P, N, C)
        targets: 未来交通状态序列 (B, Q, N, C)
        c_inputs: 历史交通序列的STE (B, P, N, d_model)
        c_targets: 未来交通序列的STE (B, Q, N, d_model)
        enc_outputs: encoder输出 (B, P, N, d_model)
        transition_matrices: (num_TMs, N, N)
        batches_seen:
        """
        # 非元学习或时空嵌入存在
        if self.noMeta or (self.noTE and self.noSE):
            B, P, N, _ = enc_outputs.shape
            # 键、值来自编码器，分头
            enc_K = self.linear_k(enc_outputs).reshape((B, P, N, self.num_heads, self.d_k))
            enc_V = self.linear_v(enc_outputs).reshape((B, P, N, self.num_heads, self.d_k))
        # else:
        #     W_k, W_v = self.meta_learner(c_inputs)  # (B, P, N, H, d_k, d_model)
        #     enc_K = multihead_linear_transform(W_k, enc_outputs)  # (B, P, N, H, d_k)
        #     enc_V = multihead_linear_transform(W_v, enc_outputs)  # (B, P, N, H, d_k)

        use_targets = False
        if self.training and (targets is not None) and self.use_curriculum_learning:
            c = np.random.uniform(0, 1)
            if c < self._compute_sampling_threshold(batches_seen):
                use_targets = True
        
        if use_targets is True:
            # 构造编码器的输入序列
            # inputs[:, -1, :, :].unsqueeze(1)只选择inputs最后一个时间步
            # .unsqueeze(1)方法在维度1上增加维度。形状变为 (B, 1, N, C)
            # targets[:, :-1, :, :]：选择targets除最后一个时间步之外的所有时间步的数据。
            # 两者在维度1拼接，生成的dec_inputs仍 (B, Q, N, C)
            dec_inputs = torch.cat((inputs[:, -1, :, :].unsqueeze(1), targets[:, :-1, :, :]), dim=1)  # (B, Q, N, C)
            if self.noMeta and ((not self.noTE) or (not self.noSE)):
                out = F.relu(self.linear(dec_inputs) + c_targets)
            else:
                out = F.relu(self.linear(dec_inputs))  # (B, Q, N, d_model)
            skip = 0
            for decoder_layer in self.layer_stack:
                out = decoder_layer(out, enc_K, enc_V, c_targets, transition_matrices)
                skip += out
            outputs = self.project(skip)  # (B, Q, N, C)
        # 不考虑use_curriculum_learning
        else:   # 自回归方式预测
            # 取inputs的最后一个
            dec_inputs = inputs[:, -1, :, :].unsqueeze(1)  # (B, 1, N, C)
            outputs = []
            # 一共预测out_len=1个时间步
            for i in range(self.out_len):
                if self.noMeta and ((not self.noTE) or (not self.noSE)):
                    out = F.relu(self.linear(dec_inputs) + c_targets[:, :(i + 1), :, :])
                else:
                    out = F.relu(self.linear(dec_inputs))  # (B, *, N, d_model)
                skip = 0
                for decoder_layer in self.layer_stack:
                    if (not self.noTE) or (not self.noSE):
                        # (B, 1, N, d_model)
                        out = decoder_layer(out, enc_K, enc_V, c_targets[:, :(i + 1), :, :], transition_matrices)
                    else:
                        out = decoder_layer(out, enc_K, enc_V, None, transition_matrices)
                    skip += out
                out = self.project(skip)  # (B, *, N, C)
                # 每次记录当前解码器输出的最后一个时间步
                outputs.append(out[:, -1, :, :])
                # 旧的dec_inputs (B, 1, N, C)，out每次只取解码器输出的最后一个时间步，新的dec_inputs (B, 2, N, C)，即P1会变
                dec_inputs = torch.cat((dec_inputs, out[:, -1, :, :].unsqueeze(1)), dim=1)
            outputs = torch.stack(outputs, dim=1)  # (B, Q, N, C)

        return outputs

class MGT(nn.Module):
    def __init__(self, cfgs):
        super(MGT, self).__init__()
        # feature size of our model， d_k * num_heads = d_model
        d_model = cfgs['d_model']
        # 表示时间特征的高维矩阵，原文为[73, 2]  73代表一天73个时间间隔，2代表[time of day, rest]
        num_embeddings = cfgs['num_embeddings']
        # 邻接图or相似性图的拉普拉斯矩阵SVD分解取前k特征值对应的特征向量，达到降维，原文为8
        eigenmaps_k = cfgs['eigenmaps_k']
        # 输入长度（输入的时间步总数）
        self.in_len = cfgs['in_len']
        # 预测长度（预测的时间步总数）
        self.out_len = cfgs['out_len']
        # 最长时间步的长度
        max_len = max(self.in_len, self.out_len)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)
        self.batches_seen = 0

        # 初始化时间嵌入TemporalEmbedding
        if not self.noTE:
            self.temporal_embedding = TemporalEmbedding(num_embeddings, d_model, max_len)
        # 初始化空间嵌入SpatialEmbedding
        if not self.noSE:
            self.spatial_embedding = SpatialEmbedding(eigenmaps_k, d_model)
        # 时间、空间嵌入都要的话，初始化时空嵌入SpatialTemporalEmbedding
        if (not self.noTE) and (not self.noSE):
            self.spatial_temporal_embedding = SpatialTemporalEmbedding(d_model)

        self.encoder = Encoder(cfgs)
        self.decoder = Decoder(cfgs)

    def forward(self, inputs, targets, *extras, **statics):
        if not self.noTE:
            # 历史和未来交通序列的时间属性信息 extras = [inputs_time0, targets_time0, inputs_time1, targets_time1]
            z_inputs, z_targets = self.temporal_embedding(extras)  # (B, P, d_model), (B, Q, d_model)
        if not self.noSE:
            # statics = {'eigenmap，s': eigenmaps, 'transition_matrices': transition_matrices}
            # eigenmaps：（N, eigenmaps_k）即（80，8）
            u = self.spatial_embedding(statics['eigenmaps'])  # (N, d_model)
        if (not self.noTE) and (not self.noSE):
            # 传入时间、空间嵌入，生成历史（输入）和未来（输出）的时空融合嵌入STE，即c_inputs、c_targets
            c_inputs, c_targets = self.spatial_temporal_embedding(
                z_inputs, z_targets, u)  # (B, P, N, d_model), (B, Q, N, d_model)
        # 不要时间嵌入，只要空间嵌入
        elif self.noTE and (not self.noSE):
            B = inputs.size(0)
            P = self.in_len
            Q = self.out_len
            N = u.size(0)
            d_model = u.size(1)
            # 直接扩展空间嵌入维度，
            # 输入(B, P, N, d_model), 目标(B, Q, N, d_model)
            c_inputs = u.expand(B, P, N, d_model)
            c_targets = u.expand(B, Q, N, d_model)
        # 不要空间嵌入，只要时间嵌入 
        elif (not self.noTE) and self.noSE:
            N = inputs.size(2)
            # (B, P, d_model)、(B, Q, d_model) -> (B, P, N, d_model)、(B, Q, N, d_model)
            c_inputs = torch.stack((z_inputs,) * N, dim=2)
            c_targets = torch.stack((z_targets,) * N, dim=2)
        else:
            c_inputs = None
            c_targets = None
        # (n,N,N), n=3是因为3种关系矩阵，[connectivity, similarity, correlation]
        transition_matrices = statics['transition_matrices']
        # encoder输入inputs(B, P, N, C)、c_inputs(B, P, N, d_model)
        # 、transition_matrices（n,N,N）
        # 多层encoder_layer返回值得累加，输出enc_outputs (B, P, N, d_model)
        enc_outputs = self.encoder(inputs, c_inputs, transition_matrices)
        # decoder输入inputs(B, P, N, C)、c_inputs(B, P, N, d_model)
        # 、c_targets(B, Q, N, d_model)、enc_outputs (B, P, N, d_model)
        # 、transition_matrices（n,N,N）
        # 输出outputs (B, Q, N, C)
        outputs = self.decoder(inputs, targets, c_inputs, c_targets, enc_outputs, transition_matrices,
                               self.batches_seen)
        # 一次模型调用对应一次batch
        if self.training:
            self.batches_seen += 1

        return outputs


# if __name__ == '__main__':
#     cfgs = yaml.safe_load(open('cfgs/Qtraffic.yaml'))['model']
#     model = MGT(cfgs)

#     # dummy data
#     # C是feature_num
#     B, P, Q, N, C = 10, 1, 1, 1448, 1
#     M = 96, 2, 2, 2, 24, 60, 2
#     eigenmaps_k = 8
#     n = 2 #transition_matrices: [connectivity, similarity]

#     inputs = torch.randn(B, P, N, C, dtype=torch.float32)
#     targets = torch.randn(B, Q, N, C, dtype=torch.float32)

#     inputs_time0 = torch.randint(M[0], (B, P), dtype=torch.int64)
#     targets_time0 = torch.randint(M[0], (B, Q), dtype=torch.int64)
#     inputs_time1 = torch.randint(M[1], (B, P), dtype=torch.int64)
#     targets_time1 = torch.randint(M[1], (B, Q), dtype=torch.int64)

#     eigenmaps = torch.randn(N, eigenmaps_k, dtype=torch.float32)

#     transition_matrices = torch.rand(n, N, N, dtype=torch.float32)

#     extras = [inputs_time0, targets_time0, inputs_time1, targets_time1]
#     statics = {'eigenmaps': eigenmaps, 'transition_matrices': transition_matrices}

#     # forward
#     outputs1 = model(inputs, targets, *extras, **statics)
#     outputs2 = model(inputs, None, *extras, **statics)