import torch
import torch.nn as nn
import numpy as np

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, layer_mode='embed') -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.layer_mode = layer_mode

        self.linear_weight = torch.transpose(self.embedding.weight, 0, 1) 
        self.linear = nn.Embedding(num_embeddings=embedding_dim, 
                                    embedding_dim=num_embeddings,
                                    padding_idx=padding_idx)

        self.linear.weight.data = self.linear_weight

    def forward(self, inputs):
        # embedding mode (16000d -> 512d)
        if self.layer_mode == 'embedding':
            output = self.embedding(inputs)

            # scaling with sqrt(embedding_dim=512)
            output *= torch.sqrt(self.embedding_dim)
            return output
        
        # pre-softmax linear transformation mode(512d -> 16000d)
        elif self.layer_mode == 'linear':
            output = self.linear(inputs)

            return output
            

class PositionalEncodingLayer(nn.modules):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, n_position, model_dim):
        position_encoding = [[position / np.power(10000, 2*i/model_dim) for i in range(model_dim)] for position in range(n_position)]
        position_encoding = np.array(position_encoding)

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        position_encoding = torch.FloatTensor(position_encoding)

        if torch.cuda.is_available():
            position_encoding = position_encoding.to('cuda')

        return position_encoding


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, model_dim, q_dim, k_dim, v_dim, n_heads):
        super().__init__()
        self.model_dim = model_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_heads = n_heads

        self.wq = nn.Linear(model_dim, n_heads * q_dim, bias=False)
        self.wk = nn.Linear(model_dim, n_heads * k_dim, bias=False)
        self.wv = nn.Linear(model_dim, n_heads * v_dim, bias=False)
        self.wo = nn.Linear(n_heads * v_dim, model_dim, bias=False)

        self.ScaledDotProductAttention = ScaledDotProductAttentionLayer(k_dim)

    def forward(self, q, k, v):

        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.wq(q).view(batch_size, len_q, self.n_heads, self.q_dim)
        k = self.wk(k).view(batch_size, len_k, self.n_heads, self.k_dim)
        v = self.wv(v).view(batch_size, len_v, self.n_heads, self.v_dim)

        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)

        attention = self.ScaledDotProductAttention(q, k, v)

        attention = torch.transpose(attention, 1, 2)
        attention = attention.view(batch_size, len_q, -1)

        out = self.wo(attention)

        return out

        

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()

        self.w1 = nn.Linear(in_dim, hid_dim)
        self.w2 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.w2(self.relu(self.w1(x)))

        return out


class ScaledDotProductAttentionLayer(nn.Module):
    def __init__(self, k_dim, mask=False):
        super().__init__()
        self.k_dim = k_dim
        self.mask = mask
    
    def forward(self, q, k, v):
        attention = torch.matmul(q, torch.transpose(k, 2, 3))
        attention = attention / torch.sqrt(self.k_dim)

        if self.mask is True:
            attention.masked_fill(self.mask, 1e-9)

        attention = torch.softmax(attention, -1)
        attention = torch.matmul(attention, v)

        return attention



class EncoderLayer(nn.Module):
    def __init__(self, model_dim, q_dim, k_dim, v_dim, n_heads, in_dim, hid_dim, out_dim):
        super().__init__()

        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

        self.MultiHeadAttention = MultiHeadAttentionLayer(model_dim, q_dim, k_dim, v_dim, n_heads)
        self.PositionWiseFeedForward = PositionWiseFeedForwardLayer(in_dim, hid_dim, out_dim)


    def forward(self, x):
        residual = x

        q = k = v = x
        attention = self.MultiHeadAttention(q, k, v)

        out = self.dropout(attention)
        out += residual
        out = self.layer_norm(out)

        residual = out
        out = self.PositionWiseFeedForward(out)

        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, q_dim, k_dim, v_dim, n_heads, in_dim, hid_dim, out_dim):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

        self.MultiHeadAttention = MultiHeadAttentionLayer(model_dim, q_dim, k_dim, v_dim, n_heads)
        self.MaskedMultiHeadAttention = MultiHeadAttentionLayer(model_dim, q_dim, k_dim, v_dim, n_heads, mask=True)
        self.PositionWiseFeedForward = PositionWiseFeedForwardLayer(in_dim, hid_dim, out_dim)
    
    def forward(self, decoder_input, encoder_output):
        residual = decoder_input

        q = k = v = decoder_input
        attention = self.MaskedMultiHeadAttention(q, k, v)

        v = self.dropout(attention)
        v += residual
        v = self.layer_norm(v)

        residual = v

        q, k = encoder_output, encoder_output
        attention = self.MultiHeadAttention(q, k, v)

        out = self.dropout(attention)
        out += residual
        out = self.layer_norm(out)

        residual = out

        out = self.PositionWiseFeedForward(out)

        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)






