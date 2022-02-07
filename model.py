import torch
import torch.nn as nn

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
            

# class PositionalEncodingLayer(nn.modules):
#     pass


# class Encoder(nn.Module):
#     pass


# class Decoder(nn.Module):
#     pass


# class MultiHeadAttentionLayer(nn.Module):
#     pass


# class FeedForwardLayer(nn.Module):
#     pass





