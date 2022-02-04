import torch
import torch.nn as nn
from collections import OrderedDict

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

        # 미완성
        if layer_mode == 'linear':
            self.linear_weight = torch.transpose(self.embedding.weight, 0, 1) 
            self.linear = nn.Embedding(num_embeddings=embedding_dim, 
                                       embedding_dim=num_embeddings,
                                       padding_idx=padding_idx)

            print("before\n", self.linear.weight)
            self.linear.parameters = self.linear_weight
            print("after\n", self.linear.weight)

    def forward(self, inputs):
        # embedding mode (16000d -> 512d)
        if self.layer_mode == 'embedding':
            output = self.embedding(inputs)

            # sqrt(embedding_dim=512)로 scaling
            output *= torch.sqrt(self.embedding_dim)
            return output
        
        # pre-softmax linear transformation mode(512d -> 16000d)
        elif self.layer_mode == 'linear':
            output = self.linear(inputs)

            return output
            
# padding index를 어떻게 처리할 지 생각하기
embedding_layer = EmbeddingLayer(num_embeddings=16000, embedding_dim=512, padding_idx=10, layer_mode='linear')
# sentences = torch.LongTensor([[1,3,4,1], [3,4,5,6]])
# print(sentences)
# outputs = embedding_layer(sentences)
# print(outputs.shape)
# print(outputs)
# print(embedding_layer.parameters)