import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import EmbeddingLayer
from tokenizer import Tokenizer
from datasets import Dataset

tokenizer = Tokenizer().load_model()
dataset = Dataset(src='de', trg='en', data_type='train', tokenizer=tokenizer, max_seq_len=20)
train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)
embed = EmbeddingLayer(num_embeddings=16000, embedding_dim=512, padding_idx=0, layer_mode='embedding')

for epoch in range(10):
    for i, data in enumerate(train_loader):
        inputs, outputs = data
        print("input shape: ", inputs.shape)
        temp = embed(inputs)
        print("output shape: ", temp.shape)
        break




