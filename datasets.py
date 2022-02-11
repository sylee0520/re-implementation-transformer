import torch
from tokenizer import Tokenizer
from torch.utils.data import DataLoader


class Dataset():
    def __init__(self, src, trg, data_type, tokenizer, max_seq_len) -> None:

        with open(file=f'./iwslt14.tokenized.de-en/tmp/{data_type}.{src}') as f:
            self.input = f.readlines()
        
        with open(file=f'./iwslt14.tokenized.de-en/tmp/{data_type}.{trg}') as f:
            self.output = f.readlines()
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.len_dataset = len(self.input)

    def __getitem__(self, index):
        inputs = self.input[index]
        inputs = torch.tensor(self.tokenizer.encode(inputs), dtype=torch.float64, requires_grad=True).long()
        outputs = self.output[index]
        outputs = torch.tensor(self.tokenizer.encode(outputs), dtype=torch.float64, requires_grad=True).long()

        return inputs, outputs
    
    def __len__(self):
        return self.len_dataset
    
    def collate_fn(self, data):

        def merge(sequences):
            padded_sequences = torch.zeros(len(sequences), self.max_seq_len, requires_grad=True).long()
            if torch.cuda.is_available():
                padded_sequences = padded_sequences.to('cuda')
            
            for i, seq in enumerate(sequences):
                padded_sequences[i][:min(self.max_seq_len, len(seq))] = seq[:min(self.max_seq_len, len(seq))]

            return padded_sequences
        
        data.sort(key=lambda x: len(x[0]), reverse=True)

        src_seqs, trg_seqs = zip(*data)

        src_seqs = merge(src_seqs)
        trg_seqs = merge(trg_seqs)

        return src_seqs, trg_seqs


def test():
    tokenizer = Tokenizer().load_model()
    dataset = Dataset(src='de', trg='en', data_type='train', max_seq_len=20, tokenizer=tokenizer)
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)

    for epoch in range(10):
        for i, data in enumerate(train_loader):                
            inputs, outputs = data
            print(f"inputs size: {inputs.shape}")
            print(f"outputs size: {outputs.shape}")
            break

if __name__ == "__main__":
    test()
    