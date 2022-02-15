import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter

from model import *
from tokenizer import Tokenizer
from datasets import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
summary = SummaryWriter()

def get_args():
    # arguments parser
    parser = argparse.ArgumentParser()
    
    # argments
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--src', default='de', type=str)
    parser.add_argument('--trg', default='en', type=str)
    parser.add_argument('--max_seq_len', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--vocab_size', default=16000, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--model_name', default='model', type=str)

    args = parser.parse_args()

    return args


def main(args):
    if args.mode == 'train':
        # 1. Load tokenizer
        tokenizer = Tokenizer().load_model()

        # 2. Load dataset
        # train dataset
        train_dataset = Dataset(src=args.src, trg=args.trg, data_type='train', tokenizer=tokenizer, max_seq_len=args.max_seq_len)

        # validation dataset
        val_dataset = Dataset(src=args.src, trg=args.trg, data_type='valid', tokenizer=tokenizer, max_seq_len=args.max_seq_len)

        # 3. Load dataloader
        # train dataloader
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)

        # validation dataloader
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

        # 4. Load transformer
        # Transformer(num_embeddings=16000, embedding_dim=512, padding_idx=0, 
        #             model_dim=512, q_dim=64, k_dim=64, v_dim=64, n_heads=8,
        #             in_dim=512, hid_dim=2048, out_dim=512, max_seq_len=20, N=6)
        model = Transformer(num_embeddings=args.vocab_size, max_seq_len=args.max_seq_len)
        
        if torch.cuda.is_available():
            model.to('cuda')

        # 5. Load criterion
        # Softmax + NLLLoss -> CrossEntropyLoss
        # Since model includes final softmax layer, I'll use NLLLoss
        criterion = nn.NLLLoss(ignore_index=0)

        # 6. Load optimizer
        # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)

        # 7. Train transformer
        EPOCH = args.epoch
        train_step = val_step = 0
        best_loss = float("inf")

        for epoch in range(EPOCH):
            train_loss = val_loss = 0
            train_total = val_total = 0

            # train
            model.train()
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'EPOCH {epoch}'):
                inputs, outputs = data
                targets = outputs

                output_probabilities = model(inputs, outputs)

                loss = criterion(output_probabilities.view(-1, args.vocab_size), targets.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_total += 1

                if (train_step + 1) % 10 == 0:
                    summary.add_scalar('loss/train_loss', loss.item(), train_step)
                
                train_step += 1
            
            train_loss /= train_total

            # val
            model.val()
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'EPOCH {epoch}'):
                    inputs, outputs = data
                    targets = outputs
                    
                    output_probabilities = model(inputs, outputs)
                    loss = criterion(output_probabilities.view(-1, args.vocab_size), targets.view(-1))
                    val_loss += loss.item()
                    val_total += 1

                    if (val_step + 1) % 10 == 0:
                        summary.add_scalar('loss/val_loss', loss.item(), val_step)
                        print(f"Golden Sequence: {tokenizer.decode(outputs.tolist())[0]}")
                        print(f"Generated Sequence: {tokenizer.decode(torch.argmax(output_probabilities, -1).tolist())[0]}")
                    val_step += 1

                val_loss /= val_total
            
            # results
            print("EPOCH: {}/{} Train_Loss: {:.3f} Val_Loss: {:.3f}".format(epoch+1, EPOCH, train_loss, val_loss))

            # save model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"outputs/{args.model_name}-{epoch}.pt")
                print("model saved!")


    elif args.mode == 'val':
        pass



