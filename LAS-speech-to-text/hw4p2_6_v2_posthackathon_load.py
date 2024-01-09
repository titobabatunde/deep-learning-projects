# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchaudio
from torch import nn, Tensor
# import torchsummary

import numpy as np

import gc
import time

import pandas as pd
from tqdm.notebook import tqdm as blue_tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as tat
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from datetime import datetime
import random
import csv
import seaborn
import json
from tqdm import tqdm
import wandb
import subprocess

import math
from typing import Optional, List
"""
added weight initialization for LSTM Cell
added augmentation
"""

# %%
#imports for decoding and distance calculation
try:
    import wandb
    import torchsummaryX
    import Levenshtein
except:
    print("Didnt install some/all imports")

import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

# %%
"""# Config"""

config = dict (
    train_dataset       = 'train-clean-460', #train-clean-360, train-clean-460
    batch_size          = 32, # 64
    epochs              = 150,
    learning_rate       = 1e-4,
    weight_decay        = 1e-6,
    cepstral_norm       = True
)


# %%
VOCAB = [
    '<pad>', '<sos>', '<eos>',
    'A',   'B',    'C',    'D',
    'E',   'F',    'G',    'H',
    'I',   'J',    'K',    'L',
    'M',   'N',    'O',    'P',
    'Q',   'R',    'S',    'T',
    'U',   'V',    'W',    'X',
    'Y',   'Z',    "'",    ' ',
]

VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}

PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]

print(f"Length of vocab : {len(VOCAB)}")
print(f"Vocab           : {VOCAB}")
print(f"PAD_TOKEN       : {PAD_TOKEN}")
print(f"SOS_TOKEN       : {SOS_TOKEN}")
print(f"EOS_TOKEN       : {EOS_TOKEN}")

# %%
class SpeechDatasetME(torch.utils.data.Dataset): # Memory efficient
    # Loades the data in get item to save RAM

    def __init__(self, root, partition= "train-clean-360", transforms = None, cepstral=True):
        # transforms is for data transformations

        self.VOCAB      = VOCAB_MAP
        self.cepstral   = cepstral

        if partition == "train-clean-100" or partition == "train-clean-360" or partition == "dev-clean":
            # path to the mfccs
            mfcc_dir       = os.path.join(root, partition, 'mfcc')
            # path to the transcripts
            transcript_dir = os.path.join(root, partition, 'transcripts')

            # create a list of paths for all the mfccs in the mfcc directory
            mfcc_files          = [os.path.join(mfcc_dir, file) for file in sorted(os.listdir(mfcc_dir))] # X
            # create a list of paths for all the transcripts in the transcript directory
            transcript_files    = [os.path.join(transcript_dir, file) for file in sorted(os.listdir(transcript_dir))] # Y 

        else: # test-clean-460 this is ALL or dev-clean
            # path to the mfccs in the train clean 100 partition
            partition1 = "train-clean-100"
            mfcc_dir       = os.path.join(root, partition1, 'mfcc')
            # path to the transcripts in the train clean 100 partition
            transcript_dir = os.path.join(root, partition1, 'transcripts')

            # create a list of paths for all the mfccs in the mfcc directory
            mfcc_files          = [os.path.join(mfcc_dir, file) for file in sorted(os.listdir(mfcc_dir))] # X
            # create a list of paths for all the transcripts in the transcript directory
            transcript_files    = [os.path.join(transcript_dir, file) for file in sorted(os.listdir(transcript_dir))] # Y

            partition2 = "train-clean-360"
            # path to the mfccs in the train clean 360 partition
            mfcc_dir       = os.path.join(root, partition2, 'mfcc')
            # path to the transcripts in the train clean 100 partition
            transcript_dir = os.path.join(root, partition2, 'transcripts')

            # add the list of mfcc and transcript paths from train-clean-360 to the list of 
            # paths  from train-clean-100
            mfcc_files.extend([os.path.join(mfcc_dir, file) for file in sorted(os.listdir(mfcc_dir))]) # X
            transcript_files.extend([os.path.join(transcript_dir, file) for file in sorted(os.listdir(transcript_dir))]) # Y

        assert len(mfcc_files) == len(transcript_files)
        self.mfcc_files         = mfcc_files
        self.transcript_files   = transcript_files
        self.length             = len(transcript_files)
        print("Loaded file paths ME: ", partition)
    # end def

    def __len__(self):
        return self.length
    # end def

    def __getitem__(self, ind):

        # Load the mfcc and transcripts from the mfcc and transcript paths created earlier
        mfcc        = self.mfcc_files[ind]
        transcript  = self.transcript_files[ind]

        # Normalize the mfccs and map the transcripts to integers
        # cepstral normalization
        mfcc                =  np.load(mfcc)
        if self.cepstral:
            mfcc                -= np.mean(mfcc, axis=0)
            mfcc                /= np.std(mfcc, axis=0) 

        transcript          =  np.load(transcript)
        transcript_mapped   = np.array([self.VOCAB[i] for i in transcript])

        return torch.FloatTensor(mfcc), torch.LongTensor(transcript_mapped)

    def collate_fn(self,batch):

        batch_x, batch_y, lengths_x, lengths_y = [], [], [], []

        for x, y in batch:
            # Add the mfcc, transcripts and their lengths to the lists created above
            batch_x.append(x)
            batch_y.append(y)
            lengths_x.append(x.shape[0])
            lengths_y.append(y.shape[0])
        # end for

        # pack the mfccs and transcripts using the pad_sequence function from pytorch
        batch_x_pad = pad_sequence(batch_x, batch_first=True, padding_value=PAD_TOKEN)
        batch_y_pad = pad_sequence(batch_y, batch_first=True, padding_value=PAD_TOKEN)

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


# %%
class SpeechDatasetTest(torch.utils.data.Dataset):

    def __init__(self, root, partition="test-clean", cepstral=False):
        self.cepstral = cepstral
        # path to the test-clean mfccs
        self.mfcc_dir   = os.path.join(root,partition,'mfcc')
        # list files in the mfcc directory
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) # X

        self.mfccs = []
        for i, filename in enumerate(blue_tqdm(self.mfcc_files)):
            # load the mfccs
            mfcc = np.load(os.path.join(self.mfcc_dir, filename))
            if cepstral:
                # Normalize the mfccs
                mfcc -= np.mean(mfcc, axis=0)
                mfcc /= np.std(mfcc, axis=0)
            # append the mfcc to the mfcc list created earlier
            self.mfccs.append(mfcc)
            self.length = len(self.mfccs)

        print("Loaded: ", partition)
    # end def

    def __len__(self):
        return self.length
    # end def

    def __getitem__(self, ind):
        return torch.FloatTensor(self.mfccs[ind])
    # end def

    def collate_fn(self,batch):

        batch_x, lengths_x = [], []
        for x in batch:
            # Append the mfccs and their lengths to the lists created above
            batch_x.append(x)
            lengths_x.append(x.shape[0])

        # pack the mfccs using the pad_sequence function from pytorch
        batch_x_pad = pad_sequence(batch_x, batch_first=True, padding_value=PAD_TOKEN)

        return batch_x_pad, torch.tensor(lengths_x)
    # end def
# end class

# %%
DATA_DIR        = 'data/11-785-f23-hw4p2'
PARTITION       = config['train_dataset']
CEPSTRAL        = config['cepstral_norm']

# train_dataset   = SpeechDatasetME( # Or AudioDatasetME
#     root        = DATA_DIR,
#     partition   = PARTITION,
#     cepstral    = CEPSTRAL
# )
valid_dataset   = SpeechDatasetME(
    root        = DATA_DIR,
    partition   = 'dev-clean',
    cepstral    = CEPSTRAL
)
test_dataset    = SpeechDatasetTest(
    root        = DATA_DIR,
    partition   = 'test-clean',
    cepstral    = CEPSTRAL,
)

# %%
gc.collect()
# train_loader    = DataLoader(
#     dataset     = train_dataset,
#     batch_size  = config['batch_size'],
#     shuffle     = True,
#     num_workers = 3, # 4
#     pin_memory  = False, # True
#     collate_fn  = train_dataset.collate_fn
# )

valid_loader    = DataLoader(
    dataset     = valid_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True, # True
    collate_fn  = valid_dataset.collate_fn
)

test_loader     = DataLoader(
    dataset     = test_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn
)

# print("No. of train mfccs   : ", train_dataset.__len__())
print("Batch size           : ", config['batch_size'])
# print("Train batches        : ", train_loader.__len__())
print("Valid batches        : ", valid_loader.__len__())
print("Test batches         : ", test_loader.__len__())

# %%
# print("\nChecking the shapes of the data...")
# for batch in train_loader:
#     x, y, x_len, y_len = batch
#     print(x.shape, y.shape, x_len.shape, y_len.shape)
#     print(y)
#     break

# %%
# def verify_dataset(dataset, partition= 'train-clean-100'):
#     print("\nPartition loaded     : ", partition)
#     if partition != 'test-clean':
#         print("Max mfcc length          : ", np.max([data[0].shape[0] for data in dataset]))
#         print("Avg mfcc length          : ", np.mean([data[0].shape[0] for data in dataset]))
#         print("Max transcript length    : ", np.max([data[1].shape[0] for data in dataset]))
#         print("Max transcript length    : ", np.mean([data[1].shape[0] for data in dataset]))
#     else:
#         print("Max mfcc length          : ", np.max([data.shape[0] for data in dataset]))
#         print("Avg mfcc length          : ", np.mean([data.shape[0] for data in dataset]))

# verify_dataset(train_dataset, partition= 'train-clean-100')
# verify_dataset(valid_dataset, partition= 'dev-clean')
# verify_dataset(test_dataset, partition= 'test-clean')
# dataset_max_len  = max(
#     np.max([data[0].shape[0] for data in train_dataset]),
#     np.max([data[0].shape[0] for data in valid_dataset]),
#     np.max([data.shape[0] for data in test_dataset])
# )
# print("\nMax Length: ", dataset_max_len)

# %%
"""
## Utils
"""
class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)
    # end def
# end class

def plot_attention(attention, epoch_num):
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    seaborn.heatmap(attention, cmap='GnBu')
    plt.savefig('attention-'+str(epoch_num)+'.svg')
    plt.show()
# end def

# TRAINING SETUP
def save_model(model, optimizer, scheduler, tf_scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         'tf_scheduler'             : tf_scheduler,
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )

def load_model(best_path, epoch_path, model, mode= 'best', metric= 'valid_acc', optimizer= None, scheduler= None, tf_scheduler= None):

    if mode == 'best':
        checkpoint  = torch.load(best_path)
        print("Loading best checkpoint: ", checkpoint[metric])
    else:
        checkpoint  = torch.load(epoch_path)
        print("Loading epoch checkpoint: ", checkpoint[metric])

    model.load_state_dict(checkpoint['model_state_dict'], strict= False)

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #optimizer.param_groups[0]['lr'] = 1.5e-3
        optimizer.param_groups[0]['weight_decay'] = 1e-5
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if tf_scheduler != None:
        tf_scheduler    = checkpoint['tf_scheduler']

    epoch   = checkpoint['epoch']
    metric  = torch.load(best_path)[metric]

    return [model, optimizer, scheduler, tf_scheduler, epoch, metric]
# end def

class TimeElapsed():
    def __init__(self):
        self.start  = -1
    # end def

    def time_elapsed(self):
        if self.start == -1:
            self.start = time.time()
        else:
            end = time.time() - self.start
            hrs, rem    = divmod(end, 3600)
            min, sec    = divmod(rem, 60)
            min         = min + 60*hrs
            print("Time Elapsed: {:0>2}:{:02}".format(int(min),int(sec)))
            self.start  = -1
        # end if-else
    # end def
# end class

# %%

class PositionalEncoding(torch.nn.Module):
    def __init__(self, projection_size, n=10000):
        super().__init__()
        # https://machinelearningmastery.com/a-gentle-introduction-to-positional-
        # encoding-in-transformer-models-part-1/
        # https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
        # Read the Attention Is All You Need paper to learn how to 
        # code the positional encoding
        self.projection_size = projection_size
        self.n = n
    # end def

    def forward(self, x, seq_len):

        # penc = torch.zeros(seq_len, self.projection_size)
        # # position of indices for each position in the sequence
        # position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # # scales up the position indices
        # denominator = torch.exp(torch.arange(0, self.projection_size, 2).float()) * -(math.log(self.n)/self.projection_size)
        # penc[:, 0::2] = torch.sin(position * denominator)
        # penc[:, 1::2] = torch.cos(position * denominator)
        # print(penc)
        penc = torch.zeros(seq_len, self.projection_size)
        for pos in range(seq_len):
            for i in range(int(self.projection_size//2)):
                denominator = math.pow(self.n, (2*i)/self.projection_size)
                penc[pos,2*i] = math.sin(pos/denominator)
                penc[pos, (2*i)+1] = math.cos(pos/denominator)    
        # print(f'penc size is {penc.shape}')   
        # print(f'x size is {x.shape}')   
        # print(penc)
        x += penc.unsqueeze(0).to(x.device)
        return x
    # end def
# end class

# https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py#L5
class LockedDropout(torch.nn.Module):
    def __init__(self, dropout=0.5, batch_first=True):
        super().__init__()
        self.dropout = dropout
        self.batch_first=batch_first
    # end def

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        # end if
        # T, B, C to B,T,C
        # x, x_lens = pad_packed_sequence(x, batch_first=self.batch_first)
        # m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1-self.dropout)
        # mask = Variable(m, requires_grad=False)/(1-self.dropout)
        # mask = mask.expand_as(x)
        # x = mask*x

        x = x.clone()
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        x =  x * mask
        # x = pack_padded_sequence(x, x_lens, batch_first=self.batch_first, enforce_sorted=False)
        return x
    # end def
# end class

# %%
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        # print('initializer')
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # print('atten_scores multiplication')
        # print(K.transpose(-2,-1).shape)
        # print(Q.shape)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # print('attn_scores shape')
        # print(attn_scores.shape)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        # print('batch_size, seq_lens')
        # print(batch_size, seq_length)
        out = x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        return out
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        # print('batch_size, seq_lens')
        # print(batch_size, seq_length)
        out = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return out
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        # print('multihead...')
        # print('d_model:',self.d_model)
        # print(f'Q shape: {Q.shape}, K shape {K.shape}, V shape: {V.shape}')
        # print(Q.shape) # torch.Size([128, 176, 256])
        # print(K.shape)
        # print(V.shape)
        # print(self.W_q) # Linear(in_features=128, out_features=128, bias=True)
        if Q.shape[2] != self.d_model:
            Q=Q.permute(2,1,0) # torch.Size([256, 176, 128])
            K=K.permute(2,1,0)
            V=V.permute(2,1,0)
            # print(f'if permuted Q shape: {Q.shape}, K shape {K.shape}, V shape: {V.shape}')
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        # print(f'weights Q shape: {Q.shape}, K shape {K.shape}, V shape: {V.shape}')

        # print('what is going on?')
        # print(Q.shape)
        # print(self.W_q(Q))
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # print('Q,K, and V shapes')
        # print(Q.shape, K.shape, V.shape)
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        # print(f'multihead output: {output.shape}')

        return output

# %%
# x_sample    = torch.rand(128, 176, 256)
# projection_size = 128
# num_heads = 8
# mhead = MultiHeadAttention(projection_size, num_heads)

# %%
# mhead.forward(x_sample,x_sample,x_sample)

# %%
## Transformer Encoder ##

class TransformerEncoder(torch.nn.Module):
    def __init__(self, projection_size, num_heads=1, dropout= 0.0):
        super().__init__()
        # projection_size = 128
        # Compute multihead attention. You are free to use the version provided by pytorch
        self.attention  = MultiHeadAttention(d_model=projection_size,
                                             num_heads=num_heads)

        self.bn1        = torch.nn.BatchNorm1d(projection_size)

        self.bn2        = torch.nn.BatchNorm1d(projection_size)
        self.dropout    = torch.nn.Dropout(dropout)

        # Feed forward neural network
        # OH: not necessary to be so deep
        self.MLP        = torch.nn.Sequential(
            torch.nn.Linear(projection_size, 2*projection_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*projection_size, projection_size),

        )

    def forward(self, x):
        # compute the key, query and value

        # print(x.shape)
        # torch.Size([128, 176, 256])
        # [batch_size, seq_length, feature_size] 

        # compute the output of the attention module
        # print('1')
        # print(f'x shape before attention in encoder: {x.shape}')
        out1 = self.attention(x, x, x)
        # print(f'shape after attention in encoder: {out1.shape}')

        # print('2')
        # print('passing attention')
        out1 = self.dropout(out1)
        # print(f'shape after dropout in encoder: {out1.shape}')
        # out1 = out1.permute(2,1,0)
        # print(f'shape after permute in encoder: {out1.shape}')
        
        out1 += x
        # print(f'shape after residual in encoder: {out1.shape}')

        # print(out1.shape)
        out1 = out1.permute(0,2,1)
        # print(f'shape after permute in encoder: {out1.shape}')
        # print(out1.shape)
        out1 = self.bn1(out1)
        # print('after bn..')
        # print(out1.shape)
        out1 = out1.permute(0,2,1)
        # print('after permute')
        # print(out1.shape)

        # Apply the output of the feed forward network
        out2    = self.MLP(out1)
        # print('after MLP')
        # Apply a residual connection between the input and output of the  FFN
        out2 = self.dropout(out2)
        # print('out2 shape')
        # print(out2.shape)
        # print('out1 shpape')
        # print(out1.shape)
        out2    += out1

        out2 = out2.permute(0,2,1)
        # print('after permute ..')
        # print(out2.shape)
        out2 = self.bn2(out2)
        # print(out2.shape)
        out2 = out2.permute(0,2,1)
        # print(out2.shape)
        

        return out2
    # end def forward
# end TransformerEncoder

# model   = TransformerEncoder(
#     projection_size  = 128,
#     num_heads=4
# ).to(DEVICE)

# model   = TransformerEncoder(
#     projection_size  = 128,
#     num_heads=4
# )
# # projection size must be divisible by num_heads!


# print(model)

# %%
# x_sample    = torch.rand(128, 176, 256)
# # torchsummaryX.summary(model, x_sample.to(DEVICE))
# # torchsummaryX.summary(model, x_sample)
# model.forward(x_sample)
# del x_sample


# %%

class PLockedDropout(torch.nn.Module):
    def __init__(self, dropout=0.5, batch_first=True):
        super().__init__()
        self.dropout = dropout
        self.batch_first=batch_first
    # end def

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        # end if
        # T, B, C to B,T,C
        x, x_lens = pad_packed_sequence(x, batch_first=self.batch_first)
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1-self.dropout)
        mask = Variable(m, requires_grad=False)/(1-self.dropout)
        mask = mask.expand_as(x)
        x = mask*x
        x = pack_padded_sequence(x, x_lens, batch_first=self.batch_first, enforce_sorted=False)
        return x
    # end def
# end class

class pBLSTM(torch.nn.Module):
    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    def __init__(self, input_size, hidden_size, dropout=0.0, num_layers=3, bidirectional=True, batch_first=True, lockdropout=0.2):
        super(pBLSTM, self).__init__()
        # Initialize a single layer bidirectional LSTM with the given input_size and hidden_size
        self.batch_first=batch_first
        input_size = input_size *2
        self.blstm = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   dropout=dropout,
                                   batch_first=batch_first)
        self.dropout = PLockedDropout(lockdropout)

    def forward(self, x_packed): # x_packed is a PackedSequence
        # Pad Packed Sequence
        if isinstance(x_packed, PackedSequence):
            x, x_lens = pad_packed_sequence(x_packed, batch_first=self.batch_first)
        else:
            x = x_packed
            x_lens = None
        # end if

        # Call self.trunc_reshape() which downsamples the time steps of x and increases the feature 
        # dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        x, x_lens = self.trunc_reshape(x, x_lens)
        # Pack Padded Sequence. What output(s) would you get?
        if x_lens is not None:
            x = pack_padded_sequence(x, x_lens.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        # end if

        # Pass the sequence through bLSTM
        output, hidden = self.blstm(x)
        output = self.dropout(output)

        # What do you return?

        return output, hidden

    def trunc_reshape(self, x, x_lens):
        # If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        # import pdb 
        # pdb.set_trace()
        batch_size, length, dim = x.shape # dim is features
        if length%2 != 0:
            # if odd, trim to even length so we can downsample
            x = x[:,:-1,:]
            x_lens -= 1
            length -= 1
        # end if

        # Reshape x. When reshaping x, you have to reduce number of timesteps by 
        # a downsampling factor while increasing number of features by the same factor
        # x = x.contiguous().view(batch_size, length//2, dim*2)
        x = torch.reshape(x, shape=(batch_size, length//2, 2*dim))

        # Reduce lengths by the same downsampling factor
        x_lens = torch.clamp(x_lens, max=length//2, out=None)
        return x, x_lens
    # # end def
# end pLSTM class

class TransformerListener(torch.nn.Module):

    def __init__(self,
                 input_size,
                 base_lstm_layers        = 3,
                 listener_hidden_size    = 256,
                 n_heads                 = 4,
                 tf_blocks               = 2,
                 dropout                 = 0.3):
        super().__init__()

        self.embedding      = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_size,
                            out_channels=listener_hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm1d(listener_hidden_size),
            torch.nn.GELU(),
            torch.nn.Conv1d(in_channels=listener_hidden_size,
                            out_channels=listener_hidden_size*2,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm1d(listener_hidden_size*2),
            torch.nn.GELU(),
            torch.nn.Conv1d(in_channels=listener_hidden_size*2,
                            out_channels=listener_hidden_size*2,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm1d(listener_hidden_size*2),
            torch.nn.GELU(),
        )

        # create an lstm layer
        # OH: try reducing to two
        pblstm          = []
        for i in range(3):
            if i == 0:
                pb_input = listener_hidden_size*2
                pb_hidden = listener_hidden_size//2
            else:
                pb_input  = listener_hidden_size
                pb_hidden = listener_hidden_size//2
            # end if-else

            pblstm.append(pBLSTM(input_size=pb_input,
                                      hidden_size=pb_hidden,
                                      dropout=dropout,
                                      lockdropout=dropout,
                                      batch_first=True,
                                      num_layers=base_lstm_layers,
                                      bidirectional=True))
    
        self.pblstm = torch.nn.ModuleList(pblstm)

        # HACK: reduce to 1 base_lstm and increase base_lstm layers, downsample in embedding
        # HACK: also try pblstm


        self.dropout = LockedDropout(dropout)

        # compute the postion encoding
        self.positional_encoding    = PositionalEncoding(projection_size=listener_hidden_size)
        self.permute = PermuteBlock()
 
        # create a sequence of transformer blocks
        self.transformer_encoder    = torch.nn.Sequential(
            *[TransformerEncoder(listener_hidden_size, n_heads) for _ in range(tf_blocks)]
        )
        # for i in range(tf_blocks):
            

    def forward(self, x, x_len):
        # print(f'x shape: {x.shape}')
        x                  = self.permute(x)
        # print(f'after first permute out shape: {output.shape}')
        x                  = self.embedding(x)
        x                  = self.permute(x)
        # print(f'after embedding: {x.shape}')


        # pack the inputs before passing them to the LSTm
        output                = pack_padded_sequence(x, x_len,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        # Pass the packed sequence through the lstm
        for pblstm_layer in self.pblstm:
            output, _ = pblstm_layer(output)
            # unpacked_output, output_lengths = pad_packed_sequence(output, batch_first=True)
            # print(f'Shape after pBLSTM layer: {unpacked_output.shape}')
        # end for

        # Unpack the output of the lstm
        output, output_lengths  = pad_packed_sequence(output,
                                                      batch_first=True)

        # calculate the position encoding
        output  = self.positional_encoding(output, output.size(1))
        # Pass the output of the positional encoding through the transformer encoder
        # print(f'in listener after pos_enc output shape: {output.shape}')
        # output = output.permute(2,1,0)
        # print(f'in listener after pos_enc output shape Aft permute: {output.shape}')
        output  = self.transformer_encoder(output)
        # print(f'post TransList output shape: {output.shape}')
        
        
        # print(f'after transformer encoder {output.shape}')

        return output, output_lengths
    # end def
# end TransformerListener

# %%
# Assuming PositionalEncoding, PermuteBlock, and TransformerEncoder are defined

# # Parameters for TransformerListener
# input_size = 128
# batch_size = 96
# sequence_length = 176
# listener_hidden_size = 256

# # Create an instance of TransformerListener
# transformer_listener = TransformerListener(input_size=input_size,
#                                            listener_hidden_size=listener_hidden_size)
# print(transformer_listener)

# # Prepare test input
# x_samplel = torch.randn(batch_size, sequence_length, input_size)
# # print(x_samplel.shape)

# # Prepare sequence lengths (for simplicity, here all sequences are of full length)
# x_lenl = torch.full((batch_size,), sequence_length, dtype=torch.int64)

# # Pass the input through the model
# output, output_lengths = transformer_listener(x_samplel, x_lenl)

# Print output to verify
# print(output)
# print(output_lengths)


# %%
class Attention(torch.nn.Module):
  def __init__(self, listener_hidden_size, speller_hidden_size, projection_size):
    super().__init__()
    # initialize the linear layers
    self.VW = torch.nn.Linear(listener_hidden_size, projection_size)
    self.KW = torch.nn.Linear(listener_hidden_size, projection_size)
    self.QW = torch.nn.Linear(speller_hidden_size, projection_size)
    self.softmax = torch.nn.Softmax(dim=1)

  def set_key_value(self, encoder_outputs):
    # compute the key and value from the 
    # encoder_outputs using the respective 
    # linear transformations
    # (batch_size, timesteps, projection_size)
    self.key = self.KW(encoder_outputs)
    self.value = self.VW(encoder_outputs)
    return self.key, self.value

  def compute_context(self, decoder_context):
    # create query from the decoder context
    # (batch_size, projection_size)
    query = self.QW(decoder_context)
    query_len = query.shape[1]

    # compute the raw weights (batch_size, timesteps)
    # using batch matrix multiplication
    # unsqueeze adds an extra dimension: (batch_size, projection_size, 1)
    # squeeze removes single dimension: (batch_size, timestems)
    raw_weights = torch.bmm(self.key, torch.unsqueeze(query,2)).squeeze(2)

    # apply softmax to get attention weights
    attention_weights = self.softmax(raw_weights/np.sqrt(query_len))

    # calculate attention context weights by combining wieghts and values
    # (batch_size, projection_size)
    attention_context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)
    return attention_context, attention_weights
  # end def compute_context
# end Attention

# %%
class Speller(torch.nn.Module):

  # Refer to your HW4P1 implementation for help with setting up the 
  # language model.
  # The only thing you need to implement on top of your HW4P1 model is 
  # the attention module and teacher forcing.


  def __init__(self, vocab_size, embedding_size, hidden_size, speller_size, attender:Attention, dropout=0.3, max_timesteps=500):
    super(). __init__()

    self.attend = attender # Attention object in speller
    self.max_timesteps = max_timesteps
    self.hidden_size = hidden_size

    self.embedding =  torch.nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_TOKEN)

    # Create a sequence of LSTM Cells
    self.lstm_cells =  torch.nn.Sequential(
        torch.nn.LSTMCell(embedding_size + hidden_size, speller_size),
        torch.nn.LSTMCell(speller_size, speller_size),
        torch.nn.LSTMCell(speller_size, speller_size)
    )

    # For CDN (Feel free to change)
    # Linear module to convert outputs to correct hidden size (Optional: TO make dimensions match)
    self.output_to_char = torch.nn.Linear(speller_size+hidden_size, embedding_size)
    # Check which activation is suggested
    self.activation = torch.nn.Tanh()

    # self.dropout = torch.nn.Dropout(dropout)
    self.dropout = LockedDropout(dropout)
    self.ldropout = torch.nn.Dropout(dropout)
    # Linear layer to convert hidden space back to logits for token classification
    self.linear1   = torch.nn.Linear(embedding_size, embedding_size)
    self.char_prob = torch.nn.Linear(embedding_size, vocab_size)
    # Weight tying (From embedding layer)
    self.char_prob.weight = self.embedding.weight
    self.apply(self._initialize_weights)

  def _initialize_weights(self, module):
    if isinstance(module, nn.LSTMCell):
        torch.nn.init.uniform_(module.weight_ih, -0.1, 0.1)
        torch.nn.init.uniform_(module.weight_hh, -0.1, 0.1)
    # end if
  # end def

  def lstm_step(self, lstm_input, hidden_states):

    for i in range(len(self.lstm_cells)):
        # Feed the input through each LSTM Cell
        hidden_states[i] = self.lstm_cells[i](lstm_input, hidden_states[i])
        lstm_input = hidden_states[i][0]
    return lstm_input, hidden_states # What information does forward() need?
  # end def

  def CDN(self, input):
    # Make the CDN here, you can add the output-to-char
    # output = self.dropout(input)
    output = self.output_to_char(input)
    output = self.activation(output)
    output = self.ldropout(output)

    # add a middle layer
    output = self.linear1(output)
    output = self.activation(output)
    output = self.ldropout(output)

    # output = self.dropout(output)
    # output = self.output_to_char(output)
    # output = self.activation(output)
    prob = self.char_prob(output)
    return prob
  # end def
    

  def forward (self, x, y=None, teacher_forcing_ratio=1):
    batch_size = x.shape[0]
    # initial context tensor for time t = 0
    attn_context = torch.zeros(batch_size, self.hidden_size).to(DEVICE) 
    # Set it to SOS for time t = 0
    output_symbol = torch.full((batch_size,), SOS_TOKEN).to(DEVICE)
    raw_outputs = []
    attention_plot = []

    if y is None:
      timesteps = self.max_timesteps
      teacher_forcing_ratio = 0 #Why does it become zero?
    else:
      # How many timesteps are we predicting for?
      timesteps = y.shape[1]
    # end if-else

    # Initialize your hidden_states list here similar to HW4P1
    hidden_states_list = [None] * len(self.lstm_cells)

    for t in range(timesteps):
      # generate a probability p between 0 and 1
      p =random.uniform(0,1)
      if p < teacher_forcing_ratio and t > 0: # Why do we consider cases only when t > 0? What is considered when t == 0? Think.
        # Take from y, else draw from probability distribution
        output_symbol = y[:,t-1]
    #   else:
    #     output_symbol = output_symbol.argmax(dim=-1)
      # Embed the character symbol

      char_embed = self.embedding(output_symbol)

      # Concatenate the character embedding and context from attention, as shown in the diagram
    #   if y is None:
    #     print(f'shape of char_embed {char_embed.shape}, shape of attn_context {attn_context.shape}')
      lstm_input = torch.cat([char_embed, attn_context], dim=1)
      
      # Feed the input through LSTM Cells and attention.
      lstm_out, hidden_states_list = self.lstm_step(lstm_input, hidden_states_list) 
      # What should we retrieve from forward_step to prepare for the next timestep?
      
      # Feed the resulting hidden state into attention
      attn_context, attn_weights = self.attend.compute_context(lstm_out) 
      
      # You need to concatenate the context from the attention module with the LSTM output hidden state, as shown in the diagram
      cdn_input = torch.cat((lstm_out, attn_context), dim=1)
      
      # call CDN with cdn_input
      raw_pred = self.CDN(cdn_input)

      # Generate a prediction for this timestep and collect it in output_symbols
      # Draw correctly from raw_pred
      output_symbol = torch.argmax(raw_pred, dim=1)

      raw_outputs.append(raw_pred) # for loss calculation
      attention_plot.append(attn_weights) # for plotting attention plot


    attention_plot = torch.stack(attention_plot, dim=1)
    raw_outputs = torch.stack(raw_outputs, dim=1)

    return raw_outputs, attention_plot


# %%

class ASRModel(torch.nn.Module):
  def __init__(self, vocab_size, embedding_size, input_size, encoder_hidden_size, listener_size, speller_size, projection_size, num_heads, 
               encoder_blocks, dropout): # add parameters
    super().__init__()
    
    # Pass the right parameters here
    self.listener = TransformerListener(input_size=input_size,
                                        listener_hidden_size=encoder_hidden_size,
                                        n_heads=num_heads,
                                        tf_blocks=encoder_blocks,
                                        dropout=dropout)
    self.attend = Attention(listener_size, speller_size, projection_size)
    self.speller = Speller(vocab_size,
                           embedding_size, 
                           projection_size, 
                           speller_size, 
                           self.attend,
                           dropout)

    # you can add augmentation here
    self.aug = torch.nn.Sequential(
        PermuteBlock(),
        torchaudio.transforms.TimeMasking(dropout*100),
        torchaudio.transforms.FrequencyMasking(dropout*100),
        PermuteBlock()
    )


  def forward(self, x,lx,y=None,teacher_forcing_ratio=1):
    # Encode speech features
    # print('begin ASR model')
    # print(f'x shape: {x.shape}\n')
    if self.training:
        x = self.aug(x)

    encoder_outputs, encoder_lens = self.listener(x,lx)

    # We want to compute keys and values ahead of the decoding step, as they are constant for all timesteps
    # Set keys and values using the encoder outputs
    # print(f'encoder_outputs shape: {encoder_outputs.shape}')
    key, value = self.attend.set_key_value(encoder_outputs)

    # Decode text with the speller using context from the attention
    # print(f'x shape: {x.shape}, y shape: {y.shape}')
    raw_outputs, attention_plots = self.speller(x, y,teacher_forcing_ratio=teacher_forcing_ratio)
    # print(f'raw_output shape: {raw_outputs.shape}, attention_plots shape: {attention_plots.shape}')

    return raw_outputs, attention_plots

# %%
torch.cuda.empty_cache()
gc.collect()

# model = ASRModel(

#     vocab_size=len(VOCAB),
#     embedding_size=350,
#     input_size=28,
#     encoder_hidden_size=240,
#     listener_size=4*240,
#     speller_size=560,
#     projection_size=300
# )

# model = ASRModel(

#     vocab_size=len(VOCAB),
#     embedding_size=256,
#     input_size=28,
#     encoder_hidden_size=256,
#     listener_size=256,
#     speller_size=512,
#     projection_size=128,
#     num_heads=4,
#     encoder_blocks=2
# )

# model = ASRModel(

#     vocab_size=len(VOCAB),
#     embedding_size=128,
#     input_size=28,
#     encoder_hidden_size=128,
#     listener_size=128,
#     speller_size=256,
#     projection_size=64,
#     num_heads=4,
#     encoder_blocks=2
# )
# OH: try training with 460 or 360?
model = ASRModel(

    vocab_size=len(VOCAB),
    embedding_size=256,
    input_size=28,
    encoder_hidden_size=256,
    listener_size=256,
    speller_size=512,
    projection_size=256,
    num_heads=2,
    encoder_blocks=2,
    dropout=0.3
)

# %%
model = model.to(DEVICE)
print(model)

# Counting the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Listener parameters
listener_params = sum(p.numel() for p in model.listener.parameters())
print(f"Number of parameters in listener: {listener_params}")

# Attender parameters
attender_params = sum(p.numel() for p in model.attend.parameters())
print(f"Number of parameters in attender: {attender_params}")

# Speller parameters
speller_params = sum(p.numel() for p in model.speller.parameters())
print(f"Number of parameters in speller: {speller_params}")


# %%
"""# Loss Function, Optimizers, Scheduler"""

optimizer   = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

# HACK: using mean
criterion   = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_TOKEN).to(DEVICE)

# scaler      = torch.cuda.amp.GradScaler(enabled=False)

scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.5, 
                                                         patience=5, 
                                                         verbose=True)


# %%
"""# Levenshtein Distance"""

# We have given you this utility function which takes a sequence of indices and converts them to a list of characters
def indices_to_chars(indices, vocab):
    tokens = []
    for i in indices: # This loops through all the indices
        if int(i) == SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[int(i)])
    return tokens

# To make your life more easier, we have given the Levenshtein distantce / Edit distance calculation code
def calc_edit_distance(predictions, y, y_len, vocab= VOCAB, print_example= False):

    dist                = 0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size):

        y_sliced    = indices_to_chars(y[batch_idx,0:y_len[batch_idx]], vocab)
        pred_sliced = indices_to_chars(predictions[batch_idx], vocab)

        # Strings - When you are using characters from the AudioDataset
        y_string    = ''.join(y_sliced)
        pred_string = ''.join(pred_sliced)

        dist        += Levenshtein.distance(pred_string, y_string)
        # Comment the above abd uncomment below for toy dataset
        # dist      += Levenshtein.distance(y_sliced, pred_sliced)

    if print_example:
        # Print y_sliced and pred_sliced if you are using the toy dataset
        print("\nGround Truth : ", y_string)
        print("Prediction   : ", pred_string)

    dist    /= batch_size
    return dist

# %%
gc.collect()
torch.cuda.empty_cache()

# %%
"""# Train and Validation functions

"""
gc.collect()
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
def train(model, dataloader, criterion, optimizer, teacher_forcing_rate, clip_value = 1.0):

    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    running_loss        = 0.0
    running_perplexity  = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        optimizer.zero_grad()

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        # with torch.cuda.amp.autocast():

        raw_predictions, attention_plot = model(x, lx, y, teacher_forcing_rate)

        # Predictions are of Shape (batch_size, timesteps, vocab_size).
        # Transcripts are of shape (batch_size, timesteps) Which means that you have batch_size amount of batches with timestep number of tokens.
        # So in total, you have batch_size*timesteps amount of characters.
        # Similarly, in predictions, you have batch_size*timesteps amount of probability distributions.
        # How do you need to modify transcipts and predictions so that you can calculate the CrossEntropyLoss? Hint: Use Reshape/View and read the docs
        # Also we recommend you plot the attention weights, you should get convergence in around 10 epochs, if not, there could be something wrong with
        # your implementation
        # print(f'raw predictions shape before permute: {raw_predictions.shape}')
        raw_predictions = torch.permute(raw_predictions, (0,2,1))
        # print(f'raw predictions shape after permute: {raw_predictions.shape}')
        # print(f'y shape ]: {y.shape}')
        loss        =  criterion(raw_predictions, y)
        
        # batch_size, timesteps, vocab_size = raw_predictions.shape
        # loss        =  criterion(raw_predictions.view(batch_size * timesteps, -1), y.view(batch_size * timesteps)) # TODO: Cross Entropy Loss

        # loss        = criterion(raw_predictions.view(-1, raw_predictions.shape[2]), y.view(-1))

        perplexity  = torch.exp(loss) # Perplexity is defined the exponential of the loss
        # print(f'loss shape, {loss.shape} supposed to be Scalar')
        # print(f'loss shape, {loss} supposed to be Scalar')
        # print(f'perplexity shape, {perplexity.shape} supposed to be Scalar')
        running_loss        += loss.item()
        running_perplexity  += perplexity.item()

        # Backward on the masked loss
        loss.backward()
        # scaler.scale(loss).backward() # for mixed precision

        # Optional: Use torch.nn.utils.clip_grad_norm to clip gradients to prevent them from exploding, if necessary
        # If using with mixed precision, unscale the Optimizer First before doing gradient clipping
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()


        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            perplexity="{:.04f}".format(running_perplexity/(i+1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            tf_rate='{:.02f}'.format(teacher_forcing_rate))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    batch_bar.close()

    return running_loss, running_perplexity, attention_plot


# %%
def validate(model, dataloader):

    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    running_lev_dist = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        with torch.inference_mode():
            raw_predictions, attentions = model(x, lx, y = None)

        # Greedy Decoding
        # How do you get the most likely character from each distribution in the batch?
        greedy_predictions   =  torch.argmax(raw_predictions, dim=2)

        # Calculate Levenshtein Distance
        running_lev_dist    += calc_edit_distance(greedy_predictions, y, ly, VOCAB, print_example = False) # You can use print_example = True for one specific index i in your batches if you want
  
        batch_bar.set_postfix(
            dist="{:.04f}".format(running_lev_dist/(i+1)))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    batch_bar.close()
    running_lev_dist /= len(dataloader)

    return running_lev_dist


# %%
# Login to Wandb
# Initialize your Wandb Run Here
# Save your model architecture in a txt file, and save the file to Wandb
run = wandb.init(
    name = "early-submission", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw4p2-ablations", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)

# %%
def plot_attention(attention, epoch_num):
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    seaborn.heatmap(attention, cmap='GnBu')
    plt.savefig('attention-'+str(epoch_num)+'.svg')
    plt.show()

# %%
def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1], 
         'epoch'                    : epoch}, 
         path
    )

# %%
gc.collect()
torch.cuda.empty_cache()

"""# Experiment"""

best_lev_dist = float("inf")
tf_rate = 1.0
patience = 15  # Number of epochs to wait for improvement before stopping
counter = 0   # Tracks how many epochs have passed without improvement
delta = 0.001  # Minimum change to qualify as an improvement
best_model_path = "Run3_model_0.0001_32_150.pth"

checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

# %% [markdown]
# 

# %%
torch.cuda.empty_cache()
gc.collect()

# %%
def testing(model, dataloader):

    results = []
    model.eval()

    # batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Testing")

    running_lev_dist = 0.0

    for i, (x,lx) in enumerate(dataloader):

        x, lx = x.to(DEVICE), lx

        with torch.inference_mode():
            raw_predictions, attentions = model(x, lx, y = None)

        greedy_predictions   = torch.argmax(raw_predictions, dim=2) # TODO: How do you get the most likely character from each distribution in the batch?

        del x, lx
        results.extend(greedy_predictions)
    return results
# end def

results = testing(model, test_loader)
valid_dist = validate(model, valid_loader)

with open('submissions.csv', 'w') as file:
    file.write("index,label\n")
    for i,pred in enumerate(results):
        pred_sliced = indices_to_chars(results[i], VOCAB)
        pred_string = ''.join(pred_sliced)
        file.write(f"{i},{pred_string}\n")


command = "kaggle competitions submit -c idl-hw4p2-slack -f submissions.csv -m 'I made it!'"
subprocess.run(command, shell=True)


