import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Embedding_Layer(nn.Module):
    def __init__(self, pre_embed, embedding_dim):
        super(Embedding_Layer, self).__init__()
        self.embedding = nn.Embedding(pre_embed.shape[0], embedding_dim)
        self.embedding.weight.data.copy_(torch.Tensor(pre_embed))
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        x = x.to(torch.int64)
        embedsm = self.embedding(x)
        return embedsm


class ADLSTM_Gender__Model(nn.Module):
    def __init__(self, class_num, pre_embed, embedding_dim, hidden_dim, bidirectional=True):
        super(ADLSTM_Gender__Model, self).__init__()
        self.hidden_dim = hidden_dim

        for index, (pre_e, e_size) in enumerate(zip(pre_embed, embedding_dim)):
            setattr(self, 'embedding_layer_{}'.format(index), Embedding_Layer(pre_e, e_size))

        self.lstm = nn.LSTM(sum(embedding_dim), hidden_dim, num_layers=1, bidirectional=bidirectional)
        if bidirectional == True:
            fc_hd = hidden_dim * 2
        else:
            fc_hd = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fc_hd, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, class_num)
        )

    def forward(self, x):
        for i in range(len(x)):
            x[i] = x[i].to(torch.int64)
        embedding_buffer = [getattr(self, 'embedding_layer_{}'.format(index))(inp_embed) for index, inp_embed in
                            enumerate(x)]
        embedsm = torch.cat(embedding_buffer, dim=2)
        output, hidden = self.lstm(embedsm, None)
        output = output.mean(axis=1)
        output = self.classifier(output)
        output = F.log_softmax(output, dim=1)

        return output

class LSTM_Extraction_Layer(nn.Module):
    def __init__(self, pre_embed, embedding_dim, hidden_dim,bidirectional=True):
        super(LSTM_Extraction_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(pre_embed.shape[0], embedding_dim)
        self.embedding.weight.data.copy_(torch.Tensor(pre_embed))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=bidirectional)

    def forward(self, x):
        x = x.to(torch.int64)
        embedsm = self.embedding(x)
        output, hidden = self.lstm(embedsm, None)
        output = output.mean(axis=1)
        return output


class ADLSTM_Age_Model(nn.Module):
    def __init__(self, class_num, pre_embed, embedding_dim, hidden_dim, bidirectional=True):
        super(ADLSTM_Age_Model, self).__init__()
        self.hidden_dim = hidden_dim

        self.n_extraction = len(embedding_dim)
        for index, (pre_e, e_size, h_size) in enumerate(zip(pre_embed, embedding_dim, hidden_dim)):
            setattr(self, 'extraction_layer_{}'.format(index),
                    LSTM_Extraction_Layer(pre_e, e_size, h_size, bidirectional=bidirectional))

        if bidirectional == True:
            fc_hd = sum(hidden_dim) * 2
        else:
            fc_hd = sum(hidden_dim)

        #         self.dnn_dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(fc_hd, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, class_num)
        )

    def forward(self, x):
        for i in range(len(x)):
            x[i] = x[i].to(torch.int64)
        extract_buffer = [getattr(self, 'extraction_layer_{}'.format(index))(inp_embed) for index, inp_embed in
                          enumerate(x)]
        output = torch.cat(extract_buffer, dim=1)
        #         output = self.dnn_dropout(output)
        output = self.classifier(output)
        output = F.log_softmax(output, dim=1)

        return output

class ADEM_Age_Model(nn.Module):
    def __init__(self, class_num, pre_embed, embedding_dim, hidden_dim,bidirectional=True):
        super(ADEM_Age_Model, self).__init__()
        self.hidden_dim = hidden_dim
        
        for index,(pre_e,e_size) in enumerate(zip(pre_embed,embedding_dim)):
            setattr(self, 'embedding_layer_{}'.format(index), Embedding_Layer(pre_e,e_size))
        
        self.lstm = nn.LSTM(sum(embedding_dim),hidden_dim, num_layers=1,bidirectional=bidirectional)
        if bidirectional==True:
            fc_hd = hidden_dim*2
        else:
            fc_hd = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fc_hd, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, class_num)
        )

    def forward(self, x):
        for i in range(len(x)):
            x[i] = x[i].to(torch.int64)
        embedding_buffer = [getattr(self, 'embedding_layer_{}'.format(index))(inp_embed) for index, inp_embed in enumerate(x)]
        embedsm = torch.cat(embedding_buffer,dim=2)
        output, hidden = self.lstm(embedsm,None)
        output = output.mean(axis=1)
        output = self.classifier(output)
        output = F.log_softmax(output, dim=1)
        
        return output

