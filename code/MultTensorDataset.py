import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch.nn.functional as F

class MultTensorDataset(torch.utils.data.Dataset):
    def __init__(self,feat, data, target=None):
        if target is not None:
            target = torch.tensor(target)
        self.target_tensor = target
        self.data_tensor = []
        for f in feat:
            self.data_tensor.append(
                pad_sequence([torch.from_numpy(np.array(x)) for x in data[f + '_int_seq']], batch_first=True))

    def __getitem__(self, index):
        data_tensor = []
        for i in range(len(self.data_tensor)):
            data_tensor.append(self.data_tensor[i][index])

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]

        return data_tensor, target_tensor

    def __len__(self):
        return len(self.data_tensor[0])