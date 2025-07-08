import torch
from torch.utils import data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import constants as cst
import time
from torch.utils import data
from utils.utils_data import one_hot_encoding_type, tanh_encoding_type

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x, y, seq_size):
        """Initialization""" 
        self.seq_size = seq_size
        self.length = y.shape[0]
        # Always copy numpy arrays before converting to torch tensors
        if isinstance(x, np.ndarray):
            x = x.copy()
        if isinstance(y, np.ndarray):
            y = y.copy()
        self.x = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        self.y = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y
        self.data = self.x

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, i):
        input = self.x[i:i+self.seq_size, :]
        label = self.y[i]
        # Convert to torch tensor if not already
        if not torch.is_tensor(input):
            input = torch.tensor(input)
        if not torch.is_tensor(label):
            label = torch.tensor(label)
        # Ensure contiguous
        input = input.contiguous()
        label = label.contiguous()
        # Always squeeze label to scalar (0-dim)
        label = label.squeeze()
        print(f"__getitem__ idx={i}, input type={type(input)}, shape={getattr(input, 'shape', None)}, label type={type(label)}, shape={getattr(label, 'shape', None)}")
        return input, label
    

class DataModule(pl.LightningDataModule):
    def   __init__(self, train_set, val_set, batch_size, test_batch_size,  is_shuffle_train=True, test_set=None, num_workers=16):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.is_shuffle_train = is_shuffle_train
        if train_set.data.device.type != cst.DEVICE:       #this is true only when we are using a GPU but the data is still on the CPU
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=self.is_shuffle_train,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

        
    