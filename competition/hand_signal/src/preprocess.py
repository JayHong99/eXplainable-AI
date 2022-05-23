import numpy as np
from pathlib import Path
import torch

from torch.utils.data import DataLoader, random_split, Dataset
from .dataset import custom_dataset

class Preprocess : 
    def __init__(self, root_path : Path, transforms, train_ratio : float, batch_size : int, random_seed : int) : 
        self.root_path = root_path
        self.transforms = transforms
        self.train_ratio = train_ratio
        self.batch_size = batch_size

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def load_dset(self, data_path : Path, mode : str
        ) -> Dataset : 
        return custom_dataset(data_path = data_path, transforms= self.transforms, mode = mode)
    
    def calculate_split_ratio(self, dset) :
        total_len = len(dset)
        train_size = int(total_len * self.train_ratio)
        valid_size = total_len - train_size 
        return train_size, valid_size

    def split_dset(self, dset
        ) -> Dataset : 
        train_size, valid_size = self.calculate_split_ratio(dset)
        return random_split(
                            dataset = dset, 
                            lengths = [train_size, valid_size]
                            )
    
    def load_dataloader(self, dset, shuffle
        ) -> DataLoader : 
        loader = DataLoader(dataset = dset, 
                            batch_size = self.batch_size, 
                            shuffle=shuffle,
                            num_workers=4,
                            pin_memory=True)
        return loader

    def process_by_mode(self, mode : str
        ) -> DataLoader : 
        path = self.root_path.joinpath(mode)
        dset = self.load_dset(path, mode)
        if mode == 'train' : 
            train_dset, valid_dset = self.split_dset(dset)
            return self.load_dataloader(train_dset, True), self.load_dataloader(valid_dset, False), dset.label_dict
        else : 
            return self.load_dataloader(dset, False)
    
    def process(self) : 
        train_loader, valid_loader, label_dict = self.process_by_mode('train')
        test_loader = self.process_by_mode('test')
        
        return train_loader, valid_loader, test_loader, label_dict