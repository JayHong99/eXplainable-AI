from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from .dataset import audio_dataset

class preprocess : 
    def __init__(self, data_path : Path,
                        valid_ratio : float,
                        random_seed,
                        batch_size : int,
                        log_func
                ) -> DataLoader : 

        self.data_path = data_path
        self.valid_ratio = valid_ratio
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.log_func = log_func
        self.log_func('Setting Data')
    
    def set_dataset(self) : 
        self.log_func('\tLoading Dataset')
        total_dset = audio_dataset(self.data_path, 'train')
        meta_info = total_dset.info_df
        train_idx, valid_idx = train_test_split(
                                                range(total_dset.__len__()), 
                                                test_size = self.valid_ratio,
                                                random_state = self.random_seed,
                                                shuffle = True,
                                                stratify = meta_info['covid19'].values
                                                )
        self.train_dset = Subset(total_dset, train_idx)
        self.valid_dset = Subset(total_dset, valid_idx)
        self.test_dset = audio_dataset(self.data_path, 'test')

    def set_dataloader(self) : 
        self.log_func('\tLoading Dataloader')

        train_dl = DataLoader(self.train_dset, batch_size = self.batch_size, shuffle=True)
        valid_dl = DataLoader(self.valid_dset, batch_size = self.batch_size, shuffle=False)
        test_dl = DataLoader(self.test_dset, batch_size = self.batch_size, shuffle=False)

        return train_dl, valid_dl, test_dl