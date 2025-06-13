import torch
from torch.utils.data import Dataset

import numpy as np
from typing import Optional, Any

class GenreDataset(Dataset):
    """
    Dataset class for genre classification
    """
    def __init__(self, spectrograms:np.ndarray, labels:np.ndarray, transform:Optional[Any]=None):
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, index:int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = self.spectrograms[index]
        label = self.labels[index]
        
        spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, torch.LongTensor([label]).squeeze()