from pathlib import Path
from PIL import Image
import numpy as np
import torch

def calculate_normalize(root_path : Path
    ) -> list : 
    means = []
    stds = []
    for path in root_path.joinpath('train').glob('*.png') : 
        image = torch.Tensor(np.array(Image.open(path)).transpose(-1,0,1))
        normalize_mean = image.mean([1,2]).numpy() / 255
        normalize_std = image.mean([1,2]).numpy() / 255
        means.append(normalize_mean)
        stds.append(normalize_std)
    return list(np.mean(means, 0)), list(np.mean(stds, 0))

