import os
import pickle
import numpy as np
import torch

class Preprocessing():
    
    def __init__(self):
        self.window = [0, 6.5]
        self.scaling = [0.6048597782240399, 0.6051688035793145]
        self.crop_coords = [0, 1, 9, 81, 29, 101]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __call__(self, xb):
        
        if isinstance(xb, np.ndarray):
            xb = torch.from_numpy(xb)
        
        xb = xb.to(self.device).float()
        
        assert xb.ndim in [2,3,4], f'image should have 2-4 dimensions, got {xb.ndim}'
        
        while xb.ndim < 5:
            xb = xb.unsqueeze(0)
        
        xb = xb.clip(*self.window)
        xb = (xb - self.scaling[0]) / self.scaling[1]
        
        # crop to striatum
        cc = self.crop_coords
        xb = xb[:, :, :, cc[2]:cc[3], cc[4]:cc[5]]
        
        return xb        
