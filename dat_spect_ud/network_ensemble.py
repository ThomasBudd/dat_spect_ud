from dat_spect_ud.network_architecture import customResNet
from dat_spect_ud.preprocessing import Preprocessing
import torch
import os
import numpy as np

class network_ensemble():
    
    def __init__(self, mode):
        
        assert mode in ['acc', 'sens', 'spec'], f"input 'mode' must be one of 'acc', 'sens', 'spec'. Got {mode}"
        
        self.preprocessing = Preprocessing()
        path_to_weights = os.path.join(os.path.realpath(__file__),
                                       'network_weights')
        weight_files = [f"network_weights_{mode}_{i}" for i in range(5)]
        
        self.networks = []
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        for weight_file in weight_files:
            # create pytorch module
            network = customResNet()
            # load weights
            network.load_state_dict(torch.load(os.path.join(path_to_weights,
                                                            weight_file)))
            # to cuda and eval
            self.networks.append(network.to(self.dev).eval())
        
        if mode == 'acc':
            self.ens_thr = 3
        elif mode == 'sens':
            self.ens_thr = 2
        elif mode == 'spec':
            self.ens_thr = 4
    
    def __call__(self, im):
        
        xb = self.preprocessing(im)
        
        votes = []
        
        for network in self.networks:
            
            out = self.network(xb)
            sm = out[0].softmax()
            votes.append(sm.argmax().item())
        
        return int(np.sum(votes) >= self.ens_thr)
