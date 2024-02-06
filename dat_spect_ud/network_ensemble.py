from dat_spect_ud.network_architecture import customResNet
from dat_spect_ud.preprocessing import Preprocessing
import torch
import os
import numpy as np

class network_ensemble():
    
    def __init__(self, mode):
        
        assert mode in ['acc', 'sens', 'spec', 'robust'], f"input 'mode' must be one of 'acc', 'sens', 'spec', 'robust'. Got {mode}"
        
        self.use_3d_convs = mode != 'robust'
        
        self.preprocessing = Preprocessing()
        path_to_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       'network_weights')
        weight_files = [f"network_weights_{mode}_{i}" for i in range(5)]
        
        self.networks = []
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        for weight_file in weight_files:
            # create pytorch module
            network = customResNet(self.use_3d_convs)
            # load weights
            network.load_state_dict(torch.load(os.path.join(path_to_weights,
                                                            weight_file)))
            # to cuda and eval
            self.networks.append(network.to(self.dev).eval())
        
        if mode in ['acc', 'robust']:
            self.ens_thr = 3
        elif mode == 'sens':
            self.ens_thr = 2
        elif mode == 'spec':
            self.ens_thr = 4
    
    def __call__(self, im):
        
        xb = self.preprocessing(im)
        if not self.use_3d_convs:
            xb = xb[0]
        
        votes = []
        
        for network in self.networks:
            
            out = network(xb)
            if self.use_3d_convs:
                sm = out[0].softmax(0)
                votes.append(sm.argmax().item())
            else:
                vote = out.sigmoid().item()
                votes.append(int(vote > 0.5))
        
        return int(np.sum(votes) >= self.ens_thr)
