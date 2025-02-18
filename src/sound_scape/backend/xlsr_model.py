import torch, os
import numpy as np
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
import librosa
import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import fairseq

import numba
import logging

from Base_Path import get_path_relative_base

xlsr_path = get_path_relative_base("pretrained_models/XLS-R/xlsr2_300m.pt")
model_path = get_path_relative_base("pretrained_models/XLS-R/MMpaper_model.pth")


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"



def set_random_seed(random_seed, args=None):
    """ set_random_seed(random_seed, args=None)
    
    Set the random_seed for numpy, python, and cudnn
    
    input
    -----
      random_seed: integer random seed
      args: argue parser
    """
    
    # initialization                                       
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    #For torch.backends.cudnn.deterministic
    #Note: this default configuration may result in RuntimeError
    #see https://pytorch.org/docs/stable/notes/randomness.html    
    if args is None:
        cudnn_deterministic = True
        cudnn_benchmark = False
    else:
        cudnn_deterministic = args.cudnn_deterministic_toggle
        cudnn_benchmark = args.cudnn_benchmark_toggle
    
        if not cudnn_deterministic:
            print("cudnn_deterministic set to False")
        if cudnn_benchmark:
            print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return



class File_Dataset(Dataset):
	def __init__(self, paths):
            self.paths = paths
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
            return len(self.paths)


	def __getitem__(self, index):
            # X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
            X, fs = librosa.load(self.paths[index], sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,self.paths[index]



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	


def eval_dataset(dataset, model, device):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        torch.set_printoptions(threshold=10_000)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    
    return fname_list, score_list



class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        # cp_path = 'xlsr2_300m.pt'

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([xlsr_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):

        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
            layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:

        layery = layer[0].transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
        layery = layery.transpose(1, 2) # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


class Model(nn.Module):
    def __init__(self, device, args=None):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)

        return output


class xlsr_model_eval():
    def __init__(self, device):        
        logging.getLogger('numba').setLevel(logging.WARNING)

        print('setting seed...')
        #make experiment reproducible
        seed = 1234
        set_random_seed(1234)

        
        model = Model(device)

        model = nn.DataParallel(model).to(device)
        
        model.load_state_dict(torch.load(model_path,map_location=device))
        print('Model loaded : {}'.format(model_path))

        self.model = model
        self.device = device

    def eval_file(self, file):
        eval_set = File_Dataset([file])
        res = eval_dataset(eval_set, self.model, self.device)
        print(f"Results: {res[1]}")
        return res[1]

