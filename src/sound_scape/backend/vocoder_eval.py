import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import yaml
from .vocoder_model import RawNet
from torch.nn import functional as F
import librosa
import json

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

def load_sample(sample_path, max_len = 96000):
    
    y_list = []
    y, sr = librosa.load(sample_path, sr=None)
    
    if sr != 24000:
        y = librosa.resample(y, orig_sr = sr, target_sr = 24000)
        
    if(len(y) <= 96000):
        return [Tensor(pad(y, max_len))]
        
    for i in range(int(len(y)/96000)):
        if (i+1) ==  range(int(len(y)/96000)):
            y_seg = y[i*96000 : ]
        else:
            y_seg = y[i*96000 : (i+1)*96000]
        # print(len(y_seg))
        y_pad = pad(y_seg, max_len)
        y_inp = Tensor(y_pad)
        
        y_list.append(y_inp)
        
    return y_list
    
    # print(json_text)
    
    with open(output_path, 'w') as json_w:
        json.dump(json_text, json_w)

class vocoder_model:
    def __init__(self, model_path, device=None, yaml_path='model_config_RawNet.yaml'):
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.device = device
            # load model config
        with open(self.yaml_path, 'r') as f_yaml:
            yaml_parser = yaml.safe_load(f_yaml)
        
        self.model = RawNet(yaml_parser['model'], device)
        self.model =(self.model).to(device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        print('Model loaded : {}'.format(model_path))
  
        self.model.eval()
    
    def eval(self, input_path):
        out_list_multi = []
        out_list_binary = []
        for m_batch in load_sample(input_path):
            m_batch = m_batch.to(device=self.device, dtype=torch.float).unsqueeze(0)
            logits, multi_logits = self.model(m_batch)
            
            probs = F.softmax(logits, dim=-1)
            probs_multi = F.softmax(multi_logits, dim=-1)
            # print(probs)
            # out_list.append([probs[i, 1].item() for i in range(probs.size(0))][0])
            out_list_multi.append(probs_multi.tolist()[0])
            out_list_binary.append(probs.tolist()[0])

        result_multi = np.average(out_list_multi, axis=0).tolist()
        result_binary = np.average(out_list_binary, axis=0).tolist()
        return result_multi, result_binary
        # print('Multi classification result : gt:{}, wavegrad:{}, diffwave:{}, parallel wave gan:{}, wavernn:{}, wavenet:{}, melgan:{}'.format(result_multi[0], result_multi[1], result_multi[2], result_multi[3], result_multi[4], result_multi[5], result_multi[6]))
        # print('Binary classification result : fake:{}, real:{}'.format(result_binary[0], result_binary[1]))


