from opt import get_opts
import torch
from mmcv import Config
import numpy as np
import random, os
import warnings ; warnings.filterwarnings('ignore')
from lib.load_data import load_data

def seed_everything(seed):
    '''
    setting seed for all module
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def load_everything(args, cfg):
    '''
    load images / poses / cameras / data
    '''
    data_dict=load_data(cfg.data)
    
    # construct tensor
    data_dict['images']=torch.FloatTensor(data_dict['images'],device='cpu')
    data_dict['poses']=torch.FloatTensor(data_dict['poses'],device='cpu')
    return data_dict

def train(args):
    pass

if __name__ == "__main__":
    # set up arg and config
    opts=get_opts()
    cfg=Config.fromfile(opts.config)
    
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')    
        
    seed_everything(opts.seed)
    
    # set up dataset
    data_dict=load_everything(opts, cfg)
    
    # train the model
    
    # rendering testing set
    
    # render video
    
    