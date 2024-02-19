import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import pandas as pd
import scipy.stats   as stats
import scipy.special as special
from datasets import *
import os
import ray
from ray import tune
from ray.tune.search.ax import AxSearch
from models import *
import shutil
from netCDF4 import Dataset
import h5py
import sys


def get_size(obj, seen=None):
    '''Recursively finds size of objects'''
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    # Important to mark as seen *before* entering recursion to gracefully handle self-deferential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

run_once = True
run_ax = False
SAVE_ON = True
    
data_dir = '/Users/vghodges/new_directory/brainerd/outputs/'
filename = 'all_data.csv'

qa_dict = {'Slope': [0.01, 6.0]}
th_dict = {}

#define names of label and features
label_name = ['Local']

norm_feat_name = ['MODIS', 'Elevation', 'Aspect', 'Slope']

encode_feat_name = ['Land Type #']#, 'LST CMG Day/Night']

name_dict = {'label_name': label_name, 'norm_feat_name': norm_feat_name, 'encode_feat_name': encode_feat_name}

test_split = 0.1 # fraction of data to go to testing (only occurs once ?)
vali_split = 0.2 # fraction of data to go to validation (seeing how well the training worked)
random_seed = 42
batch_size = 5000
num_workers = 4 # number of 'workers' transporting data between GPU (lots of data handling capacity) and CPU (not much data handling capacity)
max_epochs = 500 # number of training cycles ?

def evalfn(config):
    #config a dictionary of information about the model or sumthin
    learning_rate = config.pop('learning_rate', None)
    
    predict_params = {'mlp_layers': config['mlp_layers'], 'mlp_dim': config['mlp_dim'], 'dropout': config['dropout']}
    
    #data manipulation...
    normed_label, normed_feat, label_dict, feature_dict, order_dict = read_land_data(data_dir+filename, qa_dict, th_dict, name_dict)
    
    normed_data = [normed_label, normed_feat]
    train_indices, val_indices, test_indices = split_data(len(normed_label), vali_split, test_split, random_seed=random_seed)
    indices = [train_indices, val_indices, test_indices]
    data = DATA(normed_data, indices, batch_size = batch_size, num_workers = num_workers)
    
    # model setup...
    model = NN(in_channels = data.in_channels, out_channels=data.out_channels, normalization=normalization, labels = label_name, label_dict=label_dict, predict_params=predict_params, learning_rate=learning_rate)
    
    checkpoint_callback = ModelCheckpoint(monitor=model.monitor, mode=model.optmode, save_top_k=1)
    
    trainer = Trainer(devices=1, accelerator='gpu', max_epochs=max_epochs, callbacks=[checkpoint_callback], enable_progress_bar=True, num_sanity_val_steps=0)#, log_every_n_steps=5)
    trainer.fit(model, datamodule=data)
    
    val_score_dict = trainer.validate(datamodule=data, ckpt_path='best')[0]
    
    if ray.tune.is_session_enabled():
        tune.report(val_mse=val_score_dict[model.monitor])
    else:
        return model, [normed_label, normed_feat, label_dict, feature_dict, order_dict]


if __name__ == '__main__':
    
    
    print(f'=======================================================================')
    print(f'                            - Model Setup -                            ')
    print(f'=======================================================================')
    
    print(f' - label: {label_name}')
    print(f' - feature: {norm_feat_name + encode_feat_name}')
    
    
    config = {}
    
    if run_once:
        
        # This is the place where we edit the configurations for the building of our model; pass this config into the evalfn function, which uses this to create NN object
        
        config['mlp_layers'] = 7 # number of neuron layers
        config['mlp_dim'] = 9 # maximum dimension of layer (see logic in MLP class in models script
        config['dropout'] = 0.025 # dropout rate; to stabilize the model, we remove a small fraction of neurons (chosen randomly) from a layer during each training process - this reduces the dependency of the model on every single neuron, making it instead dependent on most but not all neurons. Helps in the case that a neuron's data has become corrupted or is somehow incorrect, allowing other neurons to account for the loss of that neuron. Over the course of several training steps, the hope is that the effects of removing some neurons (and subsequently increasing dependency on some others) will even out across all neurons and that the model will receive stable data from the wide body of neurons.
        config['learning_rate'] = 0.05 # initial learning rate of the model; to ensure that the model gets a fair idea of the overall body of data, we define a learning rate - for example, we do not want our learning rate to be too high for a given batch or the model may get a false impression of what the data actually looks like. Therefore, we limit the learning rate such that it takes the model a longer period of time to come to a conclusion about the overall data, thus giving a more well-rounded and accurate view of the data. In this specific case, the value we provide here is the initial learning rate. Meng explained that we often set a high learning rate at the beginning, a moderate rate during the bulk of the machine learning, and a small rate towards the end for finer adjustments. Don't want the machine to all of a sudden forget what it has learned due to a high learning rate at the end. However, in this case I will only specify the beginning learning rate and the program will optimize it throughout the course of its learning.
        
        layers = str(config['mlp_layers'])
        dim = str(config['mlp_dim'])
        dropout = str(config['dropout']).split('.')[1]
        
        if len(layers) < 2:
            layers = '0' + layers
        if len(dim) < 2:
            dim = '0' + dim
        if len(dropout) < 3:
            dropout = dropout + '0'
        
        MODEL_NAME = 'sat_to_2m_n' + layers + '_d' + dim + '_dp' + dropout
        
        
        model, info = evalfn(config)
        [normed_label, normed_feat, label_dict, feature_dict, order_dict] = info
        
        
        #save the results/model? to the specified name and location
        MODEL_PATH = './SAVE/' + MODEL_NAME + '/'
        if os.path.isdir(MODEL_PATH):
            print(f' - Remove {MODEL_PATH}')
            shutil.rmtree(MODEL_PATH, ignore_errors=True)
            print(f' - Making {MODEL_PATH}')
            os.makedirs(MODEL_PATH)
        else:
            print(f' - Making {MODEL_PATH}')
            os.makedirs(MODEL_PATH)
        
        if SAVE_ON:
            
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'label_dict': label_dict, 'feature_dict': feature_dict, 'order_dict': order_dict}, MODEL_PATH + MODEL_NAME + '.pt')
            
            print(f' - Save done!')
    
    # use this if you are really new to the model/have lots and lots of data and don't know how big it needs to be; probably won't use much
    if run_ax:
        
        param_space = {'mlp_layers': tune.randint(3, 9+1), 'mlp_dim': tune.randint(8, 12+1), 'dropout': tune.uniform(0., 0.5), 'learning_rate': tune.loguniform(1e-2, 1e-1)}
        
        ray.init(log_to_driver=False, runtime_env={'env_vars': {'PL_DISABLE_FORK': '1'}})
        
        ax_search = AxSearch(metric='val_mse', mode='min', enforce_sequential_optimization=False)
        
        tuner = tune.Tuner(tune.with_resources(evalfn, resources={'cpu': 8, 'gpu': 1}), tune_config=tune.TuneConfig(search_alg=ax_search, num_samples=80, metric='val_mse', mode='min'), param_space=param_space)
        
        results = tuner.fit()
        
        print('Best Hyperparameters found were: ', results.get_best_result().config)
        ax_search._ax.save_to_json_file(filepath='opt.json')

#'''