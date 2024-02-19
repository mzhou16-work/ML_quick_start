import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import pandas as pd
import scipy.stats   as stats
import scipy.special as special

import matplotlib.pyplot as plt

BATCH_SIZE = 10000
NUM_WORKERS = 8

#did not include 'cal_sca_ang' method used to calculate scattering geometry of each pixel
def normalization(data, name, p = None, operation = 'forward', adj_params = []):
    
    def normalize_angle(angle):
        return np.mod(angle, 2.0*np.pi)
    
    def nBoxCox(x):
        '''
        Returns normalized (z-scores) Box-Cox transform.
        '''
        (x_, p_) = stats.boxcox(x)
        p = (p_, x_.mean(), x_.std())
        x_ = (x_ - p[1]) / p[2]
        
        return x_, p
    
    def inv_nBoxCox(x,p):
        '''
        Inverse normalized (z-scores) BoxCox transform.
        '''
        
        x_ = p[1] + p[2]*x
        x_ = special.inv_boxcox(x_, p[0])
        
        return x_, p
    '''
    def imultFigure(nRow, nCol, **kwargs):
        
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patheffects as path_effects
        
        nUint = kwargs.get('nUint', 25)
        nColGap = kwargs.get('nColGap', 5)
        nRowGap = kwargs.get('nRowGap', 5)
        nPlot = kwargs.get('nPlot', None)
        
        figsize = kwargs.get('figsize', (9,9))
        fontsize = kwargs.get('fontsize', 22)
        xlabel = kwargs.get('xlabel', 0.1)
        ylabel = kwargs.get('ylabel', 0.975)
        numOn = kwargs.get('numOn', False)
        
        cord = kwargs.get('cord', [90, -90, -180, 180])
        
        nRow_grid = nRow*nUint
        nCol_grid = nCol*nUint
        
        gs = gridspec.GridSpec(nRow_grid, nCol_grid)
        
        axes = []
        fig = plt.figure(figsize=figsize)
        
        if nPlot == None:
            nPlot = nRow * nCol - 1
        else:
            nPlot = nPlot - 1
        
        for i in range(nRow):
            for j in range(nCol):
                nFigure = i * nCol + j
                
                if nFigure <= nPlot:
                    txt_number = '(' + chr(97 + nFigure) + ')'
                    instant = fig.add_subplot(gs[i*nUint:(i + 1)*nUint - nColGap, j*nUint:(j+1)*nUint - nRowGap])
                    
                    if numOn:
                        text = instant.annotate(txt_number, xy = (xlabel, ylabel), xytext = (xlabel, ylabel), textcoords='axes fraction', color = 'k', ha='right', va='top', fontsize=fontsize)
                        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'), path_effects.Normal()])
                    axes.append(instant)
                    
        return [fig, axes]
    
    rows = len(name)
    cols = 3
    fontsize = 12
    fig, ax = imultFigure(rows, cols, figsize=(rows*5, cols*5), nColGap=6, nRowGap=6)
    fig.set_facecolor('white')
    fig.suptitle('Features Before and After Box-Cox/Inverse Box-Cox Transformations', fontsize=fontsize)
    '''
    if operation == 'forward':
        #for training
        
        if data.ndim == 1:
            
            x_, p_ = nBoxCox(data)
            p = [p_]
            print(f' - Normalize {name[i]} ... coef: {p_}')
        
        if data.ndim == 2:
            p = []
            _, n_loops = data.shape
            x_ = np.full_like(data, np.nan)
            
            print('n_loops', n_loops)
            for i in range(n_loops):
                '''
                #d_min = np.nanmin(data[:,i])
                #d_max = np.nanmax(data[:,i])
                
                v_mean = np.nanmean(data[:,i])
                v_std = np.nanstd(data[:,i])
                v_min = np.max((v_mean - v_std*3), 0)
                v_max = v_mean + v_std*3
                bins = np.linspace(v_min, v_max, 100)
                bins_norm = np.linspace(-3, 3, 100)
                
                _, _, _ = ax[(3*i)].hist(data[:,i], bins=bins, fc='k', density=True, alpha=0.5)
                ax[(3*i)].set_title(str(name[i]), fontsize=fontsize)
                ax[(3*i)].tick_params(labelsize=fontsize)
                #ax[(2*i)].set_xlim(round(v_min, 0), round(v_max, 0))
                '''
                print(f' - Normalize {name[i]}, {np.nanmin(data[:,i])}')
                
                if name[i] in adj_params:
                    offset = 50.
                else:
                    offset = 0.
                
                x_[:,i], p_ = nBoxCox(data[:,i] + offset)
                print(f' - Normalize {name[i]}, coef: {p_}')
                
                '''
                _, _, _ = ax[(3*i)+1].hist(x_[:,i], bins=bins_norm, fc='k', density=True, alpha=0.5)
                ax[(3*i)+1].set_title(str(name[i]) + ' Norm', fontsize=fontsize)
                ax[(3*i)+1].tick_params(labelsize=fontsize)
                #ax[(3*i)+1].set_xlim(-3,3)
                '''
                p.append(p_)
                '''
                new_data, p_ = inv_nBoxCox(x_[:,i], p_)
                
                _, _, _ = ax[(3*i)+2].hist(new_data, bins=bins, fc='k', density=True, alpha=0.5)
                ax[(3*i)+2].set_title(str(name[i]) + ' After Inv BoxCox', fontsize=fontsize)
                ax[(3*i)+2].tick_params(labelsize=fontsize)
                
                ax[i].scatter(data[:,i], new_data, color='firebrick', alpha=0.5, label='Transformed vs. Original')
                ax[i].plot([d_min, d_max], [d_min, d_max], color='black', linestyle='--', label = 'y = x')
                ax[i].set_xlim(d_min, d_max)
                ax[i].set_ylim(d_min, d_max)
                ax[i].set_xlabel('Original Data', fontsize=fontsize)
                ax[i].set_ylabel('Data After Transformations', fontsize=fontsize)
                ax[i].set_title(f'{name[i]}', fontsize=fontsize)
                ax[i].tick_params(labelsize=fontsize)
                
            
            if n_loops > 1:
                plt.savefig('testwithinv.png', bbox_inches='tight', dpi=300)
    '''
    if operation == 'predict':
        
        if data.ndim == 1:
            
            if name[i] in adj_params:
                offset = 50.
            else:
                offset = 0.
            
            key = name[i]
            ps = list(p[key])
            x_ = special.boxcox(data+offset, ps[0])
            x_ = (x_ - ps[1]) / ps[2]
            print(f' - Normalization [Predict Mode], {name[i]} ...')
        
        if data.ndim == 2:
            
            _, n_loops = data.shape
            
            x_ = np.full_like(data, np.nan)
            
            for i in range(n_loops):
                
                if name[i] in adj_params:
                    offset = 50.
                else:
                    offset = 0.
                
                key = name[i]
                ps = list(p[key])
                
                x_[:,i] = special.boxcox(data[:,i] + offset, ps[0])
                x_[:,i] = (x_[:,i] - ps[1]) / ps[2]
                print(f' - Normalization [Predict Mode], {name[i]} ...')
    
    if operation == 'inverse':
        
        x_, p = inv_nBoxCox(data, p)
    
    return x_, p


def read_land_data(filename, qa_dict, th_dict, name_dict, random_seed=87):
    
    label_name = name_dict['label_name']
    norm_feat_name = name_dict['norm_feat_name']
    encode_feat_name = name_dict['encode_feat_name']
    
    #Read the dataset
    print(f' - Reading {filename} ...')
    df = pd.read_csv(filename)
    
    print(f' - Size of the raw data {len(df)}')
    df = df.dropna()
    print(f' - Size of the valid raw data {len(df)}')
    
    df['Aspect'] = df['Aspect']*np.pi/180.0
    
    #Use local temp 40? as a threshold to segregate data
    threshold = 40
    print(f' - Using threshold {threshold} to segregate')
    idx_clean = np.where(df['Local']<threshold)
    print(f' - # of samples < {threshold}: {np.shape(idx_clean)[1]}')
    idx_pollute = np.where(df['Local']>=threshold)
    print(f' - # of samples >= {threshold}: {np.shape(idx_pollute)[1]}')
    
    indices = list(range(np.shape(idx_clean)[1]))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    if np.shape(idx_pollute)[1]*3 > np.shape(idx_clean)[1]:
        pass
    else:
        df = [df.iloc[idx_clean].iloc[indices[0: np.shape(idx_pollute)[1]*3]], df.iloc[idx_pollute]]
        df = pd.concat(df)
    
    print(f'************************************************************')
    print(f'                   After adjusting')
    print(f'************************************************************')
    idx_clean = np.where(df['Local']<threshold)
    print(f' - # of samples < {threshold}: {np.shape(idx_clean)[1]}')
    idx_pollute = np.where(df['Local']>=threshold)
    print(f' - # of samples >= {threshold}: {np.shape(idx_pollute)[1]}')
    
    #For now, skipping parts seeming to deal with NDVI and angles
    
    idx = (df['Local'] == df['Local'])
    
    for feat in name_dict['norm_feat_name']:
        print(f' - {feat}: {np.nanmin(df[feat].values)}, {np.nanmax(df[feat].values)}')
    
    print(f' - Filtering invalid label ...')
    for label in label_name:
        idx = idx & (df[label].values > 0)
    
    print(f' - Applying threshold to segregate data ...')
    for th_var in th_dict.keys():
        idx = idx & (df[th_var].values == df[th_var].values) & (df[th_var].values > th_dict[th_var][0]) & (df[th_var].values <= th_dict[th_var][1])
    
    print(f' - Applying QA to filtering data ...')
    for qa_var in qa_dict.keys():
        idx = idx & (df[qa_var].values == df[qa_var].values) & (df[qa_var].values > qa_dict[qa_var][0]) & (df[qa_var].values <= qa_dict[qa_var][1])
    
    df = df[idx]
    
    print(f' - In total training data size: {len(df)}')
    
    # Apply one hot encoding to category feature
    one_hot_code = []
    for name in encode_feat_name:
        print(f' - Applying one-hot encoding to feature {name} ...')
        one_hot = df[name].values.astype(int)
        one_hot = F.one_hot(torch.from_numpy(one_hot.reshape(-1,)), num_classes=20).detach().numpy()
        one_hot_code.append(one_hot)
    
    print(f'')
    #Normalize the labels
    print(f' - Applying normalization to labels ...')
    
    normed_label, coef_label = normalization(df[label_name].values, label_name)
    label_dict = {}
    for name, coef in zip(label_name, coef_label):
        label_dict[name] = coef
    
    print(f'')
    # Normalize the features
    print(f' - Applying normalization to features ...')
    
    adj_params = []
    
    feature_dict = {}
    normed_feat, coef_feat = normalization(df[norm_feat_name].values, norm_feat_name, adj_params=adj_params)
    for name, coef in zip(norm_feat_name, coef_feat):
        feature_dict[name] = coef
    
    normed_feat = [normed_feat]
    for data in one_hot_code:
        normed_feat.append(data)
    
    normed_feat = np.concatenate(normed_feat, axis = 1)
    
    order_dict = {}
    order_dict['order_label'] = label_name
    order_dict['order_feat'] = norm_feat_name
    order_dict['order_one_hot_feat'] = encode_feat_name
    
    print(f'')
    print(f' - Dimension of feature: {normed_feat.shape}')
    print(f' - Dimension of label: {normed_label.shape}')
    
    return normed_label, normed_feat, label_dict, feature_dict, order_dict

#-----------------------------------------------------------------------
def split_data(dataset_size, validation_split, test_split, random_seed=42, shuffle_dataset=True):
    
    print(f' - Receive {dataset_size} data')
    print(f' - Split {(1-validation_split-test_split)*100}% to training, {(validation_split)*100}% to validation, {(test_split)*100}% to testing')
    
    indices = list(range(dataset_size))
    
    train_split = int(np.floor((1-validation_split-test_split)*dataset_size))
    validation_split = int(np.floor((1-test_split)*dataset_size))
    test_split = int(np.floor(test_split*dataset_size))
    
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:validation_split]
    test_indices = indices[validation_split:]
    
    print(f' - Training: {np.shape(train_indices)[0]}, Validation: {np.shape(val_indices)[0]}, Testing: {np.shape(test_indices)[0]}')
    return train_indices, val_indices, test_indices

#-----------------------------------------------------------------------
class TorchDataset(Dataset):
    
    def __init__(self, label, feature):
        
        self.y = torch.from_numpy(label)
        self.x = torch.from_numpy(feature)
        
        print(f' - Shape of the label: {np.shape(self.y)}')
        print(f' - Shape of the features: {np.shape(self.x)}')
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#-----------------------------------------------------------------------
class DATA(LightningDataModule):
    
    def __init__(self, data, indices, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
        
        super().__init__()
        
        self.batch_size = batch_size
        
        self.num_workers = num_workers
        
        self.train_indices = indices[0]
        self.val_indices = indices[1]
        self.test_indices = indices[2]
        
        self.label = data[0]
        self.feat = data[1]
        
        self.in_channels = self.feat.shape[-1]
        self.out_channels = self.label.shape[-1]
        
        print(f' - Forming training dataset\n')
        self.train_dataset = TorchDataset(self.label[self.train_indices], self.feat[self.train_indices])
        
        print(f' - Forming validation dataset\n')
        self.val_dataset = TorchDataset(self.label[self.val_indices], self.feat[self.val_indices])
        
        print(f' - Forming testing dataset\n')
        self.test_dataset = TorchDataset(self.label[self.test_indices], self.feat[self.test_indices])
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)