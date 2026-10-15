import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import numpy as np

from utils import *



class MultivariateDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        
        if dataset == 'ionization_down_down':
            self.dim = [75,31,31]
            self.num_points = 75*31*31
            self.num_frames = 10
            self.data_list = ["H_2", "H+", "He", "PD"]
            self.data_path = '../dataset/ionization_down_down/'
        elif dataset == 'ROCOM_down_down':
            self.num_points = 518349
            self.num_frames = 5
            self.data_list = ['cboron', 'mutl', 'p', 'rhol', 'vl_0', 'vl_1', 'vl_2', 'yplus']
            self.data_path = '../dataset/ROCOM_down_down/'
        
        print(self.data_list)
        

        coords = []
        values = []

        for t in range(1,self.num_frames+1):
            if t == 1:
                temp_coord = np.load(self.data_path+"coordinate.npy")
            coord = np.hstack(((t) * np.ones((temp_coord.shape[0], 1)), temp_coord))
            coords.append(coord)
            
            value = np.load(self.data_path+'data_{:04d}'.format(t-1)+'.npy')
            values.append(value)

        self.coords = np.stack(coords, axis=0)
        self.values = np.stack(values, axis=0)

        print(f"coordinate shape: {self.coords.shape}")
        print(f"value shape: {self.values.shape}")
    
    def get_data_info(self):
        return self.num_points, self.num_frames, self.data_list
    
    def get_structure_data_info(self):
        return self.dim, self.num_frames, self.data_list
    
    def get_original_data(self):
        return self.coords, self.values
    
    def get_norm_data(self):
        norm_coords, norm_values = self.normalization(self.coords, self.values)
        return norm_coords, norm_values
    
    def normalization(self, coords, values):
        norm_coords = np.zeros_like(coords)
        norm_values = np.zeros_like(values)
        
        norm_coords[...,0] = 2*(coords[...,0]-coords[...,0].min())/max(self.num_frames-1, 1)-1
        for i in range(1, 4):
            norm_coords[...,i] = 2*(coords[...,i]-coords[...,i].min())/(coords[...,i].max()-coords[...,i].min())-1
        
        for i in range(values.shape[-1]):
            if values[...,i].max() == values[...,i].min():
                norm_values[...,i] = 2*(values[...,i]-values[...,i].min())-1
            else:
                norm_values[...,i] = 2*(values[...,i]-values[...,i].min())/(values[...,i].max()-values[...,i].min())-1
        
        return norm_coords, norm_values
    
    def get_dataloader(self, batch_size, shuffle, num_workers):
        
        norm_coords, norm_values = self.normalization(self.coords, self.values)
        
        norm_coords = norm_coords.reshape(-1,4)
        norm_values = norm_values.reshape(-1,len(self.data_list))
        
        data_input = torch.FloatTensor(norm_coords)
        data_output = torch.FloatTensor(norm_values)

        data = TensorDataset(data_input, data_output)
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return dataloader
