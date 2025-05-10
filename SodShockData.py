import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pickle as pkl
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SodShockData(Dataset):
    def __init__(self, path, n_sols, split: str = "train"):
        super(SodShockData, self).__init__()
        with open(Path(path) / f"{n_sols}_inputs_{split}.pkl", 'rb') as file:
            self.input_data = np.array(pkl.load(file))

        with open(Path(path) / f"{n_sols}_outputs_{split}.pkl", 'rb') as file:
            self.output_data = np.array(pkl.load(file))

        self.train_input_means=[]
        self.train_input_stds=[]
        self.train_output_means=[]
        self.train_output_stds=[]
    def __len__(self):
        return len(self.input_data)

    def normalize_training_data(self):

      for i in range(len(self.input_data[0])):
        mean = self.input_data[:][i].mean()
        std = self.input_data[:][i].std()
        # ensure numerical stability
        if std <= 1e-10:
          std += 1e-6
        self.input_data[:,i] = (self.input_data[:,i]-mean)/std
        self.train_input_means.append(mean)
        self.train_input_stds.append(std)

      for i in range(len(self.output_data[0])):
        mean = self.output_data[:][i].mean()
        std = self.output_data[:][i].std()

        # ensure numerical stability
        if std <= 1e-10:
          std += 1e-6
        self.output_data[:][i] = (self.output_data[:][i]-mean)/std
        self.train_output_means.append(mean)
        self.train_output_stds.append(std)

    def normalize_test_data(self, train_input_means, train_input_stds, train_output_means, train_output_stds):
      for i in range(len(self.input_data[0])):
        mean = train_input_means[i]
        std = train_input_stds[i]
        self.input_data[:][i] = (self.input_data[:][i]-mean)/std

      for i in range(len(self.output_data[0])):
        mean = train_output_means[i]
        std = train_output_stds[i]
        self.output_data[:][i] = (self.output_data[:][i]-mean)/std

    def unnormalize_training_data(self):
      for i in range(len(self.input_data[0])):
        mean = self.train_input_means[i]
        std = self.train_input_stds[i]
        self.input_data[:][i] = self.input_data[:][i]*std + mean

      for i in range(len(self.output_data[0])):
        mean = self.train_output_means[i]
        std = self.train_output_stds[i]
        self.output_data[:][i] = self.output_data[:][i]*std + mean

    def unnormalize_test_data(self, train_input_means, train_input_stds, train_output_means, train_output_stds):
      for i in range(len(self.input_data[0])):
        mean = train_input_means[i]
        std = train_input_stds[i]
        self.input_data[:][i] = self.input_data[:][i]*std + mean

      for i in range(len(self.output_data[0])):
        mean = train_output_means[i]
        std = train_output_stds[i]
        self.output_data[:][i] = self.output_data[:][i]*std + mean


    def __getitem__(self, idx:int):
        inp = self.input_data[idx]
        x, p, rho, u = inp
        input_data = torch.stack((torch.as_tensor(x),torch.as_tensor(p),
                                  torch.as_tensor(rho),torch.as_tensor(u)), dim=-1)

        out = self.output_data[idx]
        p, rho, u = out
        output_data = torch.stack((torch.as_tensor(p),torch.as_tensor(rho),
                                   torch.as_tensor(u)), dim=-1)
        return (input_data.float().to(device), output_data.float().to(device))