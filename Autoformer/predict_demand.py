# -*- coding: utf-8 -*-
"""
Demand Forecasting using Autoformer
Based on: https://github.com/thuml/Autoformer

@author: Hyunjun Choi
"""
import torch
import time
import random
import numpy as np
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_demand
from torch.utils.data import DataLoader
import pandas as pd
import os
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
import argparse
from exp.exp_main import Exp_Main#exp stands for experiments
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch import optim


# df_raw = pd.read_csv(os.path.join('dataset/','demand_data.csv'))

# cols_data = df_raw.columns[1:]
# df_data = df_raw[cols_data]

fix_seed = 2021 
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
args = parser.parse_args()
args.use_gpu = True
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

args.freq = 'h'
args.is_training = True
args.checkpoints = './checkpoints/'
args.root_path = './dataset' 
args.data = 'demand'
args.data_path ='Germany_demand_data.csv' 
args.model_id='test'
args.model = 'Autoformer'

args.features = 'S' # M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
args.target = 'Demand'
#forecasting task
args.seq_len = 96
args.label_len = 12  #24 ,12
args.pred_len = 24 #24

#model define
args.enc_in = 1 #encoder input size
args.dec_in = 1 #decoder input size
args.c_out = 1 #'output size'


args.d_model = 512 #dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 3 # num of encoder layers
args.d_layers = 2 #num of decoder layers
args.d_ff = 2048 #dimensino of fcn
args.moving_avg = 25 #window size of moving average
args.factor = 1 #attn factor
args.distil = True #store_false  # whether to use distilling in encoder, using this argument means not using distilling
args.dropout = 0.05
args.embed = 'timeF' #time features encoding, options: [timeF, fixed, learned]
args.activation = 'gelu'
args.output_attention = True #'whether to output attention in encoder'
args.use_multi_gpu = False
args.gpu = 0 

args.des = 'test' 
args.itr = 2 # experiments times
args.patience= 3
args.num_workers =0
args.train_epochs = 15
args.batch_size = 128
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False #use automatic mixed precision training'

ii = 0
print('Args in experiment:')
print(args)


#setting
setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

Exp = Exp_Main
exp = Exp(args)

# exp.train(setting)

#%% Train step
#exp main train
train_data, train_loader = exp._get_data(flag='train')
vali_data, vali_loader = exp._get_data(flag='val')
test_data, test_loader = exp._get_data(flag='test')


seq_x, seq_y, seq_x_mark, seq_y_mark = train_data[0]


path = os.path.join(exp.args.checkpoints, setting)
if not os.path.exists(path):
    os.makedirs(path)
       
time_now = time.time()

train_steps = len(train_loader)
# early_stopping = EarlyStopping(patience=args.patience, verbose=True)

model_optim = exp._select_optimizer()
criterion = exp._select_criterion()

if exp.args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

train_loss_list = []
vali_loss_list = []
test_loss_list = []

for epoch in range(exp.args.train_epochs):
    iter_count = 0
    train_loss = []

    exp.model.train()
    epoch_time = time.time()
  
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        iter_count += 1
        model_optim.zero_grad()
        batch_x = batch_x.float().to(exp.device)

        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        outputs, batch_y = exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())
        if (i + 1) % 100 == 0: 
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((exp.args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        if exp.args.use_amp: #false
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()

    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    vali_loss = exp.vali(vali_data, vali_loader, criterion)
    test_loss = exp.vali(test_data, test_loader, criterion)
    
    train_loss_list.append(train_loss)
    vali_loss_list.append(vali_loss)
    test_loss_list.append(test_loss)

    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    
    # early_stopping(vali_loss, exp.model, path)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

    adjust_learning_rate(model_optim, epoch + 1, exp.args)
    
# Model save
best_model_path = os.path.join(path, 'best_model.pth')
torch.save(exp.model.state_dict(), best_model_path) #checkpoint

#%% predict
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from exp.exp_main import Exp_Main#exp stands for experiments
import argparse
import torch

import pandas as pd
from exp_setup import *  #exp setting
import os


path = './checkpoints/test_Autoformer_demand_ftS_sl96_ll12_pl24_dm512_nh8_el3_dl2_df2048_fc1_ebtimeF_dtTrue_test_0'
# load model
best_model_path = os.path.join(path, 'best_model.pth')
exp = setup_experiment()

exp.model.load_state_dict(torch.load(best_model_path))
exp.model.eval()


df_raw = pd.read_csv(os.path.join('dataset/','Germany_demand_data.csv'))
df_raw = df_raw.iloc[:8760]

# pred_data, pred_loader = exp._get_data(flag='pred')


class custom_Pred(Dataset):
    def __init__(self, dataset, flag='', size=None,
                 features='S', data_path='demand_data.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.dataset = dataset

        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        df_raw = self.dataset
    
        border1s = 0
        border2s = 8760


        if self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_raw[['Demand']][:43800]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        df_stamp = df_raw[['date']][border1s:border2s]
        
        if self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1s:border2s]
        self.data_y = data[border1s:border2s]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

pred_dataset = custom_Pred(
    dataset = df_raw,
    flag='pred',
    size =[96, 12, 24],
    features='S',
    target = 'Demand',
    timeenc= 1,
    freq='H')

# dataset_length = len(pred_dataset)
# last_item = pred_dataset[dataset_length - 1]


A = next(iter(pred_dataset))
print(pred_dataset[0][0].shape) #(96, 1) seq_x
print(pred_dataset[0][1].shape) #(36, 1) seq_y
print(pred_dataset[0][2].shape) #(96, 4) seq_x_mark
print(pred_dataset[0][3].shape) #(36, 4) seq_y_mark


df_array = df_raw.iloc[:, 1].to_numpy()  
sequence_to_find = pred_dataset[0][0].flatten()

pred_data_loader = DataLoader(
    pred_dataset,
    batch_size=1,
    shuffle = False,
    num_workers=0,
    drop_last=True)

pred_iter = next(iter(pred_data_loader))

preds = []
groundtruth = []
historical = []


mse_list = []
rmse_list = []
mae_list = []

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_data_loader):

        batch_x = batch_x.float().to(exp.device) #seqx
        batch_y = batch_y.float() #seqy
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        outputs, batch_y = exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

        pred = outputs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
        batch_x = batch_x.detach().cpu().numpy()
        
        preds.append(pred)
        groundtruth.append(batch_y)
        historical.append(batch_x)
        
        mse = mean_squared_error(batch_y.flatten(), pred.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(batch_y.flatten(), pred.flatten())
        
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        
# mse = np.mean(mse_list)
# mae = np.mean(mae_list)
# rmse = np.mean(rmse_list)
np.save('mse_list.npy', mse_list)
np.save('rmse_list.npy', rmse_list)
np.save('mae_list.npy', mae_list)


