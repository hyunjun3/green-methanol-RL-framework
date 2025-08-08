# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:32:29 2024

@author: USER
"""

import argparse
from exp.exp_main import Exp_Main#exp stands for experiments
import torch
import random
import numpy as np
import os


def setup_experiment():
    fix_seed = 2021 
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    args = parser.parse_args()
    #gpu 설정
    args.use_gpu = True
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    args.freq = 'h'
    args.is_training = True
    args.checkpoints = './checkpoints/'
    args.root_path = './dataset' 
    args.data = 'demand'
    args.data_path ='demand_data.csv' 
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
    # 모델에 맞게 수정
    
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
    
    return exp
