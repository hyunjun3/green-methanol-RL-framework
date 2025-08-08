# -*- coding: utf-8 -*-
"""
@author: Hyunjun Choi
python SARL_evaluation.py --checkpoint-path ./results/ray_results/{experiment_name}/{checkpoint_xxx} --test-case HV_renew_case1 --target-country Germany
"""

import os
import argparse
import pickle
import numpy as np
from ray import tune
from ray.rllib.policy.policy import Policy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils.env.SARL_env_evaluation import *
import random
import pandas as pd
random.seed(2023)

parser = argparse.ArgumentParser()


parser.add_argument("--obs-length", type=int, default=24, help="Observation length of profile data")
parser.add_argument("--X-flow", type=float, default=10000/(0.65702 + 0.19576*55.7), help="Env config: Meoh flow rate")
parser.add_argument("--H2-storage-period", type=float, default=2.65, help="Env config: H2 storage capacity")
parser.add_argument("--ESS-storage-period", type=float, default=87.97, help="Env config: ESS storage capacity")
parser.add_argument("--MEOH-storage-period", type=float, default=17.595, help="Env config: MEOH storage capacity")
parser.add_argument("--op-period", type=int, default=30*24, help="Env config: operation period")
parser.add_argument("--target-country", choices=['South_Korea', 'France', 'Germany'], default='Germany', help="Target country for simulation")
parser.add_argument("--test-case", type=str, choices=['HV_renew_case1', 'HV_demand_case3', 'LV_demand_case4', 'LV_renew_case2'], default='HV_renew_case1', help="Test case scenario")
parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained model checkpoint directory")
parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing data files")
parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save evaluation results")


def register_env(env_name, env_config={}):
    # env = create_env(env_name)
    tune.register_env(env_name,
                      lambda env_config: PTX_env(config=env_config))
    
def wind_power_function(Wind_speed):

    # Turbine model: G-3120
    cutin_speed = 1.5  # [m/s]
    rated_speed = 12  # [m/s]
    cutoff_speed = 25  # [m/s]
    # Wind_speed data is collectd from 50m
    Wind_speed = Wind_speed * (80 / 50) ** (1 / 7)

    idx_zero = Wind_speed <= cutin_speed
    idx_rated = (cutin_speed < Wind_speed) & (Wind_speed <= rated_speed)
    idx_cutoff = (rated_speed < Wind_speed) & (Wind_speed <= cutoff_speed)
    idx_zero_cutoff = (Wind_speed > cutoff_speed)

    Wind_speed[idx_zero] = 0
    Wind_speed[idx_rated] = (Wind_speed[idx_rated] ** 3 - cutin_speed ** 3) / (rated_speed ** 3 - cutin_speed ** 3)
    Wind_speed[idx_cutoff] = 1
    Wind_speed[idx_zero_cutoff] = 0

    return Wind_speed  # Capacity fator =[0,1]

def solar_power_function(Solar_irradiance):
    Ht = Solar_irradiance
    H_ref = 1000  # W/m2
    idx_cutoff = Ht > H_ref
    Ht[idx_cutoff] = H_ref
    n_tot = 0.9375

    return Ht / H_ref * n_tot  # Capacity fator =[0,1]

def curtail_calculation(env):

    Power_profile = env.P_to_G    
    cond_curtail = (Power_profile>0)
    curtail = np.where(cond_curtail, Power_profile, 0)
    curtail = np.sum(curtail)
    
    cond_grid = (Power_profile<=0)
    grid = -np.where(cond_grid, Power_profile, 0) 
    grid = np.sum(grid)
    
    renew = np.sum(env.renewable[23:]) #obs=24 (720-24)
     
    pen = (renew - curtail)/(renew + grid - curtail)
    
    return pen 

#%%
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load test scenario data
    data_path = os.path.join(args.data_dir, f'{args.test_case}.pkl')
    
    try:
        with open(data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        selected_renewable = loaded_data['renewable']
        selected_smp = loaded_data['smp']
        selected_demand = loaded_data['demand']
        selected_p_demand = loaded_data['p_demand']
        
        print(f"Loaded test data from {data_path}")
        print(f"Data shape: {selected_renewable.shape}")
        
    except FileNotFoundError:
        print(f"Error: Test data file not found at {data_path}")
        print("Please ensure the data file exists in the specified directory.")
        exit(1)

    # Environment configuration
    def env_creator(env_config):
        env = PTX_env_RL(env_config)
        return env

    register_env("PTX_env_RL", env_creator)

    env_config = {}
    env_config['X_flow'] = args.X_flow
    env_config['obs_length'] = args.obs_length
    env_config['H2_storage_period'] = args.H2_storage_period
    env_config['ESS_storage_period'] = args.ESS_storage_period
    env_config['MEOH_storage_period'] = args.MEOH_storage_period
    env_config['op_period'] = args.op_period
    env_config['target_country'] = args.target_country
    
    # Load trained policy
    try:
        policy_path = os.path.join(args.checkpoint_path, 'policies', 'default_policy')
        trained_policy = Policy.from_checkpoint(policy_path)
        print(f"Loaded trained policy from {policy_path}")
        
    except Exception as e:
        print(f"Error loading trained policy: {e}")
        print("Please ensure the checkpoint path is correct and contains the policy files.")
        exit(1)

    # Determine sample size based on data shape
    sample_size = selected_renewable.shape[0]
    print(f"Evaluating on {sample_size} scenarios")

    # Initialize result arrays
    profit_list = np.zeros(shape=(sample_size, 1))
    curtail_list = np.zeros(shape=(sample_size, 1))
    
    # Run evaluation
    for i in range(sample_size):
        print(f"Evaluating scenario {i+1}/{sample_size}")
        
        test_env = PTX_env_RL(env_config)
        done = False
        obs = test_env.reset(selected_renewable[i], selected_smp[i], selected_demand[i], selected_p_demand[i])[0]

        while not done:
            action = trained_policy.compute_single_action(obs) 
            clipped_action = np.clip(action[0], test_env.action_space.low, test_env.action_space.high)
            obs, reward, done, _, _ = test_env.step(clipped_action)    

        # Calculate final metrics
        profit_list[i, 0] = np.sum(test_env.cost_list)
        curtail_list[i, 0] = curtail_calculation(test_env)

    # Save evaluation results
    results = {'profit': profit_list,
        'curtailment': curtail_list}
    
    results_path = os.path.join(args.results_dir, f'sarl_evaluation_results_{args.test_case}_{args.target_country}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Evaluation results saved to {results_path}")
