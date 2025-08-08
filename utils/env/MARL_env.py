"""
@author: Hyun jun Choi
"""
#%% Class 
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete, MultiDiscrete, Tuple
from copy import copy
import math
import random
import os
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from utils.env.modeling.methanol_SOFC import SOFC, SOFC_output
from torch.utils.data import Dataset
from ray.rllib.policy.policy import PolicySpec
from collections import OrderedDict


class PTX_env(MultiAgentEnv): #PtX power allocation model 

    def __init__(self, config = None):
        # MEOH 
        self.X_H2 = 0.19576         # specific H2 consumption for "X" production，kgH2/s / kgX/s, X is methanol
        self.X_CO2 = 1.435802       # specific CO2 consumption for "X" production，kgCO2/s / kgX/s, X is methanol
        self.P_X = 0.65702          # specific power consumption for "X" production，kW/kg/h, X is methanol
        self.C_CO2 = 50             # CO2 purchase cost, $/ tonne
        #Hydrogen
        self.SP_H2 = 55.7           # specific power consumption for H2 production，kW/kgH2/h
        self.SPC_H2 = 55.7+3.03     # specific power consumption for H2 production and compression，kW/kgH2/h
       
         # === Economic Parameters ===
        self.CAP_H2 = 751700     # H2 storage cost ($/ton)
        self.CAP_PEM = 600       # PEM cost ($/kW)
        self.CAP_solar = 740     # Solar CAPEX ($/kW)
        self.CAP_wind = 1250     # Wind CAPEX ($/kW)
        self.OPEX_solar = 12.6   # Solar OPEX ($/kW)
        self.OPEX_wind = 25.0    # Wind OPEX ($/kW)
        self.BESS_capacity_cost = 236.5  # BESS CAPEX ($/kW)
        
        self.H2_price = 5        # H2 selling price ($/kg) (Ref: powermag.com/blog/hydrogen-prices-skyrocket-over-2021-amid-tight-power-and-gas-supply/)
        self.X_price = 0.45      # Methanol price ($/kg)  (Ref: https://www.indexbox.io/search/methanol-price-per-kg/)
        self.scale = 50000       # Plant scale (kW)
        
        # === Efficiency ===
        self.ESS_eff = 0.95
        self.PEM_eff = 0.68

        # Design specification          
        self.X_flow = config.get('X_flow', 1000)
        self.X_flow_P_cap = self.X_flow * self.P_X #kWh # 568.31441576
        H2_storage_period = config.get('H2_storage_period', 24)
        ESS_storage_period = config.get('ESS_storage_period', 24)
        self.H2_cap =  self.X_flow*self.X_H2*H2_storage_period                
        self.ESS_cap = (self.X_flow_P_cap)*ESS_storage_period
        self.ESS_P_cap = (self.X_flow_P_cap*(ESS_storage_period+1))* 0.3
        self.PEM_P_cap = self.X_flow*self.X_H2*self.SP_H2*(H2_storage_period+1)   
        
        #individual, global config
        self.reward_global = config.get('reward_global', True)
        
        #MEOH storage design spec
        MEOH_storage_period = config.get('MEOH_storage_period', 24) #17.595890536056736
        self.MEOH_cap = self.X_flow * MEOH_storage_period 
        self.MEOH_tank_cap1 = self.MEOH_cap #area1 
        self.MEOH_tank_cap2 = self.MEOH_cap #area2
        
        self.fw = config.get('fw', 0.5)
        self.op_period = config.get('op_period', 720)  
        self.target_country = config.get('target_country', 'Germany') #['France', 'Germany']
        
        # === Battery desgin spec ===
        self.self_discharge_rate = 0.05 / self.op_period  # 5% per 30 days (720 hours)
        self.daily_limit_cycle = 4
        self.lifetime_cycle = 3500
        self.cumulative_cycles = 0
        self.daily_cycles = 0
        self.initial_capacity = self.ESS_cap
        self.initial_capacity_new = self.X_flow_P_cap * ESS_storage_period
        self.current_hour = 0
        self.self_discharge_history = []

        #SOFC
        self.SOFC = SOFC
        self.SOFC_output = SOFC_output
        
        #CO2 factor
        self.emission_factor = 0.5 #kg/kWh
        self.carbon_emission = 0
        self.carbon_capture = 0
        self.emission_diff = 0
        self.norm_emission_diff = 0

        max_list = []
        min_list = []
        
        path = os.getcwd()
    
        for i in range(2024-2015):
            year = i + 2015
            SMP_file_path = os.path.join(path, 'data/{country}/SMP_data/{year}.npy'.format(country = self.target_country, year = year))
            SMP = np.load(SMP_file_path, allow_pickle= True).astype(np.float64)
            max_list.append(np.max(SMP))
            min_list.append(np.min(SMP))
        self.max_SMP = max(max_list)
        self.min_SMP = min(min_list)
        del max_list, min_list, SMP, year
        
        #  demand data min, max
        d_max_list=[]
        d_min_list=[]
        for i in range(2024-2015):
            year = i + 2015
            file_path  = os.path.join(path, 'data/{country}/demand_data/Demand_data_{year}.npy'.format(country = self.target_country, year=year))
            demand_data = np.load(file_path).astype(np.float64)
            demand_data = np.sum(demand_data, axis=0)
            d_max_list.append(np.max(demand_data))
            d_min_list.append(np.min(demand_data))
        self.max_demand = max(d_max_list)
        self.min_demand = min(d_min_list)
        del d_max_list, d_min_list, demand_data, year
        
        #pred demand data min, max
        pred_max_list=[]
        pred_min_list=[]
        for i in range(2024-2015):
            year = i + 2015
            file_path  = os.path.join(path, 'data/{country}/demand_data/Pred_data_{year}.npy'.format(country = self.target_country, year=year))
            pred_demand_data = np.load(file_path).astype(np.float64)
            pred_demand_data = pred_demand_data.reshape(-1)
            pred_demand_data[pred_demand_data < 0] = 0
            pred_max_list.append(np.max(pred_demand_data))
            pred_min_list.append(np.min(pred_demand_data))
        self.pred_max_demand = max(pred_max_list)
        self.pred_min_demand = min(pred_min_list)
        del pred_max_list, pred_min_list, pred_demand_data, year
               
        #agent
        self.agents= {"PTX_agent", "Management_agent"}
        self.agent_ids = set(self.agents)
        self._skip_env_checking = True
        self.obs_length = config.get('obs_length', 24) 
        
        #State and action space
        #State1 = (renewable profile, SMP,  and SOC (%) and H2_storage tank (%)) = 24+24+2
        #State2 = (Historical demand , Predict demand, Methanol storage tank level(%), Methanol storage tank level2 (%))= 24+24+2

        space_low = np.zeros(shape=(self.obs_length * 2 + 2), dtype=np.float32) # 24+24+2
        space_high = np.zeros(shape = (self.obs_length*2 + 2),dtype = np.float32) # 24+24+2
        space_high[:] = 1
        self.observation_space = Dict({
        "PTX_agent": Box(low=space_low, high=space_high, dtype=np.float32), #50
        "Management_agent": Box(low=space_low, high=space_high, dtype=np.float32),}) #50

        # action_space : ESS: [-1, 1], PEM[0,1], stored H2 utilization[0,1], H2 to market[0,1]
        PTX_agent_action_space_low = np.array([-1, 0, 0, 0], dtype=np.float32)  
        PTX_agent_action_space_high = np.array([1, 1, 1, 1], dtype=np.float32)
        # action_space : MEOH transport [0, 1], stored MEOH utilization[0,1]
        Management_agent_action_space_low = np.array([0, 0], dtype=np.float32)
        Management_agent_action_space_high = np.array([1, 1], dtype=np.float32)
        
        self.action_space = Dict({
            "PTX_agent": Box(low=PTX_agent_action_space_low, high=PTX_agent_action_space_high, shape=(4,), dtype=np.float32),
            "Management_agent": Box(low=Management_agent_action_space_low, high=Management_agent_action_space_high, shape=(2,), dtype=np.float32),
        })              

        #Initialize list ans state
        self.penalty_weight = config.get('penalty_weigth', 1)  
        self.cost_weight = config.get('cost_weigth', 1)
        
        self.action_acc = []
        self.ESS_charge = []
        self.ESS_discharge = []
        self.grid_acc = []
        self.L_H2 = np.zeros(1)
        self.X_acc = []
        self.SOC =  np.zeros(1)
        self.L_MEOH_1 = np.zeros(1)
        self.L_MEOH_2 = np.zeros(1)
        self.penalty = 0
        self.step_count = 0
        self.trans_error = []
        
    def distillation_cost(self):
        # Column diameter
        D = ((4/3.14/0.761) *(self.X_flow/32) *2 *22.4 * (64+273)/273 *1 * 1/3600)**0.5
        # Column length
        L = 0.61 * 38 + 4.27
        # Column vessel cost
        CC = 17640 * D**1.066 * L**0.802
        # Tray cost
        TC = 229 * D**1.55 *38
        # Heat exchanger cost
        ConC = 7296 * (1063* self.X_flow/96872.7)**0.65
        ExC = 7296 * (3109* self.X_flow/96872.7)**0.65
        # Compressor cost
        cmpC = 5840 * (23238.8* self.X_flow/96872.7)**0.82
        Capex = CC+ TC +ConC + ExC + cmpC
        return Capex
          
    def _RESET(self, seed=None, options=None):
        self.step_count = 0
        self.action_acc = []
        self.ESS_charge = []
        self.ESS_discharge = []
        self.grid_acc = []        
        self.X_acc = []      

        self.normalized_grid_penalty_list = []
        self.cost_list = []
        self.production_cost_list = []
        self.reward_list = []
        self.step_reward = 0
        self.total_reward = 0
        self.step_count = 0
        self.PTX_reward_list = []
        self.Management_reward_list = []

        self.L_H2[0] = 0
        self.SOC[0] =  0
        self.L_MEOH_1[0] = 0
        self.L_MEOH_2[0] = 0
        self.L_H2_init = self.L_H2[0]
        self.SOC_init = self.SOC[0]
        
        # Reset BESS state variables
        self.cumulative_cycles = 0
        self.daily_cycles = 0
        self.current_hour = 0
        self.ESS_cap = self.initial_capacity
        
        # Set random seed and sample a random year
        random.seed(seed)
        np.random.seed(seed)
        year = random.randint(2015, 2022)
        path = os.getcwd()
        print('Reset the renewable profile: ', year, self.target_country)
        
        # Load wind data
        wind_file_path = os.path.join(path, 'data/{country}/Renewable_data/Wind_RL/Wind_data_{year}.npy'.format(country = self.target_country, year = year))
        wind_data = np.load(wind_file_path)
        idx = np.random.randint(wind_data.shape[0], size=1)
        wind_data = wind_data[idx][0]
        wind_data[np.where(wind_data<0)] = 0
        
        # Load solar data
        solar_file_path = os.path.join(path, 'data/{country}/Renewable_data/Solar_RL/Solar_data_{year}.npy'.format(country = self.target_country, year = year))
        solar_data = np.load(solar_file_path)
        idx = np.random.randint(solar_data.shape[0], size=1)
        solar_data = solar_data[idx][0]
        solar_data[np.where(solar_data<0)] = 0
        
        # Calculate combined renewable generation
        self.wind_power = self.wind_power_function(wind_data)
        self.solar_power = self.solar_power_function(solar_data)  
        self.renewable = self.scale * (self.wind_power * self.fw + self.solar_power * (1 - self.fw))
        del wind_data, solar_data
        
        #Sampling SMP profile from downloaded data
        SMP_file_path = os.path.join(path, 'data/{country}/SMP_data/{year}.npy'.format(country = self.target_country, year = year))
        total_SMP = np.load(SMP_file_path,allow_pickle= True).astype(np.float64)
        total_SMP[np.where(total_SMP<0)] = 0
        self.SMP = total_SMP
 
        #demand profile
        demand_file_path = os.path.join(path,'data/{country}/demand_data/Demand_data_{year}.npy'.format(country = self.target_country, year=year))
        self.demand_data = np.load(demand_file_path)
        self.demand_data = np.sum(self.demand_data, axis=0) 
        self.demand_data[np.where(self.demand_data<0)] = 0    
        
        #predict demand profile
        p_demand_file_path = os.path.join(path, 'data/{country}/demand_data/Pred_data_{year}.npy'.format(country = self.target_country, year=year))
        self.p_demand_data = np.load(p_demand_file_path)
        self.p_demand_data = self.p_demand_data.reshape(-1)
        self.p_demand_data[self.p_demand_data < 0] = 0
        
        # Profile sampling
        max_idx = (len(self.p_demand_data) // 24) - self.op_period
        idx = random.randint(96, max_idx)
        self.renewable = self.renewable[idx:idx+self.op_period]
        self.SMP = self.SMP[idx:idx+self.op_period]
        self.demand_data = self.demand_data[idx:idx+self.op_period]
        self.p_demand_data = self.p_demand_data[idx*24 : (idx+self.op_period)*24]
        
        # Initialize state   
        self.obs = self._obs()
                      
        self.P_to_G = np.zeros(len(self.renewable)-self.obs_length+1)
        self.SOC_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        self.SOC_profile[self.step_count] = self.SOC[0]      
        self.L_H2_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        self.L_H2_profile[self.step_count] = self.L_H2[0]        
        self.X_profile = np.zeros(len(self.renewable)-self.obs_length+1)
        self.ptx_CO2_list = np.zeros(len(self.renewable)-self.obs_length+1)
        self.P_consum_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        # self.error_profile = np.zeros(shape = (len(self.renewable)-self.obs_length+2, 6))
        self.H2_to_market = np.zeros(len(self.renewable)-self.obs_length+1)
        #area1 MEOH storage tank
        self.MEOH_profile_1 = np.zeros(len(self.renewable)-self.obs_length+2)
        self.MEOH_profile_1[self.step_count] = self.L_MEOH_1[0]
        #area2 MEOH storage tank
        self.MEOH_profile_2 = np.zeros(len(self.renewable)-self.obs_length+2)
        self.MEOH_profile_2[self.step_count] = self.L_MEOH_2[0]
        #SOFC 
        self.SOFC_profile= np.zeros(len(self.renewable)-self.obs_length+2)
        self.MEOH_transport_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        self.pred_demand_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        self.MEOH_to_market = np.zeros(len(self.renewable)-self.obs_length+2)
        self.MEOH_to_market2 = np.zeros(len(self.renewable)-self.obs_length+2)
        self.MEOH_to_market_total = np.zeros(len(self.renewable)-self.obs_length+2)
        
        self.pred_values = np.zeros((len(self.renewable) - self.obs_length + 1, 24))

        # Scaling factor
        self.ESS_penalty_factor = 1/self.ESS_P_cap
        self.PEM_penalty_factor = 1/self.PEM_P_cap 
        self.area1_tank_factor = 1/self.MEOH_tank_cap1 
        self.area2_tank_factor = 1/self.MEOH_tank_cap2 
        self.demand_penalty_factor = 1/ self.max_demand

        self.cost_factor = self.cost_scale()
        
        # supply and demand
        self.renew_supply = np.zeros(len(self.renewable)-self.obs_length+1)
        self.grid_supply = np.zeros(len(self.renewable)-self.obs_length+1)
        self.P_demand = np.zeros(len(self.renewable)-self.obs_length+1)
        self.H_demand = np.zeros(len(self.renewable)-self.obs_length+1) 
        self.Building_demand = np.zeros(len(self.renewable)-self.obs_length+1)
        self.grid_price = np.zeros(len(self.renewable)-self.obs_length+1)
        self.BESS_usage = np.zeros(len(self.renewable)-self.obs_length+1)
  
        return self.obs, {}     


    def _STEP(self, action_dict): #Action1 = 'ratio' of  ESS power & PEM power  & LH2 utilization & H2 split_fraction to market
                                  #ACtion2 = MEOH transport [0, 1], stored MEOH utilization[0,1]  
    
        PTX_agent_action = np.squeeze(np.array(action_dict.get("PTX_agent", [0, 0, 0, 0]), dtype=np.float32)) #action_dict.get(key, default)
        Management_agent_action = np.squeeze(np.array(action_dict.get("Management_agent", [0, 0]), dtype=np.float32))

        if PTX_agent_action.ndim != 1 or PTX_agent_action.shape[0] != 4:
            raise ValueError("PTX_agent_action must be a one-dim array with 4 elements.")

        if Management_agent_action.ndim != 1 or Management_agent_action.shape[0] != 2:
            raise ValueError("Management_agent_action must be a one-dim array with 2 elements.")
        
        ESS_action, PEM_power_action, LH2_util, split_fraction = PTX_agent_action
        transport_frac, LMEOH2_util = Management_agent_action    
    
        ESS_action = ESS_action * self.ESS_P_cap    
        PEM_power_action = PEM_power_action * self.PEM_P_cap   
           
            
        self.ESS_penalty = 0
        self.PEM_penalty = 0   
        self.grid_penalty = 0
        self.demand_penalty = 0
        self.tank_penalty = 0
        self.tank_penalty2 = 0

        X_load = self.X_flow_P_cap #X_flow_P_cap = total energy consumption for over a certain period
        
        #Mass balance
        H_mis = X_load/self.P_X*self.X_H2 - self.L_H2[0]*LH2_util 
        
        if H_mis >= 0: # When hydrogen is insufficient
            self.L_H2[0] = self.L_H2[0]*(1-LH2_util)

        else: 
            # # When hydrogen is sufficient
            self.PEM_penalty += (self.L_H2[0]*LH2_util- X_load/self.P_X*self.X_H2)*self.SP_H2
            self.L_H2[0] -= X_load/self.P_X*self.X_H2
            H_mis = 0
        
        # Power balance
        P_ptx = X_load
        ptx_H2 = H_mis*self.SP_H2
        ptx_CO2 = X_load/self.P_X*self.X_CO2         
        P_consum = P_ptx+ ptx_H2    
        P_mis = self.renewable[self.step_count+self.obs_length-1] - P_consum
        
        # Supply and demand
        self.renew_supply[self.step_count] = self.renewable[self.step_count+self.obs_length-1] 
        self.P_demand[self.step_count] = P_ptx
        self.H_demand[self.step_count] = ptx_H2
        self.Building_demand[self.step_count] =  self.demand_data[self.step_count+self.obs_length-1]
        self.grid_price[self.step_count] = self.SMP[self.step_count+self.obs_length-1]
        
        current_data_idx = self.step_count + self.obs_length - 1
        pred_start_idx = current_data_idx * 24
        pred_end_idx = pred_start_idx + 24
        
        if pred_end_idx <= len(self.p_demand_data):
            self.pred_values[self.step_count] = self.p_demand_data[pred_start_idx:pred_end_idx]
        else:  # If episode ends and not enough 24-hour prediction data
            self.pred_values[self.step_count] = self.p_demand_data[len(self.p_demand_data)-24:] 
            
        L_H2_prev = self.L_H2[0]
        SOC_prev = self.SOC[0]
        L_MEOH_1_prev = self.L_MEOH_1[0]
        L_MEOH_2_prev = self.L_MEOH_2[0]
        
        self.P_to_G[self.step_count] += P_mis # P_mix > 0  = surplus renew power
        
        # Update BESS state
        self.update_BESS_state()
        
        ESS_action = self.ESS_masking(ESS_action)
        self.P_to_G[self.step_count] += -ESS_action  # -ESS_action: discharging the ESS
        
 
        if PEM_power_action<0:
            self.PEM_penalty += -PEM_power_action 
            PEM_power_action = 0
            
        if H_mis>0:           
            if H_mis*self.SP_H2>PEM_power_action: 
                self.PEM_penalty += (H_mis*self.SP_H2 - PEM_power_action) 
                PEM_power_action = H_mis*self.SP_H2
            else:
                pass
        
        H2_produce = PEM_power_action/self.SP_H2 - H_mis
        
        if H2_produce<0:
            H2_produce = 0
        
        H2_to_sell = H2_produce*split_fraction
        H2_to_storage = H2_produce*(1-split_fraction)
        
        if H2_to_storage + self.L_H2[0] >= self.H2_cap: # Overcharge
            self.PEM_penalty += (H2_to_storage + self.L_H2[0]-self.H2_cap)*self.SP_H2 # overcharge penalty
            H2_to_storage = self.H2_cap-self.L_H2[0] 
            
        H2_to_sell = H2_produce - H2_to_storage
        self.P_to_G[self.step_count] += -H2_produce*self.SP_H2-H2_to_storage*(self.SPC_H2-self.SP_H2) 
        self.L_H2[0] += H2_to_storage
        
        
        if ESS_action>0:
            ESS_store = ESS_action
        else:
            self.BESS_usage[self.step_count] = -ESS_action
            ESS_store = 0
            
        # Store produced MeOH into tank 1
        produced_methanol = self.X_flow
        self.L_MEOH_1[0] += produced_methanol
        
        # area1 methanol storage tank
        MEOH_sell = 0
        if self.L_MEOH_1[0] > self.MEOH_tank_cap1: # Apply MEOH overcharge penalty
            MEOH_sell = self.L_MEOH_1[0] - self.MEOH_tank_cap1 # Sell excess methanol that cannot be stored
            self.L_MEOH_1[0] = self.MEOH_tank_cap1
            self.tank_penalty += (self.L_MEOH_1[0] - self.MEOH_tank_cap1)
             
        MEOH_transport_initial = 0
        MEOH_transport = 0
        self.pred_demand_sum = 0
        MEOH_sell_2=0

        # Methanol transport logic every 12 steps
        if self.step_count % 12 == 0 and self.step_count != 0 : 
            self.pred_demand_sum = np.sum(self.pred_values[self.step_count])
            MEOH_required = self.SOFC_output(self.pred_demand_sum) 
            
            # Calculate maximum transportable amount
            desired_transport = self.L_MEOH_1[0] * transport_frac
            max_storable_area2 = self.MEOH_tank_cap2 - self.L_MEOH_2[0]
            
            MEOH_transport_initial = desired_transport
            MEOH_transport = min(desired_transport, max_storable_area2, self.L_MEOH_1[0])
            
            # Sell if transport amount exceeds area2 storage capacity
            if desired_transport > max_storable_area2:
                MEOH_sell_2 = desired_transport - max_storable_area2
            
            self.L_MEOH_1[0] -= MEOH_transport
            
            # Penalty if transport amount is less than required amount
            M_mis = MEOH_required - MEOH_transport
            if M_mis > 0:
                self.tank_penalty += M_mis 
            else:
                # Penalty for excessive transport
                excess_transport = -M_mis
                self.tank_penalty += excess_transport 
                M_mis = 0

            # Transport to area2
            self.L_MEOH_2[0] += MEOH_transport
            
            # Transport mismatch calculation logic 
            end_idx = self.step_count + 12
            if end_idx > len(self.demand_data):
                end_idx = len(self.demand_data)
        
            actual_demand_24h = np.sum(self.demand_data[self.step_count:end_idx])
            required_meoh_24h = self.SOFC_output(actual_demand_24h)
            
            # Calculate mismatch (actual required amount - actual transport amount)
            mismatch = required_meoh_24h - MEOH_transport
            
            # Store calculated mismatch value in list
            self.trans_error.append(mismatch)

        # Handle tank2 overcharge
        if self.L_MEOH_2[0] > self.MEOH_tank_cap2:
            MEOH_sell_2 += self.L_MEOH_2[0] - self.MEOH_tank_cap2
            self.L_MEOH_2[0] = self.MEOH_tank_cap2
            

        self.available_meoh = self.L_MEOH_2[0] * LMEOH2_util
        available_power = self.SOFC(self.available_meoh)
        M2_mis =  self.Building_demand[self.step_count] - available_power #kw
        
        if M2_mis > 0:   # Case where demand cannot be met
            self.MEOH_usage = self.available_meoh
            self.power_generated = available_power
            self.demand_penalty += M2_mis 
            self.L_MEOH_2[0] -= self.MEOH_usage
        else:
            required_meoh = self.SOFC_output(self.Building_demand[self.step_count]) 
            actual_meoh_used = min(required_meoh, self.available_meoh)
            
            self.MEOH_usage = actual_meoh_used
            self.power_generated = self.Building_demand[self.step_count]
            
            # Penalty for excessive methanol usage
            excess_usage = self.available_meoh - required_meoh
            if excess_usage > 0:
                self.tank_penalty2 += excess_usage 
            
            self.L_MEOH_2[0] -= self.MEOH_usage
            M2_mis = 0
        
        self.TBOM = 0 

        self.TAOM = 0 #10.11 * 0.012 * self.H + 0.0019 * 2.96 * self.H + 0.11 * 0.012 * self.H + 0.00029 * 0.33 * self.H

        if self.P_to_G[self.step_count]<0:
            self.grid_penalty += (P_mis-self.P_to_G[self.step_count])/self.scale
            self.carbon_emission = self.emission_factor * abs(self.P_to_G[self.step_count])
        else:
            self.carbon_emission = 0    
             
        self.carbon_capture = self.X_flow * self.X_CO2
        self.emission_diff = self.carbon_capture - self.carbon_emission
        
        #emission diff calucation   
        if self.emission_diff > 0:
            self.norm_emission_diff = 100
        else:
            self.norm_emission_diff = -100
   
        
        if self.step_count == len(self.renewable)-self.obs_length:
            self.X_profile[self.step_count] = X_load/self.P_X   
            self.H2_to_market[self.step_count] += H2_to_sell
            self.penalty = self.ESS_penalty*self.ESS_penalty_factor + self.PEM_penalty*self.PEM_penalty_factor    
            
            if self.reward_global: #global reward
                reward = (-self.penalty*self.penalty_weight + self.norm_emission_diff + (self.cost_calculation(ptx_CO2)-self.cost_factor[0])/(self.cost_factor[1]-self.cost_factor[0])*self.cost_weight
                      -(self.demand_penalty*self.demand_penalty_factor) - (self.tank_penalty*self.area1_tank_factor) - (self.tank_penalty2*self.area2_tank_factor))
                rewards = {"PTX_agent" :  reward, 
                            "Management_agent" : reward,} 
            
            else: #individual reward
                reward_1 = -(self.penalty*self.penalty_weight)+(self.cost_calculation(ptx_CO2) - self.cost_factor[0])/(self.cost_factor[1]-self.cost_factor[0])*self.cost_weight + (self.norm_emission_diff)
                reward_2 = -(self.demand_penalty*self.demand_penalty_factor) - (self.tank_penalty*self.area1_tank_factor) - (self.tank_penalty2*self.area2_tank_factor)
         
                rewards = {"PTX_agent" :  reward_1, 
                            "Management_agent" : reward_2,} 
            
            self.cost_list.append(self.cost_calculation(ptx_CO2))         #profit      
            self.step_count += 1
            self.P_consum_profile[self.step_count] = P_consum
            self.SOC_profile[self.step_count] = self.SOC[0]
            self.L_H2_profile[self.step_count] = self.L_H2[0] 
            self.MEOH_profile_1[self.step_count] = self.L_MEOH_1[0]
            self.MEOH_profile_2[self.step_count] = self.L_MEOH_2[0]
            self.SOFC_profile[self.step_count] = self.power_generated
            self.MEOH_transport_profile[self.step_count] = MEOH_transport  
            self.MEOH_to_market[self.step_count] += MEOH_sell
            self.MEOH_to_market2[self.step_count] += MEOH_sell_2
            self.MEOH_to_market_total[self.step_count] += (MEOH_sell + MEOH_sell_2)
            self.pred_demand_profile[self.step_count] = self.pred_demand_sum
            dones = {"PTX_agent" : True, "Management_agent" : True, "__all__" : True}
            truncateds = {"PTX_agent" : True, "Management_agent" : True, "__all__" : True}      
            
        else:
            self.step_count += 1
            self.X_profile[self.step_count] = X_load/self.P_X  
            self.H2_to_market[self.step_count] += H2_to_sell
            self.penalty = self.ESS_penalty*self.ESS_penalty_factor + self.PEM_penalty*self.PEM_penalty_factor  
            
            if self.reward_global: #global reward
                reward = (-self.penalty*self.penalty_weight + self.norm_emission_diff + (self.cost_calculation(ptx_CO2)-self.cost_factor[0])/(self.cost_factor[1]-self.cost_factor[0])*self.cost_weight
                      -(self.demand_penalty*self.demand_penalty_factor) - (self.tank_penalty*self.area1_tank_factor) - (self.tank_penalty2*self.area2_tank_factor))
                rewards = {"PTX_agent" :  reward, 
                            "Management_agent" : reward,} 
            
            else: #individual reward
                reward_1 = -(self.penalty*self.penalty_weight)+(self.cost_calculation(ptx_CO2) - self.cost_factor[0])/(self.cost_factor[1]-self.cost_factor[0])*self.cost_weight + self.norm_emission_diff
                reward_2 = -(self.demand_penalty*self.demand_penalty_factor) - (self.tank_penalty*self.area1_tank_factor) - (self.tank_penalty2*self.area2_tank_factor)
         
                rewards = {"PTX_agent" :  reward_1, 
                            "Management_agent" : reward_2,} 
         
            self.cost_list.append(self.cost_calculation(ptx_CO2))
             
            self.P_consum_profile[self.step_count] = P_consum             
            self.L_H2_profile[self.step_count] = self.L_H2[0]
            self.SOC_profile[self.step_count] = self.SOC[0]
            self.MEOH_profile_1[self.step_count] = self.L_MEOH_1[0]
            self.MEOH_profile_2[self.step_count] = self.L_MEOH_2[0]
            self.MEOH_to_market[self.step_count] += MEOH_sell
            self.MEOH_to_market2[self.step_count] += MEOH_sell_2
            self.MEOH_to_market_total[self.step_count] += (MEOH_sell + MEOH_sell_2)
            self.SOFC_profile[self.step_count] = self.power_generated
            self.MEOH_transport_profile[self.step_count] = MEOH_transport
            self.pred_demand_profile[self.step_count] = self.pred_demand_sum
       
            self.obs = self._obs() 
            dones = {"PTX_agent" : False, "Management_agent" : False, "__all__" : False}
            truncateds = {"PTX_agent" : False, "Management_agent" : False, "__all__" : False}

        infos = {}
        
        return self.obs, rewards, dones, truncateds, infos, 
             
    def _obs(self):
    
        current_idx = self.step_count + self.obs_length - 1
        pred_start = current_idx * 24
        pred_end = pred_start + 24
        

        if pred_end <= len(self.p_demand_data):
            pred_demand_normalized = np.array((self.p_demand_data[pred_start:pred_end])-self.pred_min_demand)/(self.pred_max_demand-self.pred_min_demand)
        else:
            pred_demand_normalized = np.array((self.p_demand_data[-24:])-self.pred_min_demand)/(self.pred_max_demand-self.pred_min_demand)
        
        # renewable, SMP, SOC (%), H2 storage level(%)
        self.PTX_agent_state = np.hstack([
            self.renewable[self.step_count:self.step_count+self.obs_length]/self.scale, 
            (np.array(self.SMP[self.step_count:self.step_count+self.obs_length])-self.min_SMP)/(self.max_SMP-self.min_SMP),
            np.array([self.SOC[0]/self.initial_capacity_new, self.L_H2[0]/self.H2_cap])
        ])
        # historical demand, predict demand, MEOH sotrage level(%) 1,2
        self.Management_agent_state = np.hstack([
           np.array((self.demand_data[self.step_count:self.step_count+self.obs_length])-self.min_demand)/(self.max_demand-self.min_demand),
           # np.array((self.p_demand_data[self.step_count*24:self.step_count*24+24])-self.pred_min_demand)/(self.pred_max_demand-self.pred_min_demand),
           pred_demand_normalized,
           np.array([self.L_MEOH_1[0]/self.MEOH_tank_cap1, self.L_MEOH_2[0]/self.MEOH_tank_cap2])
       ])  
        
        return {"PTX_agent": self.PTX_agent_state,
             "Management_agent": self.Management_agent_state,}

    
    def update_BESS_state(self):
        # Track daily hours
        self.current_hour = self.step_count % 24
        if self.current_hour == 0 and self.step_count > 0:  # When a day passes
            self.daily_cycles = 0  # Reset daily accumulated cycle count
    
        # Calculate aging - degradation
        aging_factor = self.cumulative_cycles / (2 * self.lifetime_cycle) # cumulative_cycles is updated by cycle_increment
        self.ESS_cap = self.initial_capacity * (1 - aging_factor)
        if self.ESS_cap < 0:
            self.ESS_cap = 0
    

    def ESS_masking(self, ESS_action):
        #initialize
        P_charge = 0
        P_discharge = 0
        
        # Calculate effective charge and discharge power
        if ESS_action > 0:  # Charging
            P_charge = ESS_action
            P_charge_eff = P_charge * self.ESS_eff
            P_discharge_eff = 0  # No simultaneous charge and discharge
        else:  # Discharging
            P_discharge = -ESS_action 
            P_charge_eff = 0
            P_discharge_eff = P_discharge / self.ESS_eff
        
        # Update SOC considering self-discharge
        self_discharge = self.SOC[0] * self.self_discharge_rate
        self.self_discharge_history.append(self_discharge) 
        net_SOC = self.SOC[0] - self_discharge + (P_charge_eff - P_discharge_eff) 
        #net_SOC = Current state of charge after accounting for self-discharge and action
        
        # SOC limits and apply penalties
        if net_SOC > self.ESS_cap:
            self.ESS_penalty += net_SOC - self.ESS_cap
            # Prevent overcharge: adjust action to actual chargeable amount
            ESS_action = (self.ESS_cap - (self.SOC[0] - self_discharge)) / self.ESS_eff
            net_SOC = self.ESS_cap
            # P_charge = ESS_action
            # P_discharge = 0

        elif net_SOC < 0:
            self.ESS_penalty += -net_SOC
            # Prevent overdischarge: adjust action to actual dischargeable amount
            ESS_action = -(self.SOC[0] - self_discharge) * self.ESS_eff
            net_SOC = 0
            # P_charge = 0
            # P_discharge = -ESS_action

        # Check if daily cycle limit is reached
        cycle_increment = (P_charge + P_discharge) / self.ESS_cap # Complete charge-discharge cycle (fraction of full cycle)
        if self.daily_cycles + cycle_increment <= self.daily_limit_cycle:          
            self.SOC[0] = net_SOC
            # Update cumulative cycles
            if P_charge > 0 or P_discharge > 0:
                self.cumulative_cycles += cycle_increment
                self.daily_cycles += cycle_increment  # Update daily accumulated cycle count
        else: # When daily limit cycle is exceeded
            ESS_action = 0 
            self.SOC[0] = self.SOC[0] - self_discharge
        
        return ESS_action
    
    def cost_calculation(self, ptx_CO2):
        #Cost estimation
        ii = 0.08 # interest rate
        N = 25 # plant life, years
        CRF =  ii * ((ii+1) ** N) / ((ii+1) ** N - 1)
        CAP_gen = self.CAP_solar * self.scale * (1-self.fw) + self.CAP_wind * self.scale * self.fw        
        OPEX_gen = self.OPEX_solar * self.scale * (1-self.fw) + self.OPEX_wind * self.scale * self.fw
        CAP_hydrogen = self.CAP_H2*self.H2_cap/1000
        CAP_electrolyzer = (self.PEM_P_cap)*self.CAP_PEM
        CAP_distillation  = self.distillation_cost()
        BESS_cos = self.ESS_cap*self.BESS_capacity_cost
        CAP_total = CAP_gen + CAP_hydrogen + CAP_electrolyzer + CAP_distillation + BESS_cos  
        
        C_ptx = ptx_CO2*(-0.02643151*np.log10(ptx_CO2/1000) + 1.01414484*self.C_CO2/1000 + 0.215257477808955)
        OPEX_total = OPEX_gen/8760 + C_ptx - self.P_to_G[self.step_count]*self.SMP[self.step_count+self.obs_length-1] -self.H2_to_market[self.step_count]*self.H2_price

        X_flow_total = self.X_profile[self.step_count]
             
        if (OPEX_total+CAP_total*CRF/8760)<0:
            production_cost = 0
        else:
            production_cost = (OPEX_total+CAP_total*CRF/8760)/(X_flow_total/1000)
            
        meoh_revenue = self.MEOH_to_market_total[self.step_count-1] * self.X_price
      
        profit = (self.P_to_G[self.step_count]*self.SMP[self.step_count+self.obs_length-1]+self.H2_to_market[self.step_count]*self.H2_price - self.TBOM - self.TAOM + meoh_revenue)

        return profit
           
    def cost_scale(self): 
    
        H_max = (self.PEM_P_cap)/self.SP_H2
        TAOM_max = 10.11 * 0.012 * H_max + 0.0019 * 2.96 * H_max + 0.11 * 0.012 * H_max + 0.00029 * 0.33 * H_max
        
        profit_min = -np.max(self.renewable)*max(self.SMP) #- self.ESS_P_cap*30 - TAOM_max

        profit_max = np.max(self.renewable-self.X_flow_P_cap)*max(self.SMP) + max((self.PEM_P_cap/self.SP_H2)*self.H2_price-TAOM_max,0)  
        
        return profit_min, profit_max
  
    def step(self, action_dict):
        return self._STEP(action_dict)

    def reset(self,seed=None, options=None):
        return self._RESET(seed=None, options=None)
    
    def wind_power_function(self, Wind_speed):
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

    def solar_power_function(self,Solar_irradiance):
        Ht = Solar_irradiance
        H_ref = 1000  # W/m2
        idx_cutoff = Ht > H_ref
        Ht[idx_cutoff] = H_ref
        n_tot = 0.9375
        return Ht / H_ref * n_tot  # Capacity fator =[0,1]
