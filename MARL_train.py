'''
Usage of:
    - a custom PTV environment
    - Ray Tune for grid search to try different learning rate and kl coefficient
Run code with defaults:
    $ python MARL_train.py --reward-global
RL training with hyperparameter tuning
'''
# %% Import libraries
import argparse
import os
import random
import csv
from utils.env.MARL_env import *
from ray.tune.registry import register_env #custom env
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import get_trainable_cls
from ray import tune, air, train
from ray.tune.schedulers import PopulationBasedTraining #
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPOTF1Policy, PPOTF2Policy
from ray.rllib.algorithms import ppo

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()

parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
parser.add_argument("--stop-iters", type=int, default=1000, help="Number of iterations to train.")
parser.add_argument("--num-samples", type=int, default=10, help="Number of samples in populations.")
parser.add_argument("--obs-length", type=int, default=24, help="Observation length of profile data.")
parser.add_argument("--reward-global", action="store_true", help="Set agent reward global reward or individual reward.")
parser.add_argument("--X-flow", type=float, default=10000/(0.65702 + 0.19576*55.7), help="env config: Meoh flow rate.")
parser.add_argument("--H2-storage-period", type=float, default=2.65, help="env config: H2 storage capacity.")
parser.add_argument("--ESS-storage-period", type=float, default=87.97, help="env config: ESS storage capacity.")
parser.add_argument("--MEOH-storage-period", type=float, default=17.595, help="env config: MEOH storage capacity.")
parser.add_argument("--op-period", type=int, default=30*24, help="env config: grid penalty.")
parser.add_argument("--target-country", choices=['South_Korea', 'France', 'Germany'], default='Germany',help="Target country for simulation.")
parser.add_argument("--save-dir", type=str, default="./results", help="Output directory for result files.")


def register_env(env_name, env_config={}):
    tune.register_env(env_name, lambda env_config: PTX_env(config=env_config))
    
def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

def select_policy(algorithm, framework):
    if algorithm == 'PPO':
        if framework == 'torch':
            return PPOTorchPolicy
        elif framework == 'tf' :
            return PPOTF1Policy
        else:
            return PPOTF2Policy
    else:
        raise ValueError("Unknown algorithm :", algorithm)
        
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "PTX_agent":
        return "PTX_policy"
    else:
        return "Manage_policy"


#%%
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Create results directory
    results_dir = args.save_dir
    os.makedirs(results_dir, exist_ok=True)

    obs_length = args.obs_length 

    space_low = np.zeros(shape=(obs_length * 2 + 2), dtype=np.float32) # 24+24+2
    space_high = np.zeros(shape = (obs_length*2 + 2),dtype = np.float32) # 24+24+2
    space_high[:] = 1
    observation_space = Dict({
    "PTX_agent": Box(low=space_low, high=space_high, dtype=np.float32), #50
    "Management_agent": Box(low=space_low, high=space_high, dtype=np.float32),}) #50
    
    PTX_agent_action_space_low = np.array([-1, 0, 0, 0], dtype=np.float32)  
    PTX_agent_action_space_high = np.array([1, 1, 1, 1], dtype=np.float32)
    Management_agent_action_space_low = np.array([0, 0], dtype=np.float32)
    Management_agent_action_space_high = np.array([1, 1], dtype=np.float32)

    action_space = Dict({
        "PTX_agent": Box(low=PTX_agent_action_space_low, high=PTX_agent_action_space_high, shape=(4,), dtype=np.float32),
        "Management_agent": Box(low=Management_agent_action_space_low, high=Management_agent_action_space_high, shape=(2,), dtype=np.float32),
    })           
    # policies
    policies = {
        "PTX_policy": (
            select_policy("PPO", args.framework),
            observation_space["PTX_agent"],
            action_space["PTX_agent"],
            {},
        ),
        "Manage_policy": (
            select_policy("PPO", args.framework),
            observation_space["Management_agent"],
            action_space["Management_agent"],
            {},
        ),
    }
    

    # Register env
    env_name = 'ptx_env'
    env_config = {}
    env_config['X_flow'] = args.X_flow
    env_config['obs_length'] = args.obs_length
    env_config['H2_storage_period'] = args.H2_storage_period
    env_config['ESS_storage_period'] = args.ESS_storage_period
    env_config['MEOH_storage_period'] = args.MEOH_storage_period
    env_config['reward_global'] = args.reward_global
    env_config['op_period'] = args.op_period
    env_config['target_country'] = args.target_country
    
    register_env(env_name, env_config)
    
    # hyperparameter tuning range
    hyperparam_mutations = {
        "lambda": lambda: random.uniform(0.5, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": lambda: random.uniform(1e-6, 1e-4),
        "train_batch_size": lambda: random.randint(1000, 8000),  
        "num_sgd_iter": lambda: random.randint(1, 25), #30->25
        "sgd_minibatch_size": lambda: random.randint(100, 1000),
        "kl_coeff":  lambda: random.uniform(0.01, 0.5),
        "entropy_coeff": lambda: random.uniform(0, 0.1),
    }
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
        log_config=False 
    )
    
    stopping_criteria = {"training_iteration": args.stop_iters}
    
    # automated run with Tune and population based training
    tuner = tune.Tuner(
        args.run,
        tune_config=tune.TuneConfig( ## PPOConfig → TorchPolicyV2 → ModelCatalog
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples= args.num_samples,
        ), 
        param_space={
            "env": env_name,
            "env_config": env_config,
            "disable_env_checking": True,
            "kl_coeff": 0.5,
            "num_workers": 8,
            "num_cpus": 1,  # number of CPUs to use per trial
            "num_gpus": 0,  # number of GPUs to use per trial
            "model":dict(
                fcnet_hiddens=[512, 256, 256, 256,256],
                fcnet_activation="tanh",) , #vf_share_layers=False  :Policy and value networks use separate layers, default: False
            # These params are tuned from a fixed starting value.
            "clip_param": 0.2,
            "lr": 1e-4,
            # These params start off randomly drawn from a set.
            "train_batch_size": tune.choice([1000, 2000, 4000]),
            "entropy_coeff": 0.01,
            #multi-agent 
            "multiagent" : dict(
                policies = policies,
                policy_mapping_fn = policy_mapping_fn,
                policies_to_train = ["PTX_policy", "Manage_policy"],),
        },
        run_config=train.RunConfig(stop=stopping_criteria, name = 'MARL_{target_country}_rg_{reward_global}_'.format(target_country = args.target_country,
                                                                                                            reward_global = args.reward_global),
                                                                  storage_path=os.path.abspath(os.path.join(results_dir, 'ray_results')))
        )
    
    results = tuner.fit()
    best_result = results.get_best_result()
    
    # Save best trial results
    csv_path = os.path.join(results_dir, 'best_trial_results.csv')
    with open(csv_path, 'a', newline='') as csv_file:
       csv_writer = csv.writer(csv_file)
       best_check_point = {}
       best_check_point['check_point'] = best_result.checkpoint.path    
       csv_writer.writerow([best_check_point])
