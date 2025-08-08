'''
@author: Hyun jun Choi
Usage of:
    - a custom PTV environment
    - Ray Tune for grid search to try different learning rate and kl coefficient
Run code with defaults:
    $ python SARL_train.py
'''
# %% Import libraries
import argparse
import os
import random
import csv
from utils.env.SARL_env import *
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import get_trainable_cls
from ray import tune, air, train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms import ppo


torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.") #
parser.add_argument("--stop-iters", type=int, default=1000, help="Number of iterations to train.") 
parser.add_argument("--num-samples", type=int, default=10, help="Number of samples in populations.")
parser.add_argument("--obs-length", type=int, default=24, help="Observation length of profile data",)
parser.add_argument("--X-flow", type=float, default=10000/(0.65702 + 0.19576*55.7))
parser.add_argument("--H2-storage-period", type=int, default=2.65)
parser.add_argument("--ESS-storage-period", type=int, default=87.97)
parser.add_argument("--MEOH-storage-period", type=int, default=17.595)
parser.add_argument("--op-period", type=int, default=30*24)
parser.add_argument("--target-country", choices=['South_Korea', 'France', 'Germany'], default='Germany', help="Target country for simulation.")
parser.add_argument("--save-dir", type=str, default="./results", help="Output directory for result files.")


def register_env(env_name, env_config={}):
    tune.register_env(env_name, lambda env_config: PTX_env_RL(config=env_config))
    
def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

# %% 
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Create results directory
    results_dir = args.save_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Register env
    env_name = 'ptx_env_rl'
    env_config = {}
    env_config['X_flow'] = args.X_flow
    env_config['obs_length'] = args.obs_length
    env_config['H2_storage_period'] = args.H2_storage_period
    env_config['ESS_storage_period'] = args.ESS_storage_period
    env_config['MEOH_storage_period'] = args.MEOH_storage_period
    env_config['op_period'] = args.op_period
    env_config['target_country'] = args.target_country
    
    register_env(env_name, env_config)
    
     
    hyperparam_mutations = {
        "lambda": lambda: random.uniform(0.5, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": lambda: random.uniform(1e-6, 1e-4),
        "train_batch_size": lambda: random.randint(1000, 8000), 
        "num_sgd_iter": lambda: random.randint(1, 25), 
        "sgd_minibatch_size": lambda: random.randint(100, 1000),
        "kl_coeff":  lambda: random.uniform(0.01, 0.5),
        "entropy_coeff": lambda: random.uniform(0, 0.1),
    }
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        resample_probability=0.25,
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )
    
    
    stopping_criteria = {"training_iteration": args.stop_iters}
   
    # automated run with Tune and population based training
    tuner = tune.Tuner(
        args.run,
        tune_config=tune.TuneConfig(
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
                fcnet_activation="tanh") ,
            # These params are tuned from a fixed starting value.
            "clip_param": 0.2,
            "lr": 1e-4,
            # These params start off randomly drawn from a set.
            "train_batch_size": tune.choice([1000, 2000, 4000]),
            "gamma":0.99, #discount factor"
            "entropy_coeff": 0.01
        },
        run_config=train.RunConfig(
          stop=stopping_criteria, 
          name='SARL_{target_country}'.format(target_country=args.target_country),
          storage_path=os.path.abspath(os.path.join(results_dir, 'ray_results'))
      ), 
  )

    results = tuner.fit()
    best_result = results.get_best_result()
    
    # Save best trial results
    csv_path = os.path.join(results_dir, 'sarl_best_trial_results.csv')
    with open(csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        best_check_point = {}
        best_check_point['check_point'] = best_result.checkpoint.path    
        csv_writer.writerow([best_check_point])
