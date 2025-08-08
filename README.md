# Reinforcement learning-based framework for integrated green methanol supply–demand management with demand forecasting under renewable energy uncertainty

This model is decribed in the paper :  Reinforcement learning-based framework for integrated green methanol supply–demand management with demand forecasting under renewable energy uncertainty

Due to file size limitations, the training dataset is available via Google Drive:

[Download Training Dataset](https://drive.google.com/file/d/1drQCl1fr5zulhzL5bED5PzgnMTpF6XRp/view?usp=sharing)

### Dataset Structure
```
data/
├── Germany/
│   ├── demand_data/
│   ├── Renewable_data/
│   └── SMP_data/
├── HV_demand_case3.pkl
├── HV_renew_case1.pkl
├── LV_demand_case4.pkl
└── LV_renew_case2.pkl
```
### Dataset Usage
1. Download the dataset from the Google Drive link above
2. Extract the `data/` folder to the project root directory
3. Ensure the following structure:
```
green-methanol-RL-framework/
├── data/
├── utils/
├── eval/
├── MARL_train.py
├── SARL_train.py
└── README.md
```

## Code Implementation Example

### Training Models
To train the model using PPO for each agent:

**Single-Agent RL (SARL)**:
```bash
python SARL_train.py
```

**Multi-Agent RL (MARL)**:
```bash
python MARL_train.py --reward-global
```

### Evaluate the result
After training, user can evaluate the result:

**SARL Evaluation:**
```bash
python eval/SARL_evaluation.py --checkpoint-path ./results/ray_results/{experiment_name}/{checkpoint_xxx} --test-case HV_renew_case1 --target-country Germany
```
**MARL Evaluation:**
```bash
python eval/MARL_evaluation.py --checkpoint-path ./results/ray_results/{experiment_name}/{checkpoint_xxx} --test-case HV_renew_case1 --target-country Germany
```
## Demand Forecasting

For demand forecasting, we use the Autoformer model. To train and predict with the demand forecasting model:
1. Navigate to the Autoformer directory
2. Run `predict_demand.py` to train the model on demand data and generate predictions
3. The script will automatically handle training, validation, and forecasting tasks

## Acknowledgments

This work incorporates code and methodologies from the following projects:

- **Autoformer**: Decomposition Transformers with Auto-Correlation for Long-term Series Forecasting  
  [![GitHub](https://img.shields.io/badge/GitHub-thuml/Autoformer-blue)](https://github.com/thuml/Autoformer)  
  *Used for demand forecasting implementation*



