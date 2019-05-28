# rogue-gym-agents-cog19
Model-free RL Agent implementations for rogue-gym.

# Setup
1. Install pipenv
```
pip3 install pipenv -U --user
```

2. (Optional) Modify Pipfile
For example, if you want to use PyTorch 1.0.0, specify `torch = '==1.0'` in Pipfile.

3. Create the virtual env
```
pipenv --site-package --three
pipenv install
```

# Usage

## Train agents
```
python eval_seeds.py --logdir=$YOUR_LOD_DIR
```


## Evaluate agents
```
python eval_seeds.py --logdir=$YOUR_LOD_DIR
```

