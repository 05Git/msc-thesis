# Knowledge Fusion RL Policies for Fighting Games

Codebase for training an ensemble RL policy, _FuseNet_, to play fighting games as part of my dissertation on "knowledge fusion via adaptive expert policy weighting in fighting games".
1. Train expert policies through processes like reward shaping or imitation learning.
2. Use FuseNet to learn adaptive weights for each expert policy in each state.
   
This codebase requires [Diambra Arena](https://docs.diambra.ai/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/#) to run.

The main code for FuseNet is contained in ```FusionNet.py```, including the code for KL divergence regularisation.

The main training and evaluation loops are found in ```train.py``` and ```evaluate.py```, respectively.

Helper functions, callbacks and wrappers can be found in ```utils.py```, ```custom_callbacks.py``` and ```custom_wrappers.py```.

Code for imitation learning and random network distillation are in ```imitate.py``` and ```RND.py```.

The notebook ```plot_results.ipynb``` was used to collect, average out and plot results.
