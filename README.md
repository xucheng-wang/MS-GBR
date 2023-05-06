# MS-GBR

## Working Environment Setup:
- Install ```numpy, scipy, pytorch, networkx, dill, cvxpy, matplotlib, scikit-learn```.
- recommand working in conda environemnt

## Files in src
### - utils.py: defines helper functions for constructing games, communities, and etc.
1. change file path based on your working env in gen_graph to access data
2. gen_b/gen_normalized_b helps to generate parameter vectors, where gen_normalized_b generates values from 0 to 1
### - Games.py: defines complete game classes for LQG and BSG. BHG need to be fixed.
1. BHG is not yet finished. (May consider remove BHG)
### - main_LQG.py: defines run simulation function for running LQG once.
### - main_BSG.py: defines run simulation function for running BSG once.
### - run_gamename.py: runs the game for certain times. You can manipulate parameters here. e.g. learning rate(lr), game mode, maxIter etc.
### - testBS.py: a test program for running BSG game on simulated small network.
### - visualize.ipynb: visualize the experiment result here. Change file path as needed


