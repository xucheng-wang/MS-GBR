# MS-GBR

## Working Environment Setup:
- Install ```numpy, scipy, pytorch, networkx, dill, cvxpy, matplotlib, scikit-learn```.
- recommand working in conda environemnt
- The data needed to run the simulation is [here](https://www.dropbox.com/s/uss99d5xwzgapg5/data.zip?dl=0).

## Files in src
- utils.py: defines helper functions for constructing games, communities, and etc.
  - change file path based on your working env in gen_graph to access data
  - gen_b/gen_normalized_b helps to generate parameter vectors, where gen_normalized_b generates values from 0 to 1
- Games.py: defines complete game classes for LQG and BSG. BHG is not yet finished. (May consider remove BHG)
- main_LQG.py: defines run_simulation function for running LQG once.
  - returns a tuple: (regret_GBR, regret_MS_GBR)
- main_BSG.py: defines run_simulation function for running BSG once.
  - returns a tuple: (regret_GBR, regret_MS_GBR)
- run_gamename.py: runs the game for 100 times. You can manipulate parameters here. e.g. learning rate(lr), game mode, maxIter etc.
  - the results are saved as a csv file with 6 columns. 3 for recording the regrets for GBR(25/50/75 percentiles), other 3 for recording the regrets for MS-GBR.
- testBS.py: a test program for running BSG game on simulated small network.
- visualize.ipynb: visualize the experiment result here. Change file path as needed


