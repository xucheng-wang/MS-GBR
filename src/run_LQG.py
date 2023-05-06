import os
import csv
import argparse
import numpy as np
import pandas as pd
from main_code import run_simulation

parser = argparse.ArgumentParser()
parser.add_argument('--maxIter',       type=int,   default=100)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--n',             type=int,   default=100)
parser.add_argument('--graph',         type=str,   default='Facebook')
parser.add_argument('--b_mode',        type=str,   default='uniform')
parser.add_argument('--beta_mode',     type=str,   default='gaussian')
parser.add_argument('--b_var',         type=float, default=0.1)
parser.add_argument('--beta_var',      type=float, default=0.1)
parser.add_argument('--control_var',   type=float, default=0.001)
#parser.add_argument('--seed',          type=int,   default=0)
parser.add_argument('--elem',          type=int,   default=1)
parser.add_argument('--random_comm',   type=int,   default=0)
parser.add_argument('--nComm',         type=int,   default=500)
parser.add_argument('--aggre',         type=str,   default='mean')
parser.add_argument('--mode',          type=str,   default='sequential')
parser.add_argument('--output',        type=int,   default=0)
parser.add_argument('--traj',          type=int,   default=1)
parser.add_argument('--game',          type=str,   default='LQG')

args  = parser.parse_args()


N_RUNS = 100
SEEDS = list(range(N_RUNS))
iii=1

results_x_ne_L = []
results_x_hat_L = []

for seed in SEEDS:
    print('Run {} of {}'.format(iii, N_RUNS))
    x_ne_L, x_hat_L = run_simulation(args.maxIter, args.lr, args.n, args.graph, args.b_mode, args.beta_mode, args.b_var, args.beta_var, args.control_var, seed, args.elem, args.random_comm, args.nComm, args.aggre, args.mode, args.output, args.traj, args.game)
    results_x_ne_L.append(x_ne_L)
    results_x_hat_L.append(x_hat_L)
    iii+=1

# Compute the mean, lower quartile, and upper quartile for each index
mean_x_ne_L = np.median(results_x_ne_L, axis=0)
lower_quartile_x_ne_L = np.percentile(results_x_ne_L, 25, axis=0)
upper_quartile_x_ne_L = np.percentile(results_x_ne_L, 75, axis=0)

mean_x_hat_L = np.median(results_x_hat_L, axis=0)
lower_quartile_x_hat_L = np.percentile(results_x_hat_L, 25, axis=0)
upper_quartile_x_hat_L = np.percentile(results_x_hat_L, 75, axis=0)

# Save the results to a CSV file
data = pd.DataFrame({
    'mean_x_ne_L': mean_x_ne_L,
    'lower_quartile_x_ne_L': lower_quartile_x_ne_L,
    'upper_quartile_x_ne_L': upper_quartile_x_ne_L,
    'mean_x_hat_L': mean_x_hat_L,
    'lower_quartile_x_hat_L': lower_quartile_x_hat_L,
    'upper_quartile_x_hat_L': upper_quartile_x_hat_L
})

# Define the output path
output_path = os.path.join("..", "FB-g-LQG-se", "FB-g-LQG-se.csv")

output_directory = os.path.dirname(output_path)
os.makedirs(output_directory, exist_ok=True)

# Save the data to the CSV file
data.to_csv(output_path, index=False)
