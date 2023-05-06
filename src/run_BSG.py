import os
import csv
import argparse
import numpy as np
import pandas as pd
from main_BSG import run_simulation


parser = argparse.ArgumentParser()
parser.add_argument('--maxIter',       type=int,   default=200)# keep this 400
parser.add_argument('--lr',            type=float, default=0.006)# change this from 0.1 to 0.005
parser.add_argument('--n',             type=int,   default=100)
parser.add_argument('--graph',         type=str,   default='Email')
parser.add_argument('--c_mode',        type=str,   default='gaussian')
parser.add_argument('--b_var',         type=float, default=0.1)
parser.add_argument('--aggre',         type=str,   default='mean')
parser.add_argument('--mode',          type=str,   default='sequential')
parser.add_argument('--game',          type=str,   default='BSG')
args  = parser.parse_args()


N_RUNS = 100
SEEDS = list(range(21, 21+N_RUNS,1))
iii=1

results_x_ne_L = []
results_x_hat_L = []
print('g')

for seed in SEEDS:
    print('Run {} of {}'.format(iii, N_RUNS))
    x_ne_L, x_hat_L = run_simulation(args.maxIter, args.lr, args.n, args.graph, args.b_var, args.c_mode, seed, args.aggre, args.mode, args.game)
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
output_path = os.path.join("..", "EM-g-BSG3", "EM-g-BSG.csv")

output_directory = os.path.dirname(output_path)
os.makedirs(output_directory, exist_ok=True)

# Save the data to the CSV file
data.to_csv(output_path, index=False)
