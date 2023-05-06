import os
import argparse
import numpy as np
import networkx as nx
import dill as pickle
# import matplotlib.pyplot as plt

from Games import LQGame, BHGame, Bestshot
from utils import gen_graph, extract_community, gen_group_graph, gen_b, gen_beta, random_community

import warnings
warnings.filterwarnings("ignore")


def run_simulation(maxIter, lr, n, graph, b_mode, beta_mode, b_var, beta_var, control_var, seed, elem, random_comm, nComm, aggre, mode, output, traj, game):
    ...
    # Replace "args.seed" with "seed"
    ...
    candidate_games = {

        'LQG': LQGame,
        'BHG': BHGame,
        'BSG': Bestshot
    }
    
    print(game)
    game_instance = candidate_games[game]

    np.random.seed(seed)
    UB = 1.0
    LB = 0.0
    Aggre_ = np.mean if aggre == 'mean' else np.median

    G = gen_graph(n=n, graph=graph, seed=seed)
    n = len(G)
    print("Start get communities")
    ### get communities
    comms = extract_community(G, graph, nComm=nComm)
    comms = random_community(G, comms) if random_comm else comms
    nG    = len(comms)

    ### group-level graph
    GG = gen_group_graph(G, comms)


    print("Start constructing games.")
    ### define individual-level games
    b_vec     = gen_b(G, var=b_var, comms=comms, mode=b_mode)
    beta_vec  = gen_beta(G, var=beta_var, comms=comms, mode=beta_mode, control_var=control_var)
    indivGame = game_instance(n, G, b_vec, beta_vec, ub=UB, lb=LB)

    ### define group-level games
    b_vec_g    = np.array([Aggre_(b_vec[comms[k]]) for k in range(len(comms))])
    beta_vec_g = np.array([Aggre_(beta_vec[comms[k]]) for k in range(len(comms))])
    groupGame  = game_instance(nG, GG, b_vec_g, beta_vec_g, ub=UB, lb=LB)
    ### starting point
    x_start = np.random.rand(n)

    ### compute NE
    print("Computing Individual-level NE...")
    x_ne, x_ne_L = indivGame.grad_BR(maxIter=maxIter, lr=lr, mode=mode, x_init=x_start)
    print("Computing Group-level NE...")
    y_ne, y_ne_L = groupGame.grad_BR(maxIter=maxIter, lr=lr, mode=mode)
    

    ### compute x_hat and y_hat
    x_hat = np.zeros(x_ne.shape)
    y_hat = np.zeros(y_ne.shape)
    for k, com in comms.items():
        x_hat[com] = y_ne[k]
        y_hat[k]   = Aggre_(x_ne[com])
    ### generate a random individual-level profile
    x_random   = np.random.rand(n)

    ### BR with x_hat as starting point
    print("Computing BR with x_hat as starting point...")
    x_group_ne, x_hat_L = indivGame.grad_BR(maxIter=maxIter, lr=lr, x_init=x_hat, mode=mode)

    if traj:
        for i in range(len(x_hat_L)):
            print(f"{x_ne_L[i]},{x_hat_L[i]}")
        print('--------')

    return x_ne_L, x_hat_L


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxIter',       type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=0.005)
    parser.add_argument('--n',             type=int,   default=5)
    parser.add_argument('--graph',         type=str,   default='Facebook')
    parser.add_argument('--b_mode',        type=str,   default='uniform')
    parser.add_argument('--beta_mode',     type=str,   default='fully_homophily')
    parser.add_argument('--b_var',         type=float, default=0.1)
    parser.add_argument('--beta_var',      type=float, default=0.1)
    parser.add_argument('--control_var',   type=float, default=0.001)
    parser.add_argument('--seed',          type=int,   default=0)
    parser.add_argument('--elem',          type=int,   default=1)
    parser.add_argument('--random_comm',   type=int,   default=0)
    parser.add_argument('--nComm',         type=int,   default=500)
    parser.add_argument('--aggre',         type=str,   default='mean')
    parser.add_argument('--mode',          type=str,   default='simultaneous')
    parser.add_argument('--output',        type=int,   default=0)
    parser.add_argument('--traj',          type=int,   default=1)
    parser.add_argument('--game',          type=str,   default='LQG')
    args  = parser.parse_args()

    run_simulation(args.maxIter, args.lr, args.n, args.graph, args.b_mode, args.beta_mode, args.b_var, args.beta_var, args.control_var, args.seed, args.elem, args.random_comm, args.nComm, args.aggre, args.mode, args.output, args.traj, args.game)

