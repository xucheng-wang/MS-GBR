import os
import argparse
import numpy as np
import networkx as nx
import dill as pickle
# import matplotlib.pyplot as plt

from Games import LQGame, BHGame, Bestshot
from utils import gen_graph, extract_community, gen_group_graph, gen_b, gen_beta, random_community, gen_normalized_b

import warnings
warnings.filterwarnings("ignore")


def run_simulation(maxIter, lr, n, graph, b_var, c_mode, seed, aggre, mode, game):
    ...
    ...
    candidate_games = {
        'BSG': Bestshot
    }
    # print(game)
    game_instance = candidate_games[game]

    np.random.seed(seed)
    Aggre_ = np.mean if aggre == 'mean' else np.median

    ### generate individual-level graph
    G = gen_graph(n=n, graph=graph, seed=seed)
    n = len(G)

    ### get communities
    print("Extracting communities...")
    comms = extract_community(G, graph)
    #end timer
    nG    = len(comms)

    ### group-level graph
    GG = gen_group_graph(G, comms)

    # print(len(G))
    # print(len(GG))

    ### define individual-level games
    b_vec     = gen_normalized_b(G, var=b_var, comms=comms, mode=c_mode)
    indivGame = game_instance(n, G, b_vec)

    ### define group-level games
    b_vec_g    = np.array([Aggre_(b_vec[comms[k]]) for k in range(len(comms))])
    groupGame  = game_instance(nG, GG, b_vec_g)

    ### generate a random individual-level profile
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

    ### GBR with x_hat as starting point
    print("Computing NE with x_hat as starting point...")
    x_group_ne, x_hat_L = indivGame.grad_BR(maxIter=maxIter, lr=lr, x_init=x_hat, mode=mode)

    for i in range(len(x_hat_L)):
        print(f"{x_ne_L[i]},{x_hat_L[i]}")
    print('--------')

    return x_ne_L, x_hat_L

