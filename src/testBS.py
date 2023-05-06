import numpy as np
import networkx as nx
from Games import Bestshot

# Create a random graph
n = 5
G = nx.gnp_random_graph(n, p=0.5)

# Instantiate the Bestshot game with random costs
#set seed
#b_vec = np.random.rand(n, seed=3)
#genrate b_vec with seed
b_vec = np.random.rand(n)
bestshot_game = Bestshot(n, G, b_vec)

# Set parameters for gradient descent
maxIter = 300
lr = 0.01

# Perform gradient descent
#remove np random seed

np.random.seed(2)
x_init = np.random.rand(n)  # Initial strategies
x, L = bestshot_game.grad_BR(maxIter=maxIter, lr=lr, x_init=x_init, full_update=True, elementwise=True, mode='sequential')

# Print results
print("Approximate Nash equilibrium:", x)
print("Regret over iterations:", L)
