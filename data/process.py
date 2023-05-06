### Process the YouTube graph

import os
import networkx as nx

G = nx.read_edgelist('../data/com-youtube.ungraph.txt', nodetype=int)
mapping = {n: i for i, n in enumerate(list(G.nodes()))}
G = nx.relabel_nodes(G, mapping)


### read into community info
with open('com-youtube.all.cmty.txt', 'r') as fid:
    comms = fid.readlines()


### relabel nodes w.r.t. 'mapping'
numComms = len(comms)
f = lambda x: mapping[int(x)]
for i in range(numComms):
    comms[i] = list(map(f, comms[i].rstrip().split('\t')))


### output processed graph & community information
output_folder = 'YouTube-processed/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

nx.write_edgelist(G, output_folder + 'youtube-processed-graph.txt', delimiter=' ', data=False)
with open(output_folder + 'youtube-processed-comm.txt', 'w') as fid:
    for com in comms:
        fid.write(' '.join(list(map(str, com))) + '\n')





