import string

import networkx as nx

import random



def generate_random_dag(num_nodes, num_edges, fname):

    # Create an empty directed graph

    G = nx.DiGraph()



    # Add nodes

    G.add_nodes_from(range(num_nodes))



    # Add edges

    while G.number_of_edges() < num_edges:

        # Randomly select two different nodes

        node1, node2 = random.sample(range(num_nodes), 2)



        # Add an edge if it does not create a cycle

        if not nx.has_path(G, node2, node1):

            G.add_edge(node1, node2)



    # Create a mapping from numbers to letters

    mapping = {i: string.ascii_lowercase[i] for i in range(num_nodes)}



    # Relabel the nodes

    H = nx.relabel_nodes(G, mapping)



    nx.write_adjlist(H, fname)



    return H

generate_random_dag(10, 10, "_raw_bayesian_nets/syn_10_10.adjlist")
generate_random_dag(10, 20, "_raw_bayesian_nets/syn_10_20.adjlist")
generate_random_dag(10, 30,"_raw_bayesian_nets/syn_10_30.adjlist")
generate_random_dag(20, 30,"_raw_bayesian_nets/syn_20_30.adjlist")
generate_random_dag(20, 20, "_raw_bayesian_nets/syn_20_20.adjlist")
generate_random_dag(20, 40, "_raw_bayesian_nets/syn_20_40.adjlist")
generate_random_dag(26, 30,"_raw_bayesian_nets/syn_30_30.adjlist")
generate_random_dag(26, 40, "_raw_bayesian_nets/syn_30_40.adjlist")
generate_random_dag(26, 50, "_raw_bayesian_nets/syn_30_50.adjlist")



breakpoint()