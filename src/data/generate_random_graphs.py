import string
import networkx as nx
import random
import sys
import csv
import os
import argparse

def generate_random_dag(num_nodes, num_edges, adjlist_fname):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    while G.number_of_edges() < num_edges:
        node1, node2 = random.sample(range(num_nodes), 2)
        if not nx.has_path(G, node2, node1):
            G.add_edge(node1, node2)
    mapping = {}
    letters = string.ascii_uppercase  # 'A' to 'Z'
    for i in range(num_nodes):
        if i < 26:
            mapping[i] = letters[i]
        else:
            mapping[i] = letters[(i // 26) - 1] + letters[i % 26]
    H = nx.relabel_nodes(G, mapping)
    nx.write_adjlist(H, adjlist_fname)

    return H

def generate_codebook_csv(num_nodes, codebook_fname):
    with open(codebook_fname, 'w', newline='') as csvfile:
        fieldnames = ['node', 'var_name', 'var_description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(1, num_nodes + 1):
            var_name = chr(ord('a') + (i - 1) % 26)
            var_description = i - 1
            writer.writerow({'node': i, 'var_name': var_name, 'var_description': var_description})

def generate_random_dag_and_codebook(num_nodes=10, num_edges=10, save_folder='output'):
    # Generate filenames
    adjlist_fname = os.path.join(save_folder, f"syn_{num_nodes}_{num_edges}.adjlist")
    codebook_fname = os.path.join(save_folder, f"syn_{num_nodes}_{num_edges}_codebook.csv")
    H = generate_random_dag(num_nodes, num_edges, adjlist_fname)
    generate_codebook_csv(num_nodes, codebook_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random Directed Acyclic Graphs (DAGs) and their codebook CSV files.")
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes in the DAG (default: 10)')
    parser.add_argument('--num_edges', type=int, default=10, help='Number of edges in the DAG (default: 10)')
    parser.add_argument('--graph_folder', type=str, default='_raw_bayesian_nets', help='Folder to save output files for graphs')
    parser.add_argument('--codebook_folder', type=str, default='codebook', help='Folder to save the codebook files')

    args = parser.parse_args()

    generate_random_dag_and_codebook(args.num_nodes, args.num_edges, args.save_folder)
