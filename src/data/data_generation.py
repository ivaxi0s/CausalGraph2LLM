import bnlearn as bn
import networkx as nx
import numpy as np
import pandas as pd

import os
import pprint
import sys
from collections import deque
from io import StringIO

adjacency_matrix = """
"","Treatment","SproutN","BunchN","GrapeW","WoodW","SPAD06","NDVI06","SPAD08","NDVI08","Acid","Potass","Brix","pH","Anthoc","Polyph"
"Treatment",0,1,1,0,0,1,0,0,0,0,0,1,0,0,0
"SproutN",0,0,1,1,1,1,1,0,1,1,0,0,1,0,0
"BunchN",0,0,0,1,1,0,0,0,0,1,1,0,0,1,1
"GrapeW",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
"WoodW",0,0,0,1,0,0,0,0,0,0,0,0,1,1,0
"SPAD06",0,0,0,0,1,0,1,1,0,1,1,0,1,0,0
"NDVI06",0,0,0,1,0,0,0,1,1,1,0,0,0,0,1
"SPAD08",0,0,0,0,1,0,0,0,1,0,0,0,0,0,0
"NDVI08",0,0,0,1,1,0,0,0,0,1,0,0,0,1,1
"Acid",0,0,0,1,0,0,0,0,0,0,0,0,1,0,0
"Potass",0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
"Brix",0,0,0,1,0,0,0,0,0,1,0,0,1,0,1
"pH",0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
"Anthoc",0,0,0,1,0,0,0,0,0,1,1,1,1,0,1
"Polyph",0,0,0,0,0,0,0,0,0,1,0,0,1,0,0
"""

azadjacency_matrix = """
"","APOE4","Sex","Age","Education","AV45","Tau Levels","Brain Volume","Ventricular Volume","MOCA Score"
"APOE4",0,0,0,0,1,0,0,0,0
"Sex",0,0,0,0,0,0,1,1,0
"Age",0,0,0,0,1,1,1,1,1
"Education",0,0,0,0,0,0,0,0,1
"AV45",0,0,0,0,0,1,1,0,0
"Tau Levels",0,0,0,0,0,0,1,1,1
"Brain Volume",0,0,0,0,0,0,0,1,1
"Ventricular Volume",0,0,0,0,0,0,0,0,0
"MOCA Score",0,0,0,0,0,0,0,0,0
"""



from src.data.dag_utils import order_graph, find_children_parents, get_successor_at_each_level, get_predecessors_at_each_level

pp = pprint.PrettyPrinter(width=82, compact=True)

# Utility functions to mute printing done in BNLearn
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_dataset(bn_path, n=1000):
    
    if "syn" in bn_path:
        path=bn_path.replace('.bif', '.adjlist')
        G = nx.read_adjlist(path, create_using=nx.DiGraph())
        data = None        
    elif "sangiovese" in bn_path:
        df = pd.read_csv(StringIO(adjacency_matrix), index_col=0)

        # Convert the DataFrame into a networkx graph
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        data = None
    elif "alz" in bn_path:
        df = pd.read_csv(StringIO(azadjacency_matrix), index_col=0)

        # Convert the DataFrame into a networkx graph
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        data = None
    else:
        with HiddenPrints():
            model = bn.import_DAG(bn_path, verbose=1)
    
        G = nx.from_pandas_adjacency(model["adjmat"].astype(int), create_using=nx.DiGraph)
        # Sample data
        data = bn.sampling(model, n=n, verbose=1)
        G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), data.columns)))


    G = order_graph(G)
    return G, data

def generate_missing_node_data(bn_path, n=1000000, miss_nodes = None, replacement = None):
    if "sangiovese" in bn_path:
        df = pd.read_csv(StringIO(adjacency_matrix), index_col=0)

        # Convert the DataFrame into a networkx graph
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        data = None
    elif "alz" in bn_path:
        df = pd.read_csv(StringIO(azadjacency_matrix), index_col=0)

        # Convert the DataFrame into a networkx graph
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        data = None
    else:
        with HiddenPrints():
            model = bn.import_DAG(bn_path, verbose=1)
    
        G = nx.from_pandas_adjacency(model["adjmat"].astype(int), create_using=nx.DiGraph)
        # Sample data
        data = bn.sampling(model, n=n, verbose=1)
        G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), data.columns)))
    G = order_graph(G)

    # Label nodes in causal graph
    lst = []
    all_nodes = list(G.nodes())
    if miss_nodes is not None:  
        replacement_dict = dict(zip(miss_nodes, replacement))
        H = nx.relabel_nodes(G, replacement_dict)
        return G, H, data, miss_nodes
    else:
        miss_nodes = all_nodes
        for node in miss_nodes: 
            replacement = {node:'X'}
            H = nx.relabel_nodes(G, replacement)
            lst.append(H)
        return G, lst, data, miss_nodes

def generate_two_missing_node_level(bn_path, depth, n=1000):
    if "sangiovese" in bn_path:
        df = pd.read_csv(StringIO(adjacency_matrix), index_col=0)

        # Convert the DataFrame into a networkx graph
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        data = None
    elif "alz" in bn_path:
        df = pd.read_csv(StringIO(azadjacency_matrix), index_col=0)

        # Convert the DataFrame into a networkx graph
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        data = None
    else:
        with HiddenPrints():
            model = bn.import_DAG(bn_path, verbose=1)
    
        G = nx.from_pandas_adjacency(model["adjmat"].astype(int), create_using=nx.DiGraph)
        # Sample data
        data = bn.sampling(model, n=n, verbose=1)
        G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), data.columns)))
    G = order_graph(G)

    # Label nodes in causal graph
    lst = []
    all_nodes = list(G.nodes())
    miss_nodes_tuple = []

    children = get_successor_at_each_level(G, depth)
    parents = get_predecessors_at_each_level(G, depth)
    
    for node_x in all_nodes:        
    
        replacement = {node_x:'X'}
        H = nx.relabel_nodes(G, replacement)

        for node_y in all_nodes:
            if node_y in parents[node_x] or node_y in children[node_x]:
                continue
            else:
                replacement = {node_y:'Y'}
                K = nx.relabel_nodes(H, replacement)
            miss_nodes_tuple.append((node_x, node_y))
            lst.append(K)
    
    return G, lst, data, miss_nodes_tuple

