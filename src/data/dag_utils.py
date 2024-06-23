from causaldag import DAG
import networkx as nx
import numpy as np
from collections import deque

def get_mec(G_cpdag):

    mec = []

    for dag in G_cpdag.all_dags():
        g = [edge for edge in dag]
        mec.append(g)
    
    return mec

def get_undirected_edges(true_G, verbose=False):

    dag = DAG.from_nx(true_G)
    edges = dag.arcs - dag.cpdag().arcs

    if verbose:
        print("Unoriented edges: ", edges)
    
    return edges

def get_directed_edges(G, verbose=False):

    roots = [node for node, degree in G.in_degree() if degree == 0]

    # Initialize containers for edges by depth and visited nodes
    edges_by_depth = {}
    visited = set()

    def dfs(node, depth):
        visited.add(node)
        for successor in G.successors(node):
            if depth not in edges_by_depth:
                edges_by_depth[depth] = []
            edges_by_depth[depth].append((node, successor))
            if successor not in visited:
                dfs(successor, depth + 1)

    # Perform DFS from each root node
    for root in roots:
        dfs(root, 0)

    # Compute sum of edges at each depth
    sum_edges_by_depth = {depth: len(edges) for depth, edges in edges_by_depth.items()}
    total_edges = sum(sum_edges_by_depth.values())
    all_edges = [edge for edges in edges_by_depth.values() for edge in edges]
    # print("Total Edges:", total_edges)
    return all_edges

def get_decisions_from_mec(mec, undirected_edges):
    decisions = []
    
    for edge in undirected_edges:
        node_i = edge[0]
        node_j = edge[1]
        i_j = np.sum([((node_i, node_j) in dag) for dag in mec])
        j_i = np.sum([((node_j, node_i) in dag) for dag in mec])
        # if i_j and j_i we don't have to make a decision
        if not (i_j and j_i):
            if i_j:
                decisions.append((node_i, node_j))
            else:
                decisions.append((node_j, node_i))
                
    return decisions

def order_graph(graph):
    H = nx.DiGraph()
    #print(graph.nodes)
    H.add_nodes_from(sorted(graph.nodes(data=True)))
    H.add_edges_from(graph.edges(data=True))
    return H

def find_children_parents(graph): # get direct causes and cause of for each variable
    nodes = [node for node in graph.nodes()]
    children = []
    parents = []
    keys = nodes
    nested_dict = {}
    for node in nodes:
        desc = [dec.lower() for dec in  graph.successors(node)]
        pre = [pred.lower() for pred in graph.predecessors(node)]
        nested_dict[node] = {
            "parent": pre,
            "child": desc
        }    
    return nested_dict
        
def list_of_tuples_to_digraph(list_of_tuples):
    G = nx.DiGraph()
    # Add nodes best_graph
    for edge in list_of_tuples:
        node_i = edge[0]
        node_j = edge[1]
        G.add_edge(node_i, node_j)
    G = order_graph(G)
    return G

def is_dag_in_mec(G, mec):

    for dag in mec:
        ans = True
        for edge in dag:
            if edge not in G.edges:
                ans = False
                break
        if ans:
            return 1.
        
    return 0.

def get_successor_at_each_level(G, depth):
    def nodes_at_each_depth(G, root):
        # Initialize a dictionary to hold nodes at each depth
        depth_dict = {}

        # Perform BFS and get a dictionary of depths
        depth = nx.single_source_shortest_path_length(G, root)

        # Group nodes by depth
        for node, d in depth.items():
            if d not in depth_dict:
                depth_dict[d] = []
            depth_dict[d].append(node)
        return depth_dict

    result = {}
    for node in G.nodes():
        result[node] = nodes_at_each_depth(G, node)
    accumulated_lists = {}

    for node, attributes in result.items():
        accumulated_list = []
        for i in range(depth + 1):
            if i in attributes:
                accumulated_list.extend(attributes[i])
        accumulated_lists[node] = accumulated_list

    return accumulated_lists

def get_predecessors_at_each_level(G, depth):
    predecessors = {}
    for node in G.nodes():
        predecessors[node] = {0: [node]}
        for i in range(1, len(G.nodes())):
            predecessors[node][i] = []
            for pred in predecessors[node][i-1]:
                predecessors[node][i].extend(list(G.predecessors(pred)))
            if not predecessors[node][i]:
                break
    accumulated_lists={}
    for node, attributes in predecessors.items():
        accumulated_list = []
        for i in range(depth + 1):
            if i in attributes:
                accumulated_list.extend(attributes[i])
        accumulated_lists[node] = accumulated_list

    return accumulated_lists

def replace_nodes(G, miss_nodes, replacement):
    replacement_dict = dict(zip(miss_nodes, replacement))
    H = nx.relabel_nodes(G, replacement_dict)
    return G, H, miss_nodes
