from src.data.dag_utils import get_directed_edges
from causaldag import DAG
import itertools, random, time
import openai



def verbose_single_graph(true_G, codebook):
    cpdag = DAG.from_nx(true_G).cpdag()
    graph_edges = get_directed_edges(true_G)
    full_graph = []
    data_graph = []
    for edge in (graph_edges):
        x, y = edge
        x, y = x.lower(), y.lower()
        x_desc = codebook.loc[codebook['var_name'].str.lower()==x, 'var_description'].to_string(index=False)
        y_desc = codebook.loc[codebook['var_name'].str.lower()==y, 'var_description'].to_string(index=False)

        data_str = x + " causes " + y
        final_str = x_desc + " causes " + y_desc
        data_graph.append(data_str)
        full_graph.append(final_str)
    
    return full_graph, data_graph

def verbose_list_graphs(graph_list, orig_graph, codebook):
    main_g = []
    data_g = []
    missing_desc = []
    for i in range(len(orig_graph)):
        x_desc = codebook.loc[codebook['var_name'].str.lower()==orig_graph[i].lower(), 'var_description'].to_string(index=False)
        missing_desc.append(x_desc)

    for i in range(len(graph_list)):
        true_G = graph_list[i]
        cpdag = DAG.from_nx(true_G).cpdag()
        graph_edges = get_directed_edges(true_G)
        full_graph = []
        data_graph = []
        for edge in (graph_edges):
            x, y = edge
            x, y = x.lower(), y.lower()
            x_desc = codebook.loc[codebook['var_name'].str.lower()==x, 'var_description'].to_string(index=False)
            y_desc = codebook.loc[codebook['var_name'].str.lower()==y, 'var_description'].to_string(index=False)

            data_str = x + " causes " + y
            final_str = x_desc + " causes " + y_desc
            data_graph.append(data_str)
            full_graph.append(final_str)
        
        main_g.append(full_graph)
        data_g.append(data_graph)
    return data_g, main_g, missing_desc

def verbose_details(graph, codebook, detail_dict=None):
    nodes = [node for node in graph.nodes()]
    des_nodes = []
    for node in (nodes):
        x_desc = codebook.loc[codebook['var_name'].str.lower()==node.lower(), 'var_description'].to_string(index=False)
        des_nodes.append(x_desc)
    return des_nodes

def verb_child_parent(data ,codebook):
    return 0

def verbose_lookup_nodes(nodes, codebook, detail_dict=None):
    des_nodes = []
    for node in (nodes):
        x_desc = codebook.loc[codebook['var_name'].str.lower()==node.lower(), 'var_description'].to_string(index=False)
        des_nodes.append(x_desc)
    return des_nodes