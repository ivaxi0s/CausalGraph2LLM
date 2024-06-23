from src.data.dag_utils import get_directed_edges, find_children_parents
from causaldag import DAG
import itertools, random, time
import openai
import json
import networkx as nx


def make_graph_verbose(graph, codebook):
    # Create a mapping for node relabeling
    relabel_mapping = {}

    for node in graph.nodes():
        original_node = node
        node = node.lower()

        # Look up variable descriptions from the codebook
        desc = codebook.loc[codebook['var_name'].str.lower() == node, 'var_description'].to_string(index=False)

        # Create verbose label for the node
        verbose_label = f"{desc}"

        # Add the mapping for relabeling
        relabel_mapping[original_node] = verbose_label

    # Update the graph by replacing the original node names with verbose labels
    verbose_graph = nx.relabel_nodes(graph, relabel_mapping)
    return verbose_graph



def verbose_single_graph(true_G, codebook, reverse = "normal"):
    cpdag = DAG.from_nx(true_G).cpdag()
    graph_edges = get_directed_edges(true_G)
    full_graph = []
    data_graph = []
    for edge in (graph_edges):
        x, y = edge
        x, y = x.lower(), y.lower()
        x_desc = codebook.loc[codebook['var_name'].str.lower()==x, 'var_description'].to_string(index=False)
        y_desc = codebook.loc[codebook['var_name'].str.lower()==y, 'var_description'].to_string(index=False)

        data_str = " < " + x + " > "+ "causes" + " < " + y + " > "
        final_str = " < " + x_desc + " > "+ "causes" + " < " + y_desc + " > "
        # data_str = data_str.replace("< ", "\"").replace(" >", "\"")
        # final_str = final_str.replace("< ", "\"").replace(" >", "\"")
        data_graph.append(data_str)
        full_graph.append(final_str)
    if reverse == "reverse": return full_graph[::-1], data_graph[::-1]
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
            # breakpoint()
            x, y = edge
            x, y = x.lower(), y.lower()
            x_desc = codebook.loc[codebook['var_name'].str.lower()==x, 'var_description'].to_string(index=False)
            y_desc = codebook.loc[codebook['var_name'].str.lower()==y, 'var_description'].to_string(index=False)

            data_str = " < " + x + " > "+ "causes" + " < " + y + " > "
            final_str = " < " + x_desc + " > "+ "causes" + " < " + y_desc + " > "
            # data_str = data_str.replace("< ", "\"").replace(" >", "\"")
            # final_str = final_str.replace("< ", "\"").replace(" >", "\"")

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

def verb_child_parent(data ,codebook, node_type):
    for key, values in data.items():
        parent_nodes = values.get(node_type, [])
        p = []
        for node in parent_nodes:
            p.append(codebook.loc[codebook['var_name'].str.lower()==node.lower(), 'var_description'].to_string(index=False).lower())
        values["parent"] = p
    return data

def verbose_double_list_graphs(graph_list, node_tuples, codebook):
    main_g = []
    data_g = []
    missing_desc = []
    for i in range(len(node_tuples)):
        x_desc = codebook.loc[codebook['var_name'].str.lower()==node_tuples[i][0].lower(), 'var_description'].to_string(index=False)        
        y_desc = codebook.loc[codebook['var_name'].str.lower()==node_tuples[i][1].lower(), 'var_description'].to_string(index=False)
        missing_desc.append((x_desc, y_desc))

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

            data_str = " < " + x + " > "+ "causes" + " < " + y + " > "
            final_str = " < " + x_desc + " > "+ "causes" + " < " + y_desc + " > "
            # data_str = data_str.replace("< ", "\"").replace(" >", "\"")
            # final_str = final_str.replace("< ", "\"").replace(" >", "\"")

            data_graph.append(data_str)
            full_graph.append(final_str)
        
        main_g.append(full_graph)
        data_g.append(data_graph)
    return data_g, main_g, missing_desc

'''
get two nodes missing
'''

def generate_distractors(nodes):
    prompt_template = "You will be given a phrase. Can you please suggest 3 phrases that dont mean the same thing but are fairly ok close. Return your answer in the format <Answer> Phrase1, Phrase2, Phrase3 <Answer>. Word : "
    prompt_list = []
    for node in nodes:
        prompt_list.append(prompt_template + str(node))
    return prompt_list