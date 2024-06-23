import os, json, re
import itertools
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semantic_sim_lst(lst, model_name='sentence-transformers/all-mpnet-base-v2', th = 0.5):
    similarities = np.zeros((len(lst), len(lst)))
    
    # Calculate pairwise similarities
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            similarities[i, j] = semantic_similarity(lst[i], lst[j], model_name)
            similarities[j, i] = similarities[i, j]

    # Use hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=th)  # Adjust distance_threshold
    clusters = clustering.fit_predict(similarities)

    # Organize words into clusters
    word_clusters = {}
    for word, cluster_id in zip(lst, clusters):
        if cluster_id not in word_clusters:
            word_clusters[cluster_id] = []
        word_clusters[cluster_id].append(word)

    return list(word_clusters.values())

def semantic_similarity(sentence1, sentence2, model_name='sentence-transformers/all-mpnet-base-v2'):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Tokenize sentences
    encoded_input1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt')
    encoded_input2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output1 = model(**encoded_input1)
        model_output2 = model(**encoded_input2)

    # Perform pooling to get sentence embeddings
    sentence_embedding1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
    sentence_embedding2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

    # Normalize embeddings
    sentence_embedding1 = F.normalize(sentence_embedding1, p=2, dim=1)
    sentence_embedding2 = F.normalize(sentence_embedding2, p=2, dim=1)

    # Compute similarity scores of two embeddings
    similarity_score = cosine_similarity(
        sentence_embedding1.detach().numpy(),
        sentence_embedding2.detach().numpy()
    )

    return similarity_score[0][0]

def check_dir(fold):
    if not os.path.exists(fold):
    # If the directory does not exist, create it
        os.makedirs(fold)

def extract_answers_tags(json_file_path, tag = "answer_0"):
    # breakpoint()
    
    out_file = json_file_path.replace("output", "evals")
    out_file = json_file_path.replace("interv", "interevals")

    with open(json_file_path, "r") as json_file:
        eval_dict = json.load(json_file)
    
    # eval_dict = {}
    return eval_dict, out_file
    
    for key, value in data.items():
        answer_text = value.get(tag, "")
        answer_text = answer_text.replace("\n", "")
        extracted_text = re.search(r"<Answer>(.*?)<\/Answer>", answer_text)
        if extracted_text:
            extracted_text = extracted_text.group(1)
        else:
            extracted_text = answer_text
        eval_dict[f"prompt - {key}"] = extracted_text
    
    with open(out_file, 'w') as json_file:
        json.dump(eval_dict, json_file, indent=4)
    return eval_dict, out_file


def extract_answers_tags_iter(data, tag = "answer_1"):
    eval_dict = {}
    # Get the last key-value pair from the dictionary
    last_key, last_value = list(data.items())[-1]

    answer_text = last_value.get(tag, "")
    answer_text = answer_text.replace("\n", "")

    extracted_text = re.search(r"<Answer>(.*?)<\/Answer>", answer_text)
    if extracted_text:
        extracted_text = extracted_text.group(1)
    else:
        extracted_text = answer_text

    eval_dict[f"prompt - {last_key}"] = extracted_text
    return eval_dict

    
def extract_answers_mcq(data, tag = "answer_2"):    
    eval_dict = {}
    
    for key, value in data.items():
        answer_text = value.get(tag, "")
        answer_text = answer_text.replace("\n", "")
        extracted_text = re.search(r"X = (.+)", answer_text)
        if extracted_text:
            extracted_text = extracted_text.group(1)
        else:
            extracted_text = answer_text
        eval_dict[f"prompt - {key}"] = extracted_text
    
    return extracted_text

def calculate_shd(G1, G2):
    # Convert the graphs to adjacency matrices
    nodes1 = list(G1.nodes)
    nodes2 = list(G2['model'].nodes)
    mapping = [nodes2.index(var) if var in nodes2 else None for var in nodes1]

    # # Convert the NetworkX graph to an adjacency matrix
    adj_matrix1 = nx.to_numpy_array(G1, nodelist=nodes1)

    # # Convert the pgmpy DAG to a NetworkX DiGraph
    adj_matrix2 = np.array(G2['adjmat'])
    adj_matrix2 = adj_matrix2[np.ix_([i for i in mapping if i is not None], [i for i in mapping if i is not None])]

    # # Check that the matrices are the same shape
    size_diff = adj_matrix1.shape[0] - adj_matrix2.shape[0]

    if size_diff > 0:
        adj_matrix2 = np.pad(adj_matrix2, ((0, size_diff), (0, size_diff)), 'constant')


    # # Calculate the SHD
    shd = np.sum(np.abs(adj_matrix1 - adj_matrix2))

    return shd

def save_clusters_to_file(clusters, file_path):
    with open(file_path, 'w') as file:
        for i, cluster in enumerate(clusters, start=1):
            formatted_cluster = ['[' + ', '.join(subcluster) + ']' for subcluster in cluster]
            line = f"Node {i}: {', '.join(formatted_cluster)} ({len(cluster)} clusters)"
            file.write(line + '\n\n')

def find_outcome_treatment(G):
    root_nodes = [n for n, d in G.in_degree() if d==0]
    leaf_nodes = [n for n, d in G.out_degree() if d==0]
    return root_nodes, leaf_nodes


def add_missing_nodes_to_csv(csv_path, replacement_nodes):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Check if each replacement node is in the DataFrame
    for node in replacement_nodes:
        if node not in df['var_name'].values:
            # If the node is not in the DataFrame, add it
            new_row = {'node': df['node'].max() + 1, 'var_name': node, 'var_description': node}
            df = df.append(new_row, ignore_index=True)

    # Write the DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)
    return df


def calculate_edge_counts(graph):
    out_edge_counts = {}
    in_edge_counts = {}
    total_edge_counts = {}

    for node in graph.nodes():
        # Calculate out-degree (edges going out of the node)
        out_degree = graph.out_degree(node)
        out_edge_counts[node] = out_degree

        # Calculate in-degree (edges coming into the node)
        in_degree = graph.in_degree(node)
        in_edge_counts[node] = in_degree
        total_edge_counts[node] = in_degree + out_degree
    
    sorted_total_edge_counts = dict(sorted(total_edge_counts.items(), key=lambda item: item[1], reverse=True))


    return out_edge_counts, in_edge_counts, sorted_total_edge_counts


def find_sandwich_pairs(graph):
    sandwich_pairs = {}
    
    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        
        if predecessors and successors:
            pairs = []
            for pred in predecessors:
                for succ in successors:
                    pairs.append((pred, succ))
            sandwich_pairs[node] = pairs
    
    return sandwich_pairs
def find_mediators(graph, start_node, end_node):

    mediators = []

    

    # Get all simple paths from start_node to end_node

    paths = list(nx.all_simple_paths(graph, start_node, end_node))

    

    for path in paths:

        # Remove the start and end nodes

        path.remove(start_node)

        path.remove(end_node)

        

        # Add the remaining nodes (mediators) to the list

        mediators.extend(path)

    

    # Remove duplicates

    mediators = list(set(mediators))

    

    return mediators


def find_all_mediators(graph):
    mediators_dict = {}
    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        if predecessors and successors:
            for pred in predecessors:
                for succ in successors:
                    if (pred, succ) in mediators_dict:
                        mediators_dict[(pred, succ)].append(node)
                    else:
                        mediators_dict[(pred, succ)] = [node]
    return mediators_dict

def find_pairs_without_mediators(graph):

    pairs_without_mediators = {}

    

    # Get all pairs of nodes

    node_pairs = list(itertools.combinations(graph.nodes(), 2))

    

    # Get all pairs with mediators

    pairs_with_mediators = find_all_mediators(graph).keys()

    

    for pair in node_pairs:

        # If a pair is not in the pairs with mediators, add it to the dictionary with an empty list as value

        if pair not in pairs_with_mediators and (pair[1], pair[0]) not in pairs_with_mediators:

            pairs_without_mediators[pair] = []

    

    return pairs_without_mediators

def generate_combinations(dictionary):
    keys = list(dictionary.keys())
    combinations = {}

    for key in keys:
        other_keys = [k for k in keys if k != key]
        combinations[key] = []
        for k in other_keys:
            for item in dictionary[k]:
                if key not in item and item not in dictionary[key]:
                    combinations[key].append(item)

    return combinations

import json



def save_prompts(lst, f):

    file_ = f.replace("interv","interv")

    prompts_dict = {}
    for i in range(len(lst)):

        prompts_dict[f"prompt_{i}"] = lst[i][0] if isinstance(lst[i], list) and lst[i] else lst[i]
    # breakpoint()
    with open(file_, 'w') as json_file:
        json.dump(prompts_dict, json_file, indent = 4)
    return 0

