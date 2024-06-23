from src.prompts._json.json_prompt_util import convert_to_json
from src.prompts.adjacency.adjacency_list_prompt_utils import full_graph_to_edge_strings
from src.prompts.adjacency.adjacency_matrix_prompt_utils import full_graph_to_adjacency_string
from src.prompts.graphml.graphml_prompt_utils import full_graph_to_graphml
from src.prompts.graphviz.graphviz_prompt_utils import full_graph_to_string
from src.prompts.multinode.multinode_prompt_util import verbose_multinode_graph


def into_sentence(lst):
    sentences = []

    for sentence in lst:
        modified_sentence = sentence.capitalize().strip() + '.'
        sentences.append(modified_sentence)

    result = ' '.join(sentences)
    return result

def graph_represenations(args, verbal_graph):
    
    if args.emb == 'json':
        full_verbal_causal_graph = convert_to_json(verbal_graph)
    elif args.emb == 'adj':
        full_verbal_causal_graph = into_sentence(full_graph_to_edge_strings(verbal_graph))
    elif args.emb == 'adj-matrix':
        full_verbal_causal_graph = full_graph_to_adjacency_string(verbal_graph)
    elif args.emb == 'graphml':
        full_verbal_causal_graph = full_graph_to_graphml(verbal_graph)
    elif args.emb == 'graphviz':
        full_verbal_causal_graph = full_graph_to_string(verbal_graph)
    elif args.emb == 'multinode':
        full_verbal_causal_graph = verbose_multinode_graph(verbal_graph)
    else:
        full_verbal_causal_graph = into_sentence(verbal_graph)

    return full_verbal_causal_graph