import os, json

from src.data.data_generation import generate_dataset, generate_missing_node_data, generate_two_missing_node_level
from src.data.dag_utils import get_directed_edges, find_children_parents
from src.language_model import verbose_single_graph, verbose_double_list_graphs, verbose_list_graphs, verbose_details, verb_child_parent, make_graph_verbose
from src.prompts.cp_prompt_utils import prompt_missing_models, generate_prompts, normal_prompt_models, prompt_graph_detail
from src.eval.evaluate import evaluate, compare_children
from src.utils import extract_answers_tags, find_sandwich_pairs, generate_combinations, find_all_mediators, find_pairs_without_mediators, save_prompts

from causaldag import DAG

def run_succ_pred_node(args, codebook, out_file, save_f):
    true_G, data = generate_dataset('_raw_bayesian_nets/' + args.dataset + '.bif')
    longtrueG = make_graph_verbose(true_G, codebook)
    verbal_graph, data_graph = verbose_single_graph(true_G, codebook)
    
    out_file = f'{save_f}/{out_file}'

    if os.path.exists(out_file) and not args.rerun and not args.evaluate:
        exit()    

    if args.prompt:
        child_parent = find_children_parents(longtrueG)
        described_nodes = verbose_details(true_G, codebook)
        all_prompts = prompt_graph_detail(args,verbal_graph, described_nodes, args.node_type)
        if args.evaluate:
                
                out_dict, _ = extract_answers_tags(out_file)
                compare_children(child_parent, out_dict, args.node_type)

        else:
            save_prompts(all_prompts, out_file)
            print("done save prompts")
            # exit()
            model_answers = normal_prompt_models(all_prompts, described_nodes, args.model, args.temp, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)
            save_f = os.path.join(save_f, "parse")

            with open(out_file, 'w') as output_file:
                output_file.write(json.dumps(model_answers, indent=2))