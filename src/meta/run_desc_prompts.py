import os, json

from src.data.data_generation import generate_dataset, generate_missing_node_data, generate_two_missing_node_level
from src.data.dag_utils import get_directed_edges, find_children_parents
from src.language_model import verbose_single_graph, verbose_double_list_graphs, verbose_list_graphs, verbose_details, verb_child_parent, make_graph_verbose
from src.prompts.desc_prompt_utils import prompt_missing_models, generate_prompts, normal_prompt_models, prompt_graph_detail, prompt_graph_detail_mediator
from src.eval.evaluate import evaluate, eval_source_sink, eval_mediator, compare_lists
from src.utils import extract_answers_tags, find_sandwich_pairs, generate_combinations, find_all_mediators, find_pairs_without_mediators, save_prompts
from src.causal_analysis import get_node_types
from causaldag import DAG

def desc_prompter(args, codebook, out_file, save_f):
    true_G, data = generate_dataset('_raw_bayesian_nets/' + args.dataset + '.bif')        
    verbal_graph, data_graph = verbose_single_graph(true_G, codebook, args.order)
    out_file = f'{save_f}/{out_file}'
    if os.path.exists(out_file) and not args.rerun and not args.evaluate:
        exit()

    if args.prompt:
        child_parent = find_children_parents(true_G) #find node for each plant
        described_nodes = verbose_details(true_G, codebook) #
        if args.node_type == "mediator":
            longtrueG = make_graph_verbose(true_G, codebook)
            mediators = find_all_mediators(longtrueG)
            combined_mediators = find_pairs_without_mediators(longtrueG)
            m_prompts = prompt_graph_detail_mediator(args, verbal_graph, described_nodes, args.node_type, mediators)
            extra_prompts = prompt_graph_detail_mediator(args, verbal_graph, described_nodes, args.node_type, combined_mediators)
            all_prompts = m_prompts + extra_prompts[:(len(m_prompts)*2)]
            print(len(all_prompts), len(m_prompts))
            nodes = range(len(all_prompts))
        else:
            all_prompts = prompt_graph_detail(args, verbal_graph, described_nodes, args.node_type)
            nodes = list(true_G.nodes())


        if args.evaluate:
                eval_dict, out_file = extract_answers_tags(out_file)
                all_verb_nodes = make_graph_verbose(true_G, codebook)
                all_nodes = get_node_types(all_verb_nodes)
                if args.node_type == 'source' or args.node_type == 'sink':
                     groundtruth = all_nodes[args.node_type]
                     compare_lists(groundtruth, eval_dict)
                elif args.node_type == 'mediator':
                     eval_mediator(eval_dict, len(all_prompts) - len(extra_prompts))
        else:
            save_prompts(all_prompts, out_file)
            # exit()
            model_answers = normal_prompt_models(all_prompts, nodes, args.gpt, args.temp, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)
            with open(out_file, 'w') as output_file:
                output_file.write(json.dumps(model_answers, indent=2))

