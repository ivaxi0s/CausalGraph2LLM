import openai
import pickle
import os
import argparse
import json
import re
import pandas as pd
import sys

import networkx as nx
from src.meta.run_binary_prompts import binary_prompter
from src.meta.run_desc_prompts import desc_prompter
from src.meta.run_child_parent_prompts import run_succ_pred_node
from src.utils import check_dir

from causaldag import DAG

API_KEY = ""


if __name__ == "__main__":
    openai.api_key = API_KEY
    
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--save_folder', type=str, default = "output", help='e.g. experiments/trained_models/my_model')
    commandLineParser.add_argument('--gpt', type=str, default = "gpt-3.5-turbo", help='gpt type')
    commandLineParser.add_argument('--seed', type=int, default = 0, help='e.g. experiments/trained_models/my_model')
    commandLineParser.add_argument('--temp', type=float, default = 0.0, help='temperature')
    commandLineParser.add_argument('--max_tokens', type=int, default = 2000, help='max token')    
    commandLineParser.add_argument('--top_p', type=float, default = 1.0, help=' ')
    commandLineParser.add_argument('--frequency_penalty', type=int, default = 0, help=' ')
    commandLineParser.add_argument('--presence_penalty', type=int, default = 0, help=' ')    
    commandLineParser.add_argument('--all', action='store_true', help='user input')
    commandLineParser.add_argument('--prompts', type=int, default = 1, help='total conversations within a prompt')
    commandLineParser.add_argument('--dataset', type=str, default = "cancer", help='which dataset to use')
    commandLineParser.add_argument('--prompt', action='store_true', help='prompt the model')
    commandLineParser.add_argument('--position_eval', action='store_true', help='do you want to evaluate the position analysis for mcq')
    commandLineParser.add_argument('--parse', action='store_true', help='childparent')
    commandLineParser.add_argument('--evaluate', action='store_true', help='do you want to evaluate the position analysis for mcq')
    commandLineParser.add_argument('--file_path', type=str, help='evaluation file')
    commandLineParser.add_argument('--no_context', action='store_true', help='without giving the context evaluate')
    commandLineParser.add_argument('--restart', type=int, default = 0, help='without giving the context evaluate')
    commandLineParser.add_argument('--depth', type=int, default = 1, help='determine the depth till which you dont subsequent nodes to be missing')
    commandLineParser.add_argument('--open_world', action='store_true', help='do you want to perform open world evaluation')
    commandLineParser.add_argument('--remove_causes', action='store_true', help='remove causes from the causal graph and then evaluate if language model can recover them')
    commandLineParser.add_argument('--rank', action='store_true', help='do you want ranking the')
    commandLineParser.add_argument('--prompt_mcq', action='store_true', help='prompt other options for mcq using semantic similarity')
    commandLineParser.add_argument('--causal', type=str, default='cs', help='user peter clarke for causal discovery using observational data')
    commandLineParser.add_argument('--n', type=int, default = 10000, help='number of observation data')
    commandLineParser.add_argument('--remove_sinks', action='store_true', help='remove causes from the causal graph and then evaluate if language model can recover them')
    commandLineParser.add_argument('--topk', type=int, default = 5, help='top-k suggestions by llm')
    commandLineParser.add_argument('--mediator_analysis', action='store_true', help='find NDE and NIE')
    commandLineParser.add_argument('--treatment', type=str, help='NDE/NIE treatment')
    commandLineParser.add_argument('--outcome', type=str, help='NDE/NIE outcome')
    commandLineParser.add_argument('--commonsense', action='store_true', help='return commonsense variables for the graph')
    commandLineParser.add_argument('--numerical', action='store_true', help='return commonsense variables for the graph')
    commandLineParser.add_argument('--syn', action='store_true', help='return synthetic graph')
    commandLineParser.add_argument('--node_type', type=str, help='mediator/source/sink')
    commandLineParser.add_argument('--order', type=str, default = "normal", help='dfs which order to prompt')
    commandLineParser.add_argument('--rerun', action='store_true', help='rerun the entire run')
    commandLineParser.add_argument('--query_type', type=str, default = "binary", help='binary/descriptive')
    commandLineParser.add_argument('--emb', type=str, default = "std", help='which type of embedding')
    commandLineParser.add_argument('--interv', action='store_true', help='intervention')


    args = commandLineParser.parse_args()
    if not args.evaluate:
        print(args)

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    if args.interv: args.save_folder = "interv"
    save_f = os.path.join(str(args.save_folder), str(args.query_type), str(args.dataset))  #add query type
    check_dir(save_f)

    file_name = f'{args.gpt}_topk-{args.topk}_temp-{args.temp}_max-token-{args.max_tokens}_top-p-{args.top_p}_frequency-{args.frequency_penalty}_presence-{args.presence_penalty}.json'
    if args.commonsense:
        file_name = f'commonsense_{args.node_type}_{args.gpt}_temp-{args.temp}_max-token-{args.max_tokens}_top-p-{args.top_p}_frequency-{args.frequency_penalty}_presence-{args.presence_penalty}_order{args.order}.json'
    if args.numerical:
        file_name = f'numerical_{args.node_type}_{args.gpt}_temp-{args.temp}_max-token-{args.max_tokens}_top-p-{args.top_p}_frequency-{args.frequency_penalty}_presence-{args.presence_penalty}_order{args.order}.json'
    if args.syn:
        file_name = f'syn_{args.node_type}_{args.gpt}_temp-{args.temp}_max-token-{args.max_tokens}_top-p-{args.top_p}_frequency-{args.frequency_penalty}_presence-{args.presence_penalty}_order{args.order}.json'
    # breakpoint()
    if args.emb != 'std':
        save_f = os.path.join(str(save_f), str(args.emb))
        eval_f = os.path.join("evals", str(args.query_type), str(args.dataset), str(args.emb))
        prompt_f = os.path.join("prompts", str(args.query_type), str(args.dataset), str(args.emb))
        check_dir(eval_f)
        check_dir(save_f)
        check_dir(prompt_f)


    if not os.path.exists("_raw_bayesian_nets"):
        from src.data.download_datasets import download_datasets
        download_datasets()

    if args.commonsense: codebook = pd.read_csv('codebooks/commonsense/' + args.dataset + '.csv') 
    elif args.syn: codebook = pd.read_csv('codebooks/syn/' + args.dataset + '.csv') 
    else: codebook = pd.read_csv('codebooks/numerical/' + args.dataset + '.csv')

    if args.node_type == "child" or args.node_type == "parent":
        run_succ_pred_node(args, codebook, file_name, save_f)
    elif args.query_type == "binary":
        binary_prompter(args, codebook, file_name, save_f)
    else:
        desc_prompter(args, codebook, file_name, save_f)

        