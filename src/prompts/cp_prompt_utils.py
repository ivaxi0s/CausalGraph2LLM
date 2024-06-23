from causaldag import DAG
import itertools, random, time
import openai
import json
import requests
import replicate

from itertools import chain

from src.data.dag_utils import get_directed_edges, find_children_parents
from src.prompts.context import description
from src.verbose import verbose_single_graph, verbose_lookup_nodes
from src.prompts.prompt_manager import graph_represenations

BLABLADOR_KEY = ""

def into_sentence(lst):
    sentences = []

    for sentence in lst:
        modified_sentence = sentence.capitalize().strip() + '.'
        sentences.append(modified_sentence)

    result = ' '.join(sentences)
    return result

def prompt_gpt(prompts, gpt, temp, max_tokens, top_p, frequency_penalty, presence_penalty):

    # conversation=[{"role": "system", "content": "Strictly follow the format to return the answer. It should be in this  Answer: X = _choice"}]
    conversation=[{"role": "system", "content": "Strictly follow the format to return the answer"}]
    ans_dict = {}

    if "gpt" not in gpt:
        url = 'https://helmholtz-blablador.fz-juelich.de:8000/v1/chat/completions'
        headers = {
            'accept': 'application/json',
            'Authorization': BLABLADOR_KEY,
            'Content-Type': 'application/json',
        }


    for i in range(len(prompts)):
        input_prompt = prompts[i]
        conversation.append({"role": "user", "content": input_prompt})
        # breakpoint()
        # for count in range(1,100):
        if "gpt" in gpt:
            response = openai.ChatCompletion.create(
                model=gpt,
                temperature=temp,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                messages= conversation
            )
            
            #     continue  
                  
            answer = response["choices"][0]["message"]["content"]
            print(answer)
        
        elif "llama" in gpt:
            ans = replicate.run(
            "meta/llama-2-70b-chat",
            input={
                "debug": False,
                "top_k": 50,
                "top_p": 1,
                "prompt": input_prompt,
                "temperature": 0.01,
                "system_prompt": "Strictly follow the format to return the answer. I should be in this  <Answer> [list of suggestions seperated by commas] </Answer>. For example if the suggestions are A,B,C,D,E then - <Answer> [A, B, C, D, E] </Answer>  The suggestions list should be just a list. ",
                "max_new_tokens": 500,
                "min_new_tokens": -1,
                "repetition_penalty": 1.15
                },
            )
            answer =  ''.join(ans)
            print(answer)
        else:
            data = {
                "model": gpt,
                "messages": conversation,
                "temperature": temp,
                "top_p": top_p,
                "top_k": -1,
                "n": 1,
                "max_tokens": max_tokens,
                "stop": ["string"],
                "stream": False,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "user": "string"
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response_dict = response.json()
            answer = response_dict['choices'][0]['message']['content']
            print(answer)
        ans_dict[f"prompt_{i}"] = input_prompt
        ans_dict[f"answer_{i}"] = answer
        conversation.append({"role": "assistant", "content": answer})
    return ans_dict

def get_prompt(args, graph_list, node = None, node_type = None):
    if args.commonsense: descp = "The context of the graph is " + description(args) + ". " 
    else: descp = ""
    prompt_1 = "Hello. You will be given a causal graph." + descp + "The causal relationships in this causal graph are - " + graph_represenations(args,graph_list)
    if node_type == "parent":
        prompt_2 = '''
            Now using the graph information, can you please tell me what directly causes '''+ node + '''. Return your answer in the following format.
            Keep the names the same as what was described in the graph. Think step by step. Give reasoning and then give answer within <Answer> [a1,a2,a3..] </Answer>, if Null then return <Answer>Null</Answer>. Make sure your answer is in the answer tags with list or null, nothing else.
        '''
    else:
        prompt_2 = '''
            Now using the graph information, can you please tell me what is directly caused by '''+ node + '''. Return your answer in the following format.
            Keep the names the same as what was described in the graph. Think step by step. Give reasoning and then give answer within <Answer> [a1,a2,a3..] </Answer>, if Null then return <Answer>Null</Answer>. Make sure your answer is in the answer tags with list or null, nothing else.
        '''
    return [prompt_1 + prompt_2]

def prompt_graph_detail(args, graph_list, described_nodes, node_type):
    full_prompt = []
    for node in described_nodes:
        all_p = get_prompt(args, graph_list, node=node, node_type=node_type)
        full_prompt.append(all_p)
    return full_prompt

def normal_prompt_models(prompt_list, described_nodes, gpt, temp, max_tokens, top_p, frequency_penalty, presence_penalty):
    full_dict = {}
    for i in range(len(prompt_list)):
        print("Prompt number ------------------------", i)
        ans_dict = prompt_gpt(prompt_list[i], gpt, temp, max_tokens, top_p, frequency_penalty, presence_penalty)
        full_dict[described_nodes[i]] = ans_dict
        # time.sleep(90)
    return full_dict

def prompt_graph_detail_mediator(args, graph_list, described_nodes, node_type, mediator_variables):
    full_prompt = []
    for key, item in mediator_variables.items():
        for i in item:
            all_p = get_prompt(args, graph_list, node=key, node_type=node_type, mediator_variables=i)
            full_prompt.append(all_p)
    return full_prompt


########new file
def prompt_missing_models(prompt_list, missing_desc, gpt, temp, max_tokens, top_p, frequency_penalty, presence_penalty, restart, out_file, full_dict, multiple_missing_nodes = False):
    for i in range(restart, len(prompt_list)):
        print("Prompt number ------------------------", i)
        ans_dict = prompt_gpt(prompt_list[i], gpt, temp, max_tokens, top_p, frequency_penalty, presence_penalty)
        full_dict[f"{missing_desc[i]}"] = ans_dict
        with open(out_file, 'w') as json_file:
            json.dump(full_dict, json_file, indent=2)
        # time.sleep(60)
    return full_dict

def generate_prompts(args, graph_list, missing_nodes, multiple_missing_nodes = False, position_eval = False, open_world = False): # iterate over graph list & missing node together
    full_prompt = []

    for i in range(len(missing_nodes)):
        mcq = ["weather", "book sales", "movie rating"]
        random_index = random.randint(0, len(mcq))

        if multiple_missing_nodes:
            mcq.insert(random_index, missing_nodes[i][0])
            random_index_y = random.randint(0, len(mcq))
            mcq.insert(random_index_y, missing_nodes[i][1])
        else:
            mcq.insert(random_index, missing_nodes[i])

        if position_eval:
            mcq_perm = list(itertools.permutations(mcq))
            mcq_perm = [list(perm) for perm in mcq_perm]
            p_list = []
            for mc in mcq_perm:
                p = get_prompt(graph_list[i], mc)
                p_list.append(p)
            p = p_list
        else:
            p = get_prompt(args, graph_list[i], mcq=mcq, open_world = open_world)

        full_prompt.append(p)

    return full_prompt

def open_world_sink_nodes_prompt(args, graph_list, missing_nodes, codebook):

    prompts = []
    for i, (key, value) in enumerate(missing_nodes.items()):
        starting = "Hello. You will be given a causal graph. The context of the graph is " + description(args) + ". Please understand the causal relationships between the variables - " + into_sentence(graph_list[i])
        sink = "Using your commonsense and causal graph analysis, can you suggest whether a node could exists which is caused by " + into_words(verbose_lookup_nodes(value, codebook))
        fmt = ". Return your answer as Yes or No in the following format <Answer>Yes/No</Answer>"
        inquire = "If your answer is Yes, then can you suggest what could that node be. Give 5 suggestions for what could that node be. Give reason for your suggestions. Finally return your answer (without reasoning) in the following format: <Answer> [first suggestion, second suggestion, third suggestion, fourth suggestion, fifth suggestion] </Answer> If your answer is no, return </Answer> Not Applicable </Answer>" 
        prompts.append([starting, sink+fmt, inquire])
    
    return prompts

def open_world_missing_nodes_prompt(args, graph_list, missing_nodes, codebook):

    prompts = []
    for i, (key, value) in enumerate(missing_nodes.items()):
        starting = "Hello. You will be given a causal graph. The context of the graph is " + description(args) + ". Please understand the causal relationships between the variables - " + into_sentence(graph_list[i])
        cause = "Using your commonsense and causal graph analysis, can you suggest whether a node could exists that causes " + into_words(verbose_lookup_nodes(value['causes'], codebook))
        fmt = ". Return your answer as Yes or No in the following format <Answer>Yes/No</Answer>"
        inquire = "If your answer is Yes, then can you suggest what could that node be. Give 5 suggestions for what could that node be. Give reason for your suggestions. Finally return your answer (without reasoning) in the following format: <Answer> [first suggestion, second suggestion, third suggestion, fourth suggestion, fifth suggestion] </Answer> If your answer is no, return Not Applicable. Return your answer in the following format <Answer>Yes/No</Answer>." 
        prompts.append([starting, cause+fmt, inquire])
    
    return prompts

def into_words(lst):
    ans = ""
    for i in range(len(lst)):
        ans +=lst[i]
        if len(lst)-i != 1:
            ans += " and "
    return ans    
