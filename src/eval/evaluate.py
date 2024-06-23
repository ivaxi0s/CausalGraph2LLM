import re
import json
import ast

from src.utils import semantic_similarity, semantic_sim_lst, save_clusters_to_file, extract_answers_tags
from src.language_model import verb_child_parent
from src.causal_analysis import get_node_types

def compare_lists_f1(ground_truth, eval_dict):
    
    source_ground_truth = set(map(str, ground_truth))
    prompt_a_values = set(map(str, eval_dict))

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = source_ground_truth & prompt_a_values
    FP = prompt_a_values - source_ground_truth
    FN = source_ground_truth - prompt_a_values

    precision = len(TP) / len(prompt_a_values) if prompt_a_values else 0
    recall = len(TP) / len(source_ground_truth) if source_ground_truth else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Prepare results
    results = {
        'True Positives': len(TP),
        'False Positives': len(FP),
        'False Negatives': len(FN),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Matched Elements': list(TP),
        'Unmatched Elements in Prompt A': list(FP),
        'Unmatched Elements in Source': list(FN)
    }
    print(results)
    return results

def compare_children(ground_truth, eval_dict, node):
    # Preprocess eval_dict to ensure all values are lists or 'Null'
    for key in eval_dict:
        value = eval_dict[key]
        if value != 'Null':
            try:
                # Attempt to parse the value as a list
                parsed_value = ast.literal_eval(value)
                if not isinstance(parsed_value, list):
                    parsed_value = [parsed_value]
                eval_dict[key] = str(parsed_value)
            except (ValueError, SyntaxError):
                # If parsing fails, assume it's a single value and convert to a list
                eval_dict[key] = str([value])
    results = {}
    total_TP = total_FP = total_FN = 0
    precision_sum = recall_sum = f1_score_sum = 0
    count = accuracy_sum = 0

    for key, gt_value in ground_truth.items():
        gt_children = set(map(str, gt_value[node]))
        eval_value = eval_dict.get(f'prompt - {key}', 'Null')

        if eval_value == 'Null':
            eval_children = set()
        else:
            try:
                eval_children = set(map(str, ast.literal_eval(eval_value)))
            except (ValueError, SyntaxError):
                eval_children = set()
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = gt_children & eval_children
        FP = eval_children - gt_children
        FN = gt_children - eval_children
        TN = len(eval_dict) - len(gt_children)  # Correct calculation for TN

        precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
        recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (len(TP) + (TN)) / (len(TP) + len(FP) + len(FN) + (TN)) if (len(TP) + len(FP) + len(FN) + (TN)) > 0 else 0

        results[key] = {
            'True Positives': len(TP),
            'False Positives': len(FP),
            'False Negatives': len(FN),
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Accuracy': accuracy,
            'Matched Elements': list(TP),
            'Unmatched Elements in Eval': list(FP),
            'Unmatched Elements in Ground Truth': list(FN)
        }

        total_TP += len(TP)
        total_FP += len(FP)
        total_FN += len(FN)
        total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        total_f1_score = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        precision_sum += precision
        recall_sum += recall
        f1_score_sum += f1_score
        accuracy_sum += accuracy
        count += 1

    avg_precision = precision_sum / count if count > 0 else 0
    avg_recall = recall_sum / count if count > 0 else 0
    avg_f1_score = f1_score_sum / count if count > 0 else 0
    avg_accuracy = accuracy_sum / count if count > 0 else 0

    overall_results = {
        'True Positives': total_TP,
        'False Positives': total_FP,
        'False Negatives': total_FN,
        'Precision': total_precision,
        'Recall': total_recall,
        'F1 Score': total_f1_score,
        'Average Accuracy': avg_accuracy
    }
    print(overall_results)
    return results, overall_results

def evaluate(child_parent, codebook, file_path, node_type, described_nodes):

    json_file_path = file_path
    out_file = out_file = json_file_path.replace("output", "evals")
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    pred_answer_list = []
    suc_answer_list = []
    for key, value in data.items():
        pred_answer_text = value.get("answer_0", "")
        # suc_answer_text = value.get("answer_2", "")
        null = re.search(r"<Answer>(.*?)<\/Answer>", pred_answer_text)
        if null == "Null":
            breakpoint()
        match = re.search(r'\[(.*?)\]', pred_answer_text)
        if match:
            result_string = match.group(1)
            pred_answer_list.append([item.strip() for item in result_string.split(';')])
        else:
            pred_answer_list.append(pred_answer_text)
    
    i = 0
    for key, value in child_parent.items():
        new_fields = {
            "answer": pred_answer_list[i],
        }
        value.update(new_fields)
        i+=1
    
    verbose_child_parent = verb_child_parent(child_parent, codebook, node_type)
    score = calc_score(child_parent, node_type, described_nodes)
    # print(score)
    # breakpoint()
    with open(out_file, 'w') as json_file:
        json.dump(verbose_child_parent, json_file, indent=4)
    exit()

def calc_dist(l1, l2):
    len_l1 = len(l1)
    len_l2 = len(l2)
    # print(l1, l2)
    # breakpoint()
    if len(l2[0])==0: 
        return len_l1
    
    elif len_l1!=0 and len_l2!=0:
        l1 = [s[:20].lower() for s in l1]
        l2 = [s[:20].lower() for s in l2]
        set_l1 = set(l1)
        set_l2 = set(l2)
        dist = len(set_l1.symmetric_difference(set_l2))
        return dist

    elif len_l1==0: 
        if all(element == '' for element in l2):
            return 0
        else: return len_l2
    
def calculate_f1_score(ground_truth, prediction, all_possible_labels):
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()

    # Binarize labels in a one-vs-all fashion
    lb.fit(all_possible_labels)

    # Transform multi-class labels to binary labels
    binarized_ground_truth = lb.transform(ground_truth)
    binarized_prediction = lb.transform(prediction)

    return f1_score(binarized_ground_truth, binarized_prediction, average='macro')

def calc_score(data, node_type, described_nodes):
    p_ed = 0 
    c_ed = 0
    for key, value in data.items():
        parent = data[key][node_type]
        parent_ans = data[key]['answer']
        # breakpoint()
        compare_lists_f1(parent, parent_ans)
        print(p_ed, c_ed, data[key])
    return p_ed + c_ed

def remove_brackets(s):
    if s.startswith(' ['):
        return s[2:]
    elif s.endswith('] '):
        return s[:-2]
    return s

def eval_open_world_explain(json_data_long, json_data_short, json_file_path):
    result_long, result_short = {}, {}

    for key, value in json_data_long.items():
        new_key = key.replace("prompt - ", "")
        value = value[2:-2].split('], [')
        sentences = []
        for sublist in value:
            sentences.append(remove_brackets(sublist))
        result_long[new_key] = sentences
    for key, value in json_data_short.items():
        new_key = key.replace("prompt - ", "")
        value = value.strip()[1:-1]
        item_list = [item.strip(' "') for item in value.split(',')]
        result_short[new_key] = item_list
    
    outp = {}
    total_distance = 0
    total_count = 0
    highest_distances = []
    first_distances = []

    for (key_long, values_long),(key_short, values_short) in zip(result_long.items(), result_short.items()):
        distances_long = {value: float(semantic_similarity(key_long, value)) for value in values_long}
        distances_short = {value: float(semantic_similarity(key_short, value)) for value in values_short}
        dict3 = {k1 if v1 > v2 else k2: max(v1, v2) for (k1, v1), (k2, v2) in zip(distances_long.items(), distances_short.items())}
        # dict3 = distances_short
        sorted_distances = dict(sorted(dict3.items(), key=lambda item: item[1], reverse=True))
        avg_distance = sum(sorted_distances.values()) / len(sorted_distances)
        outp[key_short] = {"average_distance": avg_distance, "distances": sorted_distances}
        total_distance += sum(sorted_distances.values())
        total_count += len(sorted_distances)
        highest_distances.append(max(sorted_distances.values()))
        first_distances.append(next(iter(sorted_distances.values())))

    overall_avg_distance = total_distance / total_count
    outp["overall_average_distance"] = overall_avg_distance
    avg_highest_distance = sum(highest_distances) / len(highest_distances)
    outp["average_highest_distance"] = avg_highest_distance
    avg_first_distance = sum(first_distances) / len(first_distances)
    outp["average_first_distance"] = avg_first_distance

    out_file = json_file_path.replace("gpt", "preprocessed-gpt") 
    with open(out_file, 'w') as json_file:
        json.dump(outp, json_file, indent=4)
    breakpoint()
    return outp, out_file


import statistics

def eval_open_world(json_data, json_file_path = None):
    result = {}
    for key, value in json_data.items():
        new_key = key.replace("prompt - ", "")

        value = value.strip()[1:-1] 
        sentences = [sublist.strip() for sublist in value.split(',')]

        result[new_key] = sentences
    
    outp = {}
    total_distance = 0
    total_count = 0
    highest_distances = []
    first_distances = []

    for key, values in result.items():
        distances = {value: float(semantic_similarity(key, value)) for value in values}
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=True))
        avg_distance = sum(sorted_distances.values()) / len(sorted_distances)
        outp[key] = {"average_distance": avg_distance, "distances": sorted_distances}
        total_distance += sum(sorted_distances.values())
        total_count += len(sorted_distances)
        highest_distances.append(max(sorted_distances.values()))
        first_distances.append(next(iter(sorted_distances.values())))

    overall_avg_distance = total_distance / total_count
    outp["overall_average_distance"] = overall_avg_distance
    avg_highest_distance = sum(highest_distances) / len(highest_distances)
    outp["average_highest_distance"] = avg_highest_distance
    avg_first_distance = sum(first_distances) / len(first_distances)
    outp["average_first_distance"] = avg_first_distance
    range_of_highest = max(highest_distances) - min(highest_distances)
    outp["range_of_highest"] = range_of_highest
    uncertainty_of_highest = statistics.stdev(highest_distances)
    outp["uncertainty_of_highest"] = uncertainty_of_highest
    if json_file_path is not None:
        out_file = json_file_path.replace("gpt", "new-preprocessed-gpt") 
        with open(out_file, 'w') as json_file:
            json.dump(outp, json_file, indent=4)
    return outp, out_file


def eval_open_world_causes(json_data, json_file_path):
    result = {}
    outp = {}
    total_distance = 0
    total_count = 0
    highest_distances = []
    first_distances = []


    for key, value in json_data.items():
        new_key = key.replace("prompt - ", "")
        value = value.strip()[1:-1]
        item_list = [item.strip(' "') for item in value.split(',')]
        if "No" in item_list or "ot Applicabl" in item_list or '' in item_list: total_count+=0
        else: result[new_key] = item_list

    for key, values in result.items():
        if isinstance(values, str):
            values = [values]
        distances = {value: float(semantic_similarity(key, value)) for value in values}
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=True))
        # sorted_distances = distances
        avg_distance = sum(sorted_distances.values()) / len(sorted_distances)
        outp[key] = {"average_distance": avg_distance, "distances": sorted_distances}
        total_distance += sum(sorted_distances.values())
        total_count += len(sorted_distances)
        highest_distances.append(max(sorted_distances.values()))
        # first_distances.append(next(iter(sorted_distances.values())))
    
    if total_count != 0:
        overall_avg_distance = total_distance / total_count
        outp["overall_average_distance"] = overall_avg_distance
        avg_highest_distance = sum(highest_distances) / len(highest_distances)
        outp["average_highest_distance"] = avg_highest_distance

    out_file = json_file_path.replace("gpt", "preprocessed-gpt") 
    with open(out_file, 'w') as json_file:
        json.dump(outp, json_file, indent=4)
    return outp

def eval_open_world_sinks(json_data, json_file_path):
    result = {}
    outp = {}
    total_distance = 0
    total_count = 0
    highest_distances = []
    first_distances = []


    for key, value in json_data.items():
        new_key = key.replace("prompt - ", "")
        value = value.strip()[1:-1]
        item_list = [item.strip(' "') for item in value.split(',')]
        if "No" in item_list or "ot Applicabl" in item_list or '' in item_list: total_count+=0
        else: result[new_key] = item_list

    for key, values in result.items():
        if isinstance(values, str):
            values = [values]
        distances = {value: float(semantic_similarity(key, value)) for value in values}
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=True))
        # sorted_distances = distances
        avg_distance = sum(sorted_distances.values()) / len(sorted_distances)
        outp[key] = {"average_distance": avg_distance, "distances": sorted_distances}
        total_distance += sum(sorted_distances.values())
        total_count += len(sorted_distances)
        highest_distances.append(max(sorted_distances.values()))
        # first_distances.append(next(iter(sorted_distances.values())))

    if total_count != 0:
        overall_avg_distance = total_distance / total_count
        outp["overall_average_distance"] = overall_avg_distance
        avg_highest_distance = sum(highest_distances) / len(highest_distances)
        outp["average_highest_distance"] = avg_highest_distance

    out_file = json_file_path.replace("gpt", "preprocessed-gpt") 
    with open(out_file, 'w') as json_file:
        json.dump(outp, json_file, indent=4)
    return outp

def eval_clustering(dict_of_lst, filen):
    filen = filen.replace("output", "evals")
    filen = filen.replace("gpt", "clusters-gpt")
    clustered_lst = []
    for key, value in dict_of_lst.items():
        value = value.strip()[1:-1]
        item_list = [item.strip(' "') for item in value.split(',')]
        clustered_lst.append(semantic_sim_lst(item_list))
    save_clusters_to_file(clustered_lst, filen)
    return clustered_lst

def eval_open_world_top_word(json_data):
    result = {}
    for key, value in json_data.items():
        new_key = key.replace("prompt - ", "")

        value = value.strip()[1:-1] 
        sentences = [sublist.strip() for sublist in value.split(',')]

        # sentences = []
        # for sublist in value:
        #     sentences.append(remove_brackets(sublist))
        result[new_key] = sentences

    outp = {}
    for key, values in result.items():

        distances = {value: float(semantic_similarity(key, value)) for value in values}
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=True))
        top_word = next(iter(sorted_distances))
        outp[key] = top_word
        return top_word, sorted_distances[top_word]


def llm_rank_eval(op, gt):
    result = {}
    for key, value in op.items():
        new_key = key.replace("prompt - ", "")
        value = value.strip()[1:-1] 
        sentences = [sublist.strip() for sublist in value.split(',')]
        result[new_key] = sentences
    
    op_lists = [values for values in result.values()]
    gt_lists = [list(inner_dict['distances'].keys()) for inner_dict in gt.values() if isinstance(inner_dict, dict) and 'distances' in inner_dict]
    
    score = 0
    for inner_list1, inner_list2 in zip(op_lists, gt_lists):

        common_positions = sum(x == y for x, y in zip(inner_list1, inner_list2))
        score += common_positions  # Increase the score by the length of the inner list
    
    return score


def two_nodes_load_and_check_answers(json_file_path, graph):
    node_types = get_node_types(graph)
    print(node_types)
    counter = {'sources': 0, 'sinks': 0, 'colliders': 0, 'other_nodes': 0}
    confusion_counter = {'sources': 0, 'sinks': 0, 'colliders': 0, 'other_nodes': 0}
    out_file = json_file_path.replace("output", "evals")
    correct = 0
    total = 0
    confusion = 0
    eval_dict = {}
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        for key, value in data.items():
            total+=1
            tuple_key = ast.literal_eval(key)
            prompt_key = list(value.keys())[0]
            answer_key = list(value.keys())[1]

            # Extracting X value from the answer
            x_value = value[answer_key].split('=')[-1].strip().replace(".", "").lower()
            eval_dict[f"prompt - {key}"] = x_value
            # Checking if X matches the first key in the tuple
            if x_value[:len(tuple_key[0])] == tuple_key[0].lower():
                for node_type, nodes in node_types.items():
                    if tuple_key[0].lower() in map(str.lower, nodes):
                        ntype = node_type
                        break

                counter[ntype] += 1 
                correct+=1
            elif x_value[:len(tuple_key[1])] == tuple_key[1].lower():
                for node_type, nodes in node_types.items():
                    if tuple_key[1].lower() in map(str.lower, nodes):
                        ntype = node_type
                        break
                confusion_counter[ntype] += 1 
                confusion+=1

    except Exception as e:
        print(f"Error: {e}")
    print("TOATLA SCORE:", correct/total, "Confusion", confusion/total)
    print(counter)
    print(confusion)
    counter = {node_type: count / total for node_type, count in counter.items()}
    confusion = {node_type: count / total for node_type, count in confusion_counter.items()}

    print(counter)
    print(confusion)
    with open(out_file, 'w') as json_file:
        json.dump(eval_dict, json_file, indent=4)

def distances_across_different_nodes(json_data, graph):
    node_types = get_node_types(graph)
    averages = {}
    with open(json_data, 'r') as json_file:
        data = json.load(json_file)
    
    data = {key.lower(): value for key, value in data.items()}

    for node_type, nodes in node_types.items():
        highest_distances = []
        print(nodes)
        for node in nodes:
            if node.lower() in map(str.lower, data.keys()):
                highest_distance = max(data[node.lower()]['distances'].values())
                print(node_type, "-------", node)
                highest_distances.append(highest_distance)

        # Calculate the average of the highest distances
        if highest_distances:
            averages[node_type] = sum(highest_distances) / len(highest_distances)
        else:
            averages[node_type] = None
    return averages


def distances_across_different_nodes_ranked(data, graph):
    node_types = get_node_types(graph)
    averages = {}

    data = {key.lower(): value for key, value in data.items()}

    for node_type, nodes in node_types.items():
        highest_distances = []
        print(nodes)
        for node in nodes:
            if node.lower() in map(str.lower, data.keys()):
                highest_distance = float(data[node.lower()])
                print(node_type, "-------", node)
                highest_distances.append(highest_distance)

        # Calculate the average of the highest distances
        if highest_distances:
            averages[node_type] = sum(highest_distances) / len(highest_distances)
        else:
            averages[node_type] = None
    return averages


def eval_source_sink(data,ground_truth):
    predicted_values = [key.replace('prompt - ', '') for key, value in data.items() if value.strip() == 'Yes']
    result = [str(ord(char) - ord('a')) for char in predicted_values]
    intersection = set(result).intersection(ground_truth)
    total_missing = len(predicted_values) + len(ground_truth) - 2 * len(intersection) 
    accuracy = (len(data) - total_missing) / len(data)
    # see if there is intersection between them,,, if not then see if not then add the rest and then divide by total.

    result = {'Accuracy': accuracy}

    print(result)
    # print(accuracy)

def eval_mediator(data, number):
    # breakpoint()
    predicted_values = [key.replace('prompt - ', '') for key, value in data.items() if value.strip() == 'Yes']
    lst = [int(x) for x in predicted_values]
    count_after_7 = sum(1 for x in lst if x > number)
    count_in_range = number - sum(1 for x in lst if 0 <= x <= number)
    fp = count_after_7
    fn = count_in_range
    accuracy = (len(data) - count_in_range - count_after_7) / len(data)
    print(accuracy, fn, fp)
    results = {
        'False Negatives': fn,
        'False Positives': fp,
        'Accuracy': accuracy
    }
    print(results)
    return 0


def compare_lists(ground_truth, eval_dict):
    # Convert all elements to strings for consistent comparison
    key = None
    for k in eval_dict.keys():
        if k.startswith('prompt - '):
            key = k
    try:
        # Try to convert it assuming it's a well-formatted list
        prompt_a_values = ast.literal_eval(eval_dict[key])
    except (ValueError, SyntaxError):
        # If that fails, try to convert it as a string with comma separated elements
        try:
            prompt_a_values = eval_dict[key].strip('[]').split(', ')
        except Exception:
            raise ValueError("Unable to process input")
    
    source_ground_truth = set(map(str, ground_truth))
    prompt_a_values = set(map(str, prompt_a_values))
    prompt_a_values = set(map(str.strip, prompt_a_values))
    source_ground_truth = set(map(str.strip, map(str, ground_truth)))
    prompt_a_values = set(map(str.strip, map(str, prompt_a_values)))    

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = source_ground_truth & prompt_a_values
    FP = prompt_a_values - source_ground_truth
    FN = source_ground_truth - prompt_a_values

    precision = len(TP) / len(prompt_a_values) if prompt_a_values else 0
    recall = len(TP) / len(source_ground_truth) if source_ground_truth else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Prepare results
    results = {
        'True Positives': len(TP),
        'False Positives': len(FP),
        'False Negatives': len(FN),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Matched Elements': list(TP),
        'Unmatched Elements in Prompt A': list(FP),
        'Unmatched Elements in Source': list(FN)
    }
    print(results)
    return results