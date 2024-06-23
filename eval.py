# import subprocess
# import json
# import statistics
# import re

# # Function to run the Python command and capture the output
# def run_and_capture(gpt, emb, dataset, node_type):
#     result = subprocess.run(
#         ["python", "main.py", "--gpt", gpt, "--emb", emb, "--dataset", dataset, "--syn", "--prompt", "--node_type", node_type, "--interv", "--query_type", "binary", "--evaluate"],
#         capture_output=True,
#         text=True
#     )
#     return result.stdout

# # Function to parse the output and extract metrics
# def parse_output(output, metric):
#     pattern = {
#         "True Positives": r"'True Positives': (\d+)",
#         "False Positives": r"'False Positives': (\d+)",
#         "False Negatives": r"'False Negatives': (\d+)",
#         "Precision": r"'Precision': ([0-9]*\.?[0-9]+)",
#         "Recall": r"'Recall': ([0-9]*\.?[0-9]+)",
#         "F1 Score": r"'F1 Score': ([0-9]*\.?[0-9]+)",
#         "Accuracy": r"'Accuracy': ([0-9]*\.?[0-9]+)",
#     }
#     match = re.search(pattern[metric], output)
#     if match:
#         return float(match.group(1))
#     return 0

# # Main script
# gpt = "gemini"
# emb_list = ["json", "adj", "adj-matrix", "graphml", "graphviz", "multinode", "std"]
# datasets = ["syn_20_20", "syn_20_30", "syn_30_30", "syn_30_40"]
# # datasets = ["alarm"]
# node_types = ["mediator"]
# metrics = ["Accuracy", "False Positives",  "False Negatives"]

# for emb in emb_list:
#     for node_type in node_types:
#         metric_values = {metric: [] for metric in metrics}
#         for dataset in datasets:
#             output = run_and_capture(gpt, emb, dataset, node_type)
#             print(output)
#             for metric in metrics:
#                 value = parse_output(output, metric)
#                 metric_values[metric].append(value)

#         # Compute and print the averages for each metric
#         for metric in metrics:
#             if metric_values[metric]:
#                 average = statistics.mean(metric_values[metric])
#             else:
#                 average = 0
#             print(f"Average {metric} for {gpt} with {emb} and {node_type}: {average:.2f}")
#         print("---------------------------------")

import subprocess
import json
import statistics
import re

# Function to run the Python command and capture the output
def run_and_capture(gpt, emb, dataset, node_type):
    result = subprocess.run(
        ["python", "main.py", "--gpt", gpt, "--emb", emb, "--dataset", dataset, "--syn", "--prompt", "--interv", "--query_type", "binary", "--evaluate"],
        capture_output=True,
        text=True
    )
    return result.stdout

# Function to parse the output and extract metrics
def parse_output(output, metric):
    pattern = {
        "True Positives": r"'True Positives': (\d+)",
        "False Positives": r"'False Positives': (\d+)",
        "False Negatives": r"'False Negatives': (\d+)",
        "Precision": r"'Precision': ([0-9]*\.?[0-9]+)",
        "Recall": r"'Recall': ([0-9]*\.?[0-9]+)",
        "F1 Score": r"'F1 Score': ([0-9]*\.?[0-9]+)",
        "Accuracy": r"'Accuracy': ([0-9]*\.?[0-9]+)",

    }
    match = re.search(pattern[metric], output)
    if match:
        return float(match.group(1))
    return 0

gpt = "alias-embeddings"
emb_list = ["json", "adj", "adj-matrix", "graphml", "graphviz", "multinode", "std"]
datasets = ["syn_30_40"]
node_types = ["mediator"]
metrics = ["False Negatives", "False Positives",  "Accuracy"]

for node_type in node_types:
    metric_values = {metric: [] for metric in metrics}
    for emb in emb_list:
        for dataset in datasets:
            output = run_and_capture(gpt, emb, dataset, None)
            print(output)
            for metric in metrics:
                value = parse_output(output, metric)
                metric_values[metric].append(value)

    # Compute and print the averages for each metric
    for metric in metrics:
        if metric_values[metric]:
            average = statistics.mean(metric_values[metric])
        else:
            average = 0
        print(f"Average {metric} for {gpt} with {node_type} over all embeddings: {average:.2f}")
    print("---------------------------------")