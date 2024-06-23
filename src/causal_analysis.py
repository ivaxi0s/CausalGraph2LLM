import networkx as nx
import random

def get_mediators(G, treatment = "smoke", outcome = "smoke" ):
  outcome_ancestors = set(nx.ancestors(G, source=outcome))
  treatment_descendants = set(nx.descendants(G, source=treatment))
  mediators = treatment_descendants.intersection(outcome_ancestors)
  return mediators

def get_confounders():
    return 0
  

def get_sources(graph):
    return [node for node in graph.nodes() if graph.in_degree(node) == 0]

def get_sinks(graph):
    # Initialize a list to store the sinks
    sinks = []

    # Iterate over all nodes in the graph
    for node in graph.nodes():
        # Check if the node has no successors (out-degree 0)
        if graph.out_degree(node) == 0:
            # Add the node to the list of sinks
            sinks.append(node)

    return sinks

def get_colliders(graph):
    return [node for node in graph.nodes() if graph.in_degree(node) > 1]

def get_other_nodes(graph):
    sources = get_sources(graph)
    sinks = get_sinks(graph)
    return [node for node in graph.nodes() if node not in sources and node not in sinks]

def get_node_types(graph):
    return {
        'source': get_sources(graph),
        'sink': get_sinks(graph),
        'colliders': get_colliders(graph),
        'other_nodes': get_other_nodes(graph)
    }

def analyze_and_intervene(G):
    # Step 2: Find the most and least populated nodes
    degrees = G.degree()
    most_populated_node = max(degrees, key=lambda x: x[1])[0]
    least_populated_node = min(degrees, key=lambda x: x[1])[0]

    # Step 3: Perform interventions
    def perform_intervention(G, node):
        G_intervened = G.copy()
        G_intervened.remove_edges_from(list(G.in_edges(node)))
        return G_intervened

    most_intervened = perform_intervention(G, most_populated_node)
    least_intervened = perform_intervention(G, least_populated_node)

    # Step 5: Check paths between all pairs of nodes for both interventions
    def check_paths(G_intervened):
        results = []
        nodes = list(G_intervened.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                path_exists = "Yes" if nx.has_path(G_intervened, node1, node2) else "No"
                results.append((node1, node2, path_exists))
        return results

    most_results = check_paths(most_intervened)
    least_results = check_paths(least_intervened)

    def get_balanced_results(results, intervened_node):
        yes_results = [(intervened_node, r[0], r[1], r[2]) for r in results if r[2] == "Yes"]
        no_results = [(intervened_node, r[0], r[1], r[2]) for r in results if r[2] == "No"]

        # Ensure 25 "Yes" and 25 "No" results
        yes_results = random.sample(yes_results, min(25, len(yes_results)))
        no_results = random.sample(no_results, min(25, len(no_results)))

        # If there are not enough "Yes" or "No" results, fill up with the other
        while len(yes_results) < 25 and no_results:
            yes_results.append(no_results.pop())
        while len(no_results) < 25 and yes_results:
            no_results.append(yes_results.pop())

        return yes_results + no_results

    most_balanced_results = get_balanced_results(most_results, most_populated_node)
    least_balanced_results = get_balanced_results(least_results, least_populated_node)

    return most_balanced_results, least_balanced_results
