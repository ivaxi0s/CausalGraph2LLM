def verbose_multinode_graph(full_graph):
    causes_dict = {}

    # Process each statement in full_graph
    for statement in full_graph:
        parts = statement.split('causes')
        x_desc = parts[0].strip().strip('<').strip('>')
        y_desc = parts[1].strip().strip('<').strip('>')

        if x_desc not in causes_dict:
            causes_dict[x_desc] = []
        causes_dict[x_desc].append(y_desc)

    # Generate multi-node description
    multi_node_descriptions = []
    for cause, effects in sorted(causes_dict.items()):
        effects_str = ', '.join(sorted(effects))
        multi_node_descriptions.append(f"{cause} causes {effects_str}")

    # Combine descriptions into a single string with better formatting
    return ". ".join(multi_node_descriptions) + "."
# from src.prompts.multinode.multinode_prompt_util import verbose_multinode_graph
# verbose_multinode_graph(verbal_graph)