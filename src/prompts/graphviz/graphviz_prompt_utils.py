def full_graph_to_string(full_graph):
    edges = []
    for statement in full_graph:
        parts = statement.split('causes')
        x_desc = parts[0].strip().strip('<').strip('>')
        y_desc = parts[1].strip().strip('<').strip('>')
        edges.append(f"{x_desc} -> {y_desc}")
    return ", ".join(edges)
