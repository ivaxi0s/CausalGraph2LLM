def create_adjacency_matrix(full_graph):
    # Extract nodes and initialize adjacency matrix
    nodes = set()
    for statement in full_graph:
        parts = statement.split('causes')
        x_desc = parts[0].strip().strip('<').strip('>')
        y_desc = parts[1].strip().strip('<').strip('>')
        nodes.add(x_desc)
        nodes.add(y_desc)

    nodes = sorted(list(nodes))
    num_nodes = len(nodes)
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Populate the adjacency matrix based on the edges
    for statement in full_graph:
        parts = statement.split('causes')
        x_desc = parts[0].strip().strip('<').strip('>')
        y_desc = parts[1].strip().strip('<').strip('>')

        x_index = nodes.index(x_desc)
        y_index = nodes.index(y_desc)

        adjacency_matrix[x_index][y_index] = 1

    return nodes, adjacency_matrix

def full_graph_to_graphml(full_graph):
    nodes, adjacency_matrix = create_adjacency_matrix(full_graph)
    node_set = set()
    edge_list = []

    # Populate nodes and edges
    for statement in full_graph:
        parts = statement.split('causes')
        x_desc = parts[0].strip().strip('<').strip('>')
        y_desc = parts[1].strip().strip('<').strip('>')
        node_set.add(x_desc)
        node_set.add(y_desc)
        edge_list.append((x_desc, y_desc))

    # Start building the GraphML string
    graphml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    graphml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
    graphml += '  <graph edgedefault="directed">\n'

    # Add nodes
    for node in sorted(node_set):
        graphml += f'    <node id="{node}"/>\n'

    # Add edges
    for source, target in edge_list:
        graphml += f'    <edge source="{source}" target="{target}"/>\n'

    # Close the GraphML tags
    graphml += '  </graph>\n'
    graphml += '</graphml>'

    return graphml
