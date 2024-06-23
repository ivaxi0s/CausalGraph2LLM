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

def adjacency_matrix_to_edge_list(nodes, adjacency_matrix):
    edges = []
    for i, row in enumerate(adjacency_matrix):
        for j, cell in enumerate(row):
            if cell == 1:
                edges.append((nodes[i], nodes[j]))
    return edges

def full_graph_to_edge_strings(full_graph):
    nodes, adjacency_matrix = create_adjacency_matrix(full_graph)
    edges = adjacency_matrix_to_edge_list(nodes, adjacency_matrix)
    return [f"({x}, {y})" for x, y in edges]
