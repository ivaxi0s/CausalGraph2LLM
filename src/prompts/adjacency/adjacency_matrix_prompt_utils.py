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

def adjacency_matrix_to_string(nodes, adjacency_matrix):
    # Create the header row
    header = "  " + " ".join(nodes) + "\n"
    
    # Create each row of the adjacency matrix
    matrix_rows = []
    for i, node in enumerate(nodes):
        row = node + " " + "".join(str(cell) for cell in adjacency_matrix[i]) + "\n"
        matrix_rows.append(row)
    
    # Combine header and rows into a single string
    return header + "".join(matrix_rows)

def full_graph_to_adjacency_string(full_graph):
    nodes, adjacency_matrix = create_adjacency_matrix(full_graph)
    return adjacency_matrix_to_string(nodes, adjacency_matrix)
