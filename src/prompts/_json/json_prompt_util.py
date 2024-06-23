import json


def convert_to_json(full_graph):
    dag_json = {}

    for statement in full_graph:
        parts = statement.split('causes')
        x_desc = parts[0].strip().strip('<').strip('>')
        y_desc = parts[1].strip().strip('<').strip('>')
        
        # Add node entries if they don't exist
        if x_desc not in dag_json:
            dag_json[x_desc] = {'parents': []}
        if y_desc not in dag_json:
            dag_json[y_desc] = {'parents': []}
        
        # Append x_desc as a parent of y_desc
        dag_json[y_desc]['parents'].append(x_desc)

    return json.dumps(dag_json, indent=4)
