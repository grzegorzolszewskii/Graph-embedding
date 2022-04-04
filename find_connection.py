# graf ze strony https://onestepcode.com/graph-shortest-path-python/?utm_source=rss&utm_medium=rss&utm_campaign=graph-shortest-path-python
from graph_import import load_graph

graph = load_graph(330)
print(graph)


def bfs(graph, node1, node2):
    path_list = [[node1]]
    path_index = 0
    # To keep track of previously visited nodes
    previous_nodes = {node1}
    if node1 == node2:
        return path_list[0]

    while path_index < len(path_list):
        current_path = path_list[path_index]
        last_node = current_path[-1]
        next_nodes = graph[last_node]
        # Search goal node
        if node2 in next_nodes:
            current_path.append(node2)
            return current_path
        # Add new paths
        for next_node in next_nodes:
            if next_node not in previous_nodes:
                new_path = current_path[:]
                new_path.append(next_node)
                path_list.append(new_path)
                # To avoid backtracking
                previous_nodes.add(next_node)
        # Continue to next path in list
        path_index += 1
    # No path is found
    return []


def longest_paths(graph, distance):
    pairs = []
    for i in range(1, len(graph)+1):
        for j in range(1, len(graph)+1):
            if len(bfs(graph, i, j)) >= distance:
                pairs.append((i, j))
    return pairs


if __name__ == "__main__":
    print(longest_paths(graph, 12))
    print(bfs(graph, 198, 241))  # 11 wierzcholkow od 198 do 241
