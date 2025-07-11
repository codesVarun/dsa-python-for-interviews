"""This is how we can represent a graph using an adjacency matrix."""

def create_adjacency_matrix(vertices, edges):
    matrix = [[0] * vertices for _ in range(vertices)]
    for u, v in edges:
        matrix[u][v] = 1
        matrix[v][u] = 1 
    return matrix # For undirected graph, add this line

adj_matix =create_adjacency_matrix(vertices=4, edges=[(0, 1), (0, 2), (1, 2), (2, 3)])

for row in adj_matix:
    print(row)



def create_waighted_adjacency_matrix(vertices, edges):
    matrix = [[0] * vertices for _ in range(vertices)]
    for u, v, weight in edges:
        matrix[u][v] = weight
        matrix[v][u] = weight  # For undirected graph, add this line
    return matrix

waighted_adj_matix =create_waighted_adjacency_matrix(vertices=4, edges=[(0, 1, 5), (0, 2, 6), (1, 2, 8), (2, 3, 9)])
for row in waighted_adj_matix:
    print(row)