# Graph Data Structure - DSA Notes

## Table of Contents
1. [Introduction](#introduction)
2. [Graph Terminology](#graph-terminology)
3. [Types of Graphs](#types-of-graphs)
4. [Graph Representation](#graph-representation)
5. [Graph Traversal](#graph-traversal)
6. [Important Graph Algorithms](#important-graph-algorithms)
7. [Common Graph Problems](#common-graph-problems)
8. [Time and Space Complexity](#time-and-space-complexity)
9. [Practice Problems](#practice-problems)

## Introduction

A **Graph** is a non-linear data structure consisting of vertices (nodes) and edges that connect these vertices. It's used to represent relationships between different entities.

### Real-world Applications
- Social networks (friends connections)
- Maps and navigation systems
- Computer networks
- Web page linking
- Dependency graphs in software
- Transportation networks

## Graph Terminology

- **Vertex/Node**: Individual data points in the graph
- **Edge**: Connection between two vertices
- **Adjacent**: Two vertices connected by an edge
- **Degree**: Number of edges connected to a vertex
- **Path**: Sequence of vertices connected by edges
- **Cycle**: Path that starts and ends at the same vertex
- **Connected Graph**: Every vertex is reachable from every other vertex
- **Disconnected Graph**: Contains isolated components

## Types of Graphs

### 1. Directed vs Undirected
- **Directed Graph (Digraph)**: Edges have direction (one-way)
- **Undirected Graph**: Edges have no direction (two-way)

### 2. Weighted vs Unweighted
- **Weighted Graph**: Edges have associated weights/costs
- **Unweighted Graph**: All edges have equal weight

### 3. Other Types
- **Cyclic Graph**: Contains at least one cycle
- **Acyclic Graph**: No cycles (Trees are acyclic connected graphs)
- **Complete Graph**: Every vertex is connected to every other vertex
- **Bipartite Graph**: Vertices can be divided into two sets with no edges within sets

## Graph Representation

### 1. Adjacency Matrix
```python
class GraphMatrix:
    def __init__(self, vertices):
        self.vertices = vertices
        self.matrix = [[0] * vertices for _ in range(vertices)]
    
    def add_edge(self, u, v, weight=1):
        self.matrix[u][v] = weight
        # For undirected graph
        # self.matrix[v][u] = weight
    
    def display(self):
        for row in self.matrix:
            print(row)
```

**Pros**: O(1) edge lookup, simple implementation
**Cons**: O(V²) space, inefficient for sparse graphs

### 2. Adjacency List
```python
from collections import defaultdict

class GraphList:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        # For undirected graph
        # self.graph[v].append(u)
    
    def display(self):
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")
```

**Pros**: Space efficient for sparse graphs, easy to iterate over neighbors
**Cons**: O(V) to check if edge exists

### 3. Edge List
```python
class GraphEdgeList:
    def __init__(self):
        self.edges = []
    
    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
```

**Pros**: Simple, good for algorithms like Kruskal's
**Cons**: Inefficient for most operations

## Graph Traversal

### 1. Depth-First Search (DFS)
```python
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=" ")
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    
    return visited

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            print(vertex, end=" ")
            
            # Add neighbors to stack
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return visited
```

**Time Complexity**: O(V + E)
**Space Complexity**: O(V)
**Use Cases**: Topological sorting, cycle detection, path finding

### 2. Breadth-First Search (BFS)
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

def bfs_shortest_path(graph, start, target):
    visited = set()
    queue = deque([(start, [start])])
    visited.add(start)
    
    while queue:
        vertex, path = queue.popleft()
        
        if vertex == target:
            return path
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None
```

**Time Complexity**: O(V + E)
**Space Complexity**: O(V)
**Use Cases**: Shortest path in unweighted graphs, level-order traversal

## Important Graph Algorithms

### 1. Cycle Detection

#### Undirected Graph (Using DFS)
```python
def has_cycle_undirected(graph, vertices):
    visited = set()
    
    def dfs(vertex, parent):
        visited.add(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                if dfs(neighbor, vertex):
                    return True
            elif neighbor != parent:
                return True
        
        return False
    
    for vertex in range(vertices):
        if vertex not in visited:
            if dfs(vertex, -1):
                return True
    
    return False
```

#### Directed Graph (Using DFS)
```python
def has_cycle_directed(graph, vertices):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * vertices
    
    def dfs(vertex):
        color[vertex] = GRAY
        
        for neighbor in graph[vertex]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        
        color[vertex] = BLACK
        return False
    
    for vertex in range(vertices):
        if color[vertex] == WHITE:
            if dfs(vertex):
                return True
    
    return False
```

### 2. Topological Sort
```python
def topological_sort(graph, vertices):
    in_degree = [0] * vertices
    
    # Calculate in-degrees
    for vertex in graph:
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    # Find vertices with no incoming edges
    queue = deque()
    for i in range(vertices):
        if in_degree[i] == 0:
            queue.append(i)
    
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == vertices else []
```

### 3. Shortest Path Algorithms

#### Dijkstra's Algorithm
```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances
```

#### Floyd-Warshall Algorithm
```python
def floyd_warshall(graph, vertices):
    dist = [[float('inf')] * vertices for _ in range(vertices)]
    
    # Initialize distances
    for i in range(vertices):
        dist[i][i] = 0
    
    for u in range(vertices):
        for v, weight in graph[u]:
            dist[u][v] = weight
    
    # Floyd-Warshall
    for k in range(vertices):
        for i in range(vertices):
            for j in range(vertices):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist
```

### 4. Minimum Spanning Tree

#### Kruskal's Algorithm
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal(edges, vertices):
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(vertices)
    mst = []
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
    
    return mst
```

## Common Graph Problems

### 1. Connected Components
```python
def count_connected_components(graph, vertices):
    visited = set()
    components = 0
    
    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
    
    for vertex in range(vertices):
        if vertex not in visited:
            dfs(vertex)
            components += 1
    
    return components
```

### 2. Bipartite Check
```python
def is_bipartite(graph, vertices):
    color = [-1] * vertices
    
    def dfs(vertex, c):
        color[vertex] = c
        for neighbor in graph[vertex]:
            if color[neighbor] == -1:
                if not dfs(neighbor, 1 - c):
                    return False
            elif color[neighbor] == c:
                return False
        return True
    
    for vertex in range(vertices):
        if color[vertex] == -1:
            if not dfs(vertex, 0):
                return False
    
    return True
```

### 3. Clone Graph
```python
def clone_graph(node):
    if not node:
        return None
    
    visited = {}
    
    def dfs(node):
        if node in visited:
            return visited[node]
        
        clone = Node(node.val)
        visited[node] = clone
        
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

## Time and Space Complexity

| Operation | Adjacency Matrix | Adjacency List |
|-----------|------------------|----------------|
| Add Vertex | O(V²) | O(1) |
| Add Edge | O(1) | O(1) |
| Remove Vertex | O(V²) | O(V + E) |
| Remove Edge | O(1) | O(V) |
| Check Edge | O(1) | O(V) |
| Space | O(V²) | O(V + E) |

### Algorithm Complexities
- **DFS/BFS**: O(V + E) time, O(V) space
- **Dijkstra**: O((V + E) log V) with heap
- **Floyd-Warshall**: O(V³) time, O(V²) space
- **Kruskal**: O(E log E) time
- **Topological Sort**: O(V + E) time

## Practice Problems

### Easy
1. Find if path exists between two nodes
2. Number of connected components
3. Clone an undirected graph
4. Valid path in graph

### Medium
1. Course Schedule (topological sort)
2. Number of islands
3. Rotting oranges (BFS)
4. Surrounded regions
5. Graph valid tree
6. Word ladder

### Hard
1. Alien dictionary
2. Minimum cost to make at least one valid path
3. Critical connections in network
4. Reconstruct itinerary

## Key Interview Tips

1. **Clarify the problem**: Ask about directed/undirected, weighted/unweighted
2. **Choose representation**: Adjacency list is usually preferred for interviews
3. **Consider edge cases**: Empty graph, single node, disconnected components
4. **Optimize space**: Use visited set instead of modifying original graph
5. **Practice both DFS and BFS**: Many problems can be solved with either
6. **Union-Find**: Essential for connectivity and MST problems
7. **Topological sort**: Common in dependency resolution problems

## Common Patterns
- **Island problems**: Use DFS/BFS to mark connected components
- **Shortest path**: BFS for unweighted, Dijkstra for weighted
- **Cycle detection**: DFS with colors or Union-Find
- **Bipartite**: DFS with 2-coloring
- **MST**: Kruskal's or Prim's algorithm