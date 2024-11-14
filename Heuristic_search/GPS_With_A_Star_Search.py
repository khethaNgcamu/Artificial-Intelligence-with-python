import heapq
import math

# Define the graph as a dictionary of nodes and their neighbors with distances
graph = {
     'A': {'B': 2, 'C': 5},
    'B': {'D': 4, 'E': 10},
    'C': {'D': 2, 'F': 7},
    'D': {'G': 1},
    'E': {'G': 3},
    'F': {'G': 2},
    'G': {}  # Destination node
    }

# Define heuristic estimates (straight-line distances) to the destination 'G'
heuristic = {
    'A': 7,
    'B': 6,
    'C': 8,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 0  # The heuristic to the goal itself is zero
}

def a_star_search(start, goal):
    # Priority queue to hold nodes to be explored, sorted by f-cost (g + h)
    open_set = []
    heapq.heappush(open_set, (0, start))  # (f-cost, node)

    # Dictionaries to keep track of shortest paths
    came_from = {}
    g_cost = {node: float('inf') for node in graph}
    g_cost[start] = 0  # Start node g-cost is 0

    while open_set:
        # Get the node with the lowest f-cost
        current_f, current = heapq.heappop(open_set)

        # If we reach the goal, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse the path to get start to goal

        # Explore neighbors
        for neighbor, distance in graph[current].items():
            tentative_g_cost = g_cost[current] + distance
            if tentative_g_cost < g_cost[neighbor]:
                # Update g-cost and the path
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic[neighbor]
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[neighbor] = current

    return None  # No path found

# Find the shortest path from 'A' to 'G'
start = 'A'
goal = 'G'
path = a_star_search(start, goal)

print(f"Shortest path from {start} to {goal}: {path}")
