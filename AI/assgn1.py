from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

def is_valid(state):
    """
    Check if a state is valid by ensuring that the goat is never left alone with the leopard or the grass.
    """
    M, L, G, Gr = state
    if (L == G and M != G) or (G == Gr and M != G):
        return False  # Unsafe state                    
    return True

def generate_next_states(state):
    """
    Generate possible next states from the current state.
    """
    M, L, G, Gr = state
    next_states = []
    
    # Man moves alone
    new_state = (1 - M, L, G, Gr)
    if is_valid(new_state):
        next_states.append(new_state)
    
    # Man moves with Leopard
    if M == L:
        new_state = (1 - M, 1 - L, G, Gr)
        if is_valid(new_state):
            next_states.append(new_state)
    
    # Man moves with Goat
    if M == G:
        new_state = (1 - M, L, 1 - G, Gr)
        if is_valid(new_state):
            next_states.append(new_state)
    
    # Man moves with Grass
    if M == Gr:
        new_state = (1 - M, L, G, 1 - Gr)
        if is_valid(new_state):
            next_states.append(new_state)
    
    return next_states

def bfs(start, goal):
    """
    Breadth-First Search to find the shortest path.
    """
    queue = deque([(start, [])])
    visited = set()
    
    while queue:
        state, path = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        
        if state == goal:
            return path + [state]
        
        for next_state in generate_next_states(state):
            queue.append((next_state, path + [state]))
    return None

def dfs(start, goal):
    """
    Depth-First Search to find a solution.
    """
    stack = [(start, [])]
    visited = set()
    
    while stack:
        state, path = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        
        if state == goal:
            return path + [state]
        
        for next_state in generate_next_states(state):
            stack.append((next_state, path + [state]))
    return None

def visualize_solution(solution, title):
    """
    Visualize the solution path as a graph.
    """
    G = nx.DiGraph()
    for i in range(len(solution) - 1):
        G.add_edge(solution[i], solution[i + 1])
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=8, font_weight='bold', edge_color='gray')
    plt.title(title)
    plt.show()

# Define initial and goal states
start_state = (0, 0, 0, 0)  # (M, L, G, Gr) all on left
goal_state = (1, 1, 1, 1)  # All on right

# Find solutions
bfs_solution = bfs(start_state, goal_state)
dfs_solution = dfs(start_state, goal_state)

print("BFS Solution:", bfs_solution)
print("DFS Solution:", dfs_solution)

# Visualize solutions
if bfs_solution:
    visualize_solution(bfs_solution, "BFS Solution Path")
if dfs_solution:
    visualize_solution(dfs_solution, "DFS Solution Path")


