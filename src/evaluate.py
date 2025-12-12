import networkx as nx
import numpy as np
import pandas as pd
import datetime
import time
from cuckoo import CuckooSearch
from whale import WhaleOptimization


def calculate_tour_cost(graph, tour):
    """Calculates the total cost of a TSP tour (closed loop)."""
    cost = 0
    if not tour or len(tour) < 2:
        return 0
        
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        if graph.has_edge(u, v):
            cost += graph[u][v]['weight']
        else:
            return float('inf') 
            
    # Check if it's a closed loop (TSP)
    if tour[0] == tour[-1]:
        return cost
        
    # If not a loop, add return to start for TSP cost estimation
    start, end = tour[0], tour[-1]
    if graph.has_edge(end, start):
        cost += graph[end][start]['weight']
        
    return cost


def graph_fitness_adapter(graph_, x_batch, y_batch):
    scores = []
    num_nodes = len(graph_.nodes())
    
    for x, y in zip(x_batch, y_batch):
        unvisited = set(range(1, num_nodes))
        path = [0]
        current_node = 0
        total_weight = 0
        
        while unvisited:
            neighbors = [n for n in graph_.neighbors(current_node) if n in unvisited]
            
            if not neighbors:
                neighbors = list(unvisited)
            
            def get_priority(node_id):
                return np.sin(node_id * x) + np.cos(node_id * y)
            
            next_node = max(neighbors, key=get_priority)
            
            if graph_.has_edge(current_node, next_node):
                total_weight += graph_[current_node][next_node]['weight']
            else:
                total_weight += 9999
            
            current_node = next_node
            unvisited.remove(next_node)
            path.append(current_node)
        
        scores.append(total_weight)
        
    return np.array(scores)


def run_cuckoo(graph_):
    csa = CuckooSearch(graph_, num_cuckoos=30, max_iterations=50, beta=1.5)
    
    # Force all cuckoo paths to start with node 0
    for cuckoo in csa.cuckoos:
        if cuckoo.path[0] != 0:
            path = cuckoo.path
            if 0 in path:
                path.remove(0)
            path.insert(0, 0)
            cuckoo.path = path
            cuckoo.fitness = cuckoo.calculate_fitness()

    start = datetime.datetime.now()
    best_path, best_fitness = csa.optimize()
    end = datetime.datetime.now()

    csa_time = end - start

    actual_cost = calculate_tour_cost(graph_, best_path)

    print("Cuckoo Search Algorithm Output:")
    print(f"Optimal path: {best_path}")
    print(f"Optimal path cost: {actual_cost}")
    print(f"CSA total Exec time => {csa_time.total_seconds()}")
    
    # Save convergence log to CSV
    df_cuckoo = pd.DataFrame(csa.convergence_log)
    df_cuckoo.to_csv('cuckoo_convergence.csv', index=False)
    print(f"\nCuckoo convergence saved to 'cuckoo_convergence.csv'")
    print("\nFirst 10 iterations:")
    print(df_cuckoo.head(10).to_string(index=False))
    print("\nLast 10 iterations:")
    print(df_cuckoo.tail(10).to_string(index=False))
    print("\n" + "="*60 + "\n")


def run_whale(graph_):
    nsols = 50
    b = 1.0
    a = 2.0
    max_iter = 50
    a_step = a / max_iter
    
    constraints = [[-5.0, 5.0], [-5.0, 5.0]]
    
    whale = WhaleOptimization(graph_fitness_adapter, graph_, constraints, nsols, b, a, a_step, maximize=False)

    start = datetime.datetime.now()
    for iteration in range(max_iter):
        whale.optimize()
    end = datetime.datetime.now()
    whale_time = end - start

    best_fitness_entry = sorted(whale._best_solutions, key=lambda x: x[0])[0]
    best_score = best_fitness_entry[0]
    best_x, best_y = best_fitness_entry[1]

    # Reconstruct the tour
    num_nodes = len(graph_.nodes())
    unvisited = set(range(1, num_nodes))
    path = [0]
    curr = 0
    total_cost = 0
    
    while unvisited:
        neighbors = [n for n in graph_.neighbors(curr) if n in unvisited]
        
        if not neighbors:
            neighbors = list(unvisited)
        
        def get_priority(node_id):
            return np.sin(node_id * best_x) + np.cos(node_id * best_y)
            
        nxt = max(neighbors, key=get_priority)
        
        if graph_.has_edge(curr, nxt):
            total_cost += graph_[curr][nxt]['weight']
        else:
            total_cost += 9999
        
        curr = nxt
        unvisited.remove(nxt)
        path.append(curr)
        
    print("Whale Optimization Algorithm Output:")
    print(f"Optimal path: {path}")
    print(f"Optimal path cost: {total_cost}")
    print(f"WOA total Exec time => {whale_time.total_seconds()}")
    
    # Save convergence log to CSV
    df_whale = pd.DataFrame(whale.convergence_log)
    df_whale.to_csv('whale_convergence.csv', index=False)
    print(f"\nWhale convergence saved to 'whale_convergence.csv'")
    print("\nFirst 10 iterations:")
    print(df_whale.head(10).to_string(index=False))
    print("\nLast 10 iterations:")
    print(df_whale.tail(10).to_string(index=False))
    print()


def main():
    Gn = nx.DiGraph()

    # 1. Define the Mapping (Smaller subset)
    # 0: ATL, 1: BOS, 2: DEN, 3: DFW, 4: LAX, 5: ORD, 6: SFO
    
    # 2. Add nodes (0 to 6)
    for i in range(7):
        Gn.add_node(i)

    # 3. Reduced Airport Data
    edges = [
        (0, 1, {'weight': 946}), (0, 3, {'weight': 731}), (0, 5, {'weight': 606}),
        (1, 0, {'weight': 946}), (1, 2, {'weight': 1754}), (1, 5, {'weight': 867}),
        (2, 3, {'weight': 641}), (2, 4, {'weight': 862}), (2, 5, {'weight': 888}), (2, 6, {'weight': 967}),
        (3, 0, {'weight': 731}), (3, 2, {'weight': 641}), (3, 4, {'weight': 1235}), (3, 5, {'weight': 802}),
        (4, 2, {'weight': 862}), (4, 3, {'weight': 1235}), (4, 6, {'weight': 337}),
        (5, 0, {'weight': 606}), (5, 1, {'weight': 867}), (5, 2, {'weight': 888}), (5, 3, {'weight': 802}),
        (6, 2, {'weight': 967}), (6, 4, {'weight': 337})
    ]

    Gn.add_edges_from(edges)
    
    run_cuckoo(Gn)
    run_whale(Gn)


if __name__ == "__main__":
    main()