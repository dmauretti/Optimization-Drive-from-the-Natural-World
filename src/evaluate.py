import networkx as nx
import numpy as np
import pandas as pd
import datetime
from cuckoo import CuckooSearch
from whale import WhaleOptimization
from ant import AntColony


def calculate_tour_cost(graph, tour):
    # Calculates the total distance of a path, including the return trip to the start.
    cost = 0
    if not tour or len(tour) < 2:
        return 0
        
    # Sum weights for each step in the path
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        if graph.has_edge(u, v):
            cost += graph[u][v]['weight']
        else:
            return float('inf') # Invalid tour (edge doesn't exist)
            
    # If the tour is already a closed loop (ends at start), we are done
    if tour[0] == tour[-1]:
        return cost
        
    # Otherwise, add the cost of the final leg back to the start
    start, end = tour[0], tour[-1]
    if graph.has_edge(end, start):
        cost += graph[end][start]['weight']
    else:
        cost += 99999 # Heavy penalty if the return flight doesn't exist
        
    return cost

def graph_fitness_adapter(graph_, x_batch, y_batch):
    # Citation: Help from Gemini
    # Converts Whale Optimization parameters (x, y) into a path cost (fitness).
    # Lower cost = Better fitness.
    scores = []
    num_nodes = len(graph_.nodes())
    
    # Process each whale in the batch
    for x, y in zip(x_batch, y_batch):
        unvisited = set(range(1, num_nodes))
        path = [0] # Always start at node 0
        current_node = 0
        total_weight = 0
        
        # Build a path by visiting all nodes
        while unvisited:
            # Find valid neighbors that haven't been visited yet
            neighbors = [n for n in graph_.neighbors(current_node) if n in unvisited]
            
            # Dead end handling: If stuck, allow jumping to any unvisited node
            if not neighbors:
                neighbors = list(unvisited)
            
            # The "Priority Function": Determines which node to visit next based on sin/cos of node ID
            def get_priority(node_id):
                return np.sin(node_id * x) + np.cos(node_id * y)
            
            # Pick the neighbor with the highest priority score
            next_node = max(neighbors, key=get_priority)
            
            # Add edge weight to total cost (add huge penalty if edge doesn't exist)
            if graph_.has_edge(current_node, next_node):
                total_weight += graph_[current_node][next_node]['weight']
            else:
                total_weight += 99999 
            
            current_node = next_node
            unvisited.remove(next_node)
            path.append(current_node)
        
        # Add the final return leg cost
        if graph_.has_edge(current_node, 0):
            total_weight += graph_[current_node][0]['weight']
        else:
            total_weight += 99999
        
        scores.append(total_weight)
        
    return np.array(scores)

def path_to_airports(path):
    airport_names = {
        0: "ATL", 1: "BOS", 2: "DEN", 3: "DFW", 4: "LAX", 5: "ORD", 6: "SFO",
        7: "SEA", 8: "MIA", 9: "JFK", 10: "PHX", 11: "IAH", 12: "CLT", 13: "MSP",
        14: "DTW", 15: "LAS", 16: "PDX", 17: "SAN", 18: "SLC", 19: "AUS"
    }
    for i in range(len(path)):
        path[i] = airport_names[path[i]]
    
    return path

def run_cuckoo(graph_):
    # Initialize Cuckoo Search
    csa = CuckooSearch(graph_, num_cuckoos=100, max_iterations=2000, beta=1.5)
    
    # Ensure all random starting paths actually begin at node 0
    for cuckoo in csa.cuckoos:
        if cuckoo.path[0] != 0:
            path = cuckoo.path
            if 0 in path:
                path.remove(0)
            path.insert(0, 0) # Force start at 0
            cuckoo.path = path
            cuckoo.fitness = cuckoo.calculate_fitness()

    # Execute Optimization
    start = datetime.datetime.now()
    best_path, best_fitness = csa.optimize()
    end = datetime.datetime.now()

    csa_time = end - start
    
    # Close the loop back for TSP
    if best_path[-1] != 0:
        best_path.append(0)

    # Calculate Cost
    actual_cost = calculate_tour_cost(graph_, best_path)

    print(f"Optimal path: {path_to_airports(best_path)}")
    print(f"Optimal path cost: {actual_cost}")
    print(f"CSA total Exec time => {csa_time.total_seconds()}")
    
    df_cuckoo = pd.DataFrame(csa.convergence_log)
    df_cuckoo.to_csv('cuckoo_convergence.csv', index=False)

def run_whale(graph_):
    # Initialize Parameters
    nsols = 100
    b = 1.0 
    a = 2.0 
    max_iter = 2000 
    a_step = a / max_iter 
    
    # Search space constraints for x and y parameters
    constraints = [[-5.0, 5.0], [-5.0, 5.0]]
    
    # Initialize Whale Optimization
    whale = WhaleOptimization(graph_fitness_adapter, graph_, constraints, nsols, b, a, a_step, maximize=False)

    start = datetime.datetime.now()
    for _ in range(max_iter):
        whale.optimize()
    end = datetime.datetime.now()
    whale_time = end - start

    # Retrieve the best solution found
    best_fitness_entry = sorted(whale._best_solutions, key=lambda x: x[0])[0]
    best_score = best_fitness_entry[0]
    best_x, best_y = best_fitness_entry[1]

    # Citation: Help from Gemini
    # Re-run the path generation logic using the best (x, y) found to see the actual route
    num_nodes = len(graph_.nodes())
    unvisited = set(range(1, num_nodes))
    path = [0]
    curr = 0
    total_cost = 0
    
    while unvisited:
        neighbors = [n for n in graph_.neighbors(curr) if n in unvisited]
        
        # Handle dead ends
        if not neighbors:
            neighbors = list(unvisited)
        
        # Priority logic using best parameters
        def get_priority(node_id):
            return np.sin(node_id * best_x) + np.cos(node_id * best_y)
            
        nxt = max(neighbors, key=get_priority)
        
        # Calculate step cost
        if graph_.has_edge(curr, nxt):
            total_cost += graph_[curr][nxt]['weight']
        else:
            total_cost += 9999
        
        curr = nxt
        unvisited.remove(nxt)
        path.append(curr)
    
    # Add return for TSP
    if graph_.has_edge(curr, 0):
        total_cost += graph_[curr][0]['weight']
        path.append(0)
    else:
        total_cost += 99999
        path.append(0)
        
    print(f"Optimal path: {path_to_airports(path)}")
    print(f"Optimal path cost: {total_cost}")
    print(f"WOA total Exec time => {whale_time.total_seconds()}")
    
    df_whale = pd.DataFrame(whale.convergence_log)
    df_whale.to_csv('whale_convergence.csv', index=False)

def run_ant(graph_):
    # Create a distance matrix
    num_nodes = 20
    distances = np.full((num_nodes, num_nodes), np.inf)

    # Fill in the distances from the graph
    for u, v in graph_.edges:
        distances[u, v] = graph_[u][v]['weight']
    
    # Set diagonal to np.inf
    np.fill_diagonal(distances, np.inf)

    ant_colony = AntColony(distances, 50, 50, 2000, 0.5, alpha=1, beta=5)
    start = datetime.datetime.now()
    shortest_path = ant_colony.run()
    end = datetime.datetime.now()
    ant_time = end - start

    # Create list of nodes in path out of tuples
    path = [shortest_path[0][0][0]]
    for edge in shortest_path[0]:
        path.append(edge[1])

    actual_cost = calculate_tour_cost(graph_, path)

    print(f"Optimal path: {path_to_airports(path)}")
    print(f"Actual tour cost: {actual_cost}")
    print(f"Ant Colony total Exec time => {ant_time.total_seconds()}")

    df_ant = pd.DataFrame(ant_colony.convergence_log)
    df_ant.to_csv('ant_convergence.csv', index=False)


def main():
    Gn = nx.DiGraph()

    # Setup Nodes (20 Airports)
    # 0: ATL, 1: BOS, 2: DEN, ... 19: AUS
    for i in range(20):
        Gn.add_node(i)

    # Flight Data
    # Comparison for just Whale and Cuckoo
    edges2 = [
        # ATL (0) connections
        (0, 1, {'weight': 946}), (0, 2, {'weight': 1199}), (0, 3, {'weight': 731}), 
        (0, 4, {'weight': 1946}), (0, 5, {'weight': 606}), (0, 6, {'weight': 2139}),
        (0, 7, {'weight': 2182}), (0, 8, {'weight': 594}), (0, 9, {'weight': 760}),
        (0, 10, {'weight': 1587}), (0, 11, {'weight': 701}), (0, 12, {'weight': 226}),
        (0, 13, {'weight': 906}), (0, 14, {'weight': 595}), (0, 15, {'weight': 1747}),
        
        # BOS (1) connections
        (1, 0, {'weight': 946}), (1, 2, {'weight': 1754}), (1, 3, {'weight': 1562}),
        (1, 4, {'weight': 2611}), (1, 5, {'weight': 867}), (1, 6, {'weight': 2704}),
        (1, 7, {'weight': 2496}), (1, 8, {'weight': 1258}), (1, 9, {'weight': 188}),
        (1, 10, {'weight': 2300}), (1, 11, {'weight': 1605}), (1, 12, {'weight': 841}),
        (1, 13, {'weight': 1123}), (1, 14, {'weight': 632}), (1, 15, {'weight': 2376}),
        
        # DEN (2) connections
        (2, 0, {'weight': 1199}), (2, 1, {'weight': 1754}), (2, 3, {'weight': 641}),
        (2, 4, {'weight': 862}), (2, 5, {'weight': 888}), (2, 6, {'weight': 967}),
        (2, 7, {'weight': 1024}), (2, 8, {'weight': 1726}), (2, 9, {'weight': 1626}),
        (2, 10, {'weight': 586}), (2, 11, {'weight': 879}), (2, 12, {'weight': 1335}),
        (2, 13, {'weight': 680}), (2, 14, {'weight': 1123}), (2, 15, {'weight': 628}),
        (2, 16, {'weight': 991}), (2, 17, {'weight': 853}), (2, 18, {'weight': 391}),
        (2, 19, {'weight': 775}),
        
        # DFW (3) connections
        (3, 0, {'weight': 731}), (3, 1, {'weight': 1562}), (3, 2, {'weight': 641}),
        (3, 4, {'weight': 1235}), (3, 5, {'weight': 802}), (3, 6, {'weight': 1464}),
        (3, 7, {'weight': 1660}), (3, 8, {'weight': 1111}), (3, 9, {'weight': 1391}),
        (3, 10, {'weight': 868}), (3, 11, {'weight': 225}), (3, 12, {'weight': 937}),
        (3, 13, {'weight': 862}), (3, 14, {'weight': 986}), (3, 15, {'weight': 1055}),
        (3, 17, {'weight': 1179}), (3, 18, {'weight': 989}), (3, 19, {'weight': 189}),
        
        # LAX (4) connections
        (4, 0, {'weight': 1946}), (4, 1, {'weight': 2611}), (4, 2, {'weight': 862}),
        (4, 3, {'weight': 1235}), (4, 5, {'weight': 1745}), (4, 6, {'weight': 337}),
        (4, 7, {'weight': 954}), (4, 8, {'weight': 2342}), (4, 9, {'weight': 2475}),
        (4, 10, {'weight': 370}), (4, 11, {'weight': 1374}), (4, 12, {'weight': 2125}),
        (4, 13, {'weight': 1535}), (4, 14, {'weight': 1979}), (4, 15, {'weight': 236}),
        (4, 16, {'weight': 834}), (4, 17, {'weight': 109}), (4, 18, {'weight': 590}),
        (4, 19, {'weight': 1247}),
        
        # ORD (5) connections
        (5, 0, {'weight': 606}), (5, 1, {'weight': 867}), (5, 2, {'weight': 888}),
        (5, 3, {'weight': 802}), (5, 4, {'weight': 1745}), (5, 6, {'weight': 1846}),
        (5, 7, {'weight': 1721}), (5, 8, {'weight': 1197}), (5, 9, {'weight': 740}),
        (5, 10, {'weight': 1453}), (5, 11, {'weight': 940}), (5, 12, {'weight': 599}),
        (5, 13, {'weight': 355}), (5, 14, {'weight': 235}), (5, 15, {'weight': 1521}),
        (5, 18, {'weight': 1249}),
        
        # SFO (6) connections
        (6, 0, {'weight': 2139}), (6, 1, {'weight': 2704}), (6, 2, {'weight': 967}),
        (6, 3, {'weight': 1464}), (6, 4, {'weight': 337}), (6, 5, {'weight': 1846}),
        (6, 7, {'weight': 679}), (6, 8, {'weight': 2585}), (6, 9, {'weight': 2586}),
        (6, 10, {'weight': 651}), (6, 11, {'weight': 1635}), (6, 12, {'weight': 2296}),
        (6, 13, {'weight': 1584}), (6, 14, {'weight': 2077}), (6, 15, {'weight': 414}),
        (6, 16, {'weight': 550}), (6, 17, {'weight': 447}), (6, 18, {'weight': 600}),
        
        # SEA (7) connections
        (7, 0, {'weight': 2182}), (7, 1, {'weight': 2496}), (7, 2, {'weight': 1024}),
        (7, 3, {'weight': 1660}), (7, 4, {'weight': 954}), (7, 5, {'weight': 1721}),
        (7, 6, {'weight': 679}), (7, 8, {'weight': 2734}), (7, 9, {'weight': 2408}),
        (7, 10, {'weight': 1107}), (7, 11, {'weight': 1891}), (7, 13, {'weight': 1399}),
        (7, 14, {'weight': 1927}), (7, 15, {'weight': 867}), (7, 16, {'weight': 129}),
        (7, 17, {'weight': 1050}), (7, 18, {'weight': 701}),
        
        # MIA (8) connections
        (8, 0, {'weight': 594}), (8, 1, {'weight': 1258}), (8, 2, {'weight': 1726}),
        (8, 3, {'weight': 1111}), (8, 4, {'weight': 2342}), (8, 5, {'weight': 1197}),
        (8, 6, {'weight': 2585}), (8, 7, {'weight': 2734}), (8, 9, {'weight': 1092}),
        (8, 10, {'weight': 1972}), (8, 11, {'weight': 964}), (8, 12, {'weight': 649}),
        (8, 14, {'weight': 1152}), (8, 19, {'weight': 1085}),
        
        # JFK (9) connections
        (9, 0, {'weight': 760}), (9, 1, {'weight': 188}), (9, 2, {'weight': 1626}),
        (9, 3, {'weight': 1391}), (9, 4, {'weight': 2475}), (9, 5, {'weight': 740}),
        (9, 6, {'weight': 2586}), (9, 7, {'weight': 2408}), (9, 8, {'weight': 1092}),
        (9, 10, {'weight': 2153}), (9, 11, {'weight': 1428}), (9, 12, {'weight': 544}),
        (9, 13, {'weight': 1018}), (9, 14, {'weight': 509}), (9, 15, {'weight': 2248}),
        
        # PHX (10) connections
        (10, 0, {'weight': 1587}), (10, 1, {'weight': 2300}), (10, 2, {'weight': 586}),
        (10, 3, {'weight': 868}), (10, 4, {'weight': 370}), (10, 5, {'weight': 1453}),
        (10, 6, {'weight': 651}), (10, 7, {'weight': 1107}), (10, 8, {'weight': 1972}),
        (10, 9, {'weight': 2153}), (10, 11, {'weight': 1009}), (10, 12, {'weight': 1773}),
        (10, 13, {'weight': 1280}), (10, 14, {'weight': 1690}), (10, 15, {'weight': 256}),
        (10, 17, {'weight': 304}), (10, 18, {'weight': 504}), (10, 19, {'weight': 872}),
        
        # IAH (11) connections
        (11, 0, {'weight': 701}), (11, 1, {'weight': 1605}), (11, 2, {'weight': 879}),
        (11, 3, {'weight': 225}), (11, 4, {'weight': 1374}), (11, 5, {'weight': 940}),
        (11, 6, {'weight': 1635}), (11, 7, {'weight': 1891}), (11, 8, {'weight': 964}),
        (11, 9, {'weight': 1428}), (11, 10, {'weight': 1009}), (11, 12, {'weight': 928}),
        (11, 13, {'weight': 1034}), (11, 14, {'weight': 1093}), (11, 15, {'weight': 1224}),
        (11, 19, {'weight': 140}),
        
        # CLT (12) connections
        (12, 0, {'weight': 226}), (12, 1, {'weight': 841}), (12, 2, {'weight': 1335}),
        (12, 3, {'weight': 937}), (12, 4, {'weight': 2125}), (12, 5, {'weight': 599}),
        (12, 6, {'weight': 2296}), (12, 8, {'weight': 649}), (12, 9, {'weight': 544}),
        (12, 10, {'weight': 1773}), (12, 11, {'weight': 928}), (12, 13, {'weight': 906}),
        (12, 14, {'weight': 505}), (12, 15, {'weight': 1887}),
        
        # MSP (13) connections
        (13, 0, {'weight': 906}), (13, 1, {'weight': 1123}), (13, 2, {'weight': 680}),
        (13, 3, {'weight': 862}), (13, 4, {'weight': 1535}), (13, 5, {'weight': 355}),
        (13, 6, {'weight': 1584}), (13, 7, {'weight': 1399}), (13, 9, {'weight': 1018}),
        (13, 10, {'weight': 1280}), (13, 11, {'weight': 1034}), (13, 12, {'weight': 906}),
        (13, 14, {'weight': 528}), (13, 15, {'weight': 1299}), (13, 18, {'weight': 991}),
        
        # DTW (14) connections
        (14, 0, {'weight': 595}), (14, 1, {'weight': 632}), (14, 2, {'weight': 1123}),
        (14, 3, {'weight': 986}), (14, 4, {'weight': 1979}), (14, 5, {'weight': 235}),
        (14, 6, {'weight': 2077}), (14, 7, {'weight': 1927}), (14, 8, {'weight': 1152}),
        (14, 9, {'weight': 509}), (14, 10, {'weight': 1690}), (14, 11, {'weight': 1093}),
        (14, 12, {'weight': 505}), (14, 13, {'weight': 528}), (14, 15, {'weight': 1765}),
        
        # LAS (15) connections
        (15, 0, {'weight': 1747}), (15, 1, {'weight': 2376}), (15, 2, {'weight': 628}),
        (15, 3, {'weight': 1055}), (15, 4, {'weight': 236}), (15, 5, {'weight': 1521}),
        (15, 6, {'weight': 414}), (15, 7, {'weight': 867}), (15, 9, {'weight': 2248}),
        (15, 10, {'weight': 256}), (15, 11, {'weight': 1224}), (15, 12, {'weight': 1887}),
        (15, 13, {'weight': 1299}), (15, 14, {'weight': 1765}), (15, 16, {'weight': 763}),
        (15, 17, {'weight': 258}), (15, 18, {'weight': 362}),
        
        # PDX (16) connections
        (16, 2, {'weight': 991}), (16, 4, {'weight': 834}), (16, 6, {'weight': 550}),
        (16, 7, {'weight': 129}), (16, 10, {'weight': 1009}), (16, 15, {'weight': 763}),
        (16, 17, {'weight': 923}), (16, 18, {'weight': 630}),
        
        # SAN (17) connections
        (17, 2, {'weight': 853}), (17, 3, {'weight': 1179}), (17, 4, {'weight': 109}),
        (17, 6, {'weight': 447}), (17, 7, {'weight': 1050}), (17, 10, {'weight': 304}),
        (17, 15, {'weight': 258}), (17, 16, {'weight': 923}), (17, 18, {'weight': 626}),
        (17, 19, {'weight': 1141}),
        
        # SLC (18) connections
        (18, 2, {'weight': 391}), (18, 3, {'weight': 989}), (18, 4, {'weight': 590}),
        (18, 5, {'weight': 1249}), (18, 6, {'weight': 600}), (18, 7, {'weight': 701}),
        (18, 10, {'weight': 504}), (18, 13, {'weight': 991}), (18, 15, {'weight': 362}),
        (18, 16, {'weight': 630}), (18, 17, {'weight': 626}),
        
        # AUS (19) connections
        (19, 2, {'weight': 775}), (19, 3, {'weight': 189}), (19, 4, {'weight': 1247}),
        (19, 8, {'weight': 1085}), (19, 10, {'weight': 872}), (19, 11, {'weight': 140}),
        (19, 17, {'weight': 1141}),
    ]

    #Comparison for Whale, Cuckoo, and Ant (complete graph)
    edges = [
        # ATL (0) connections
        (0, 1, {'weight': 946}), (0, 2, {'weight': 1199}), (0, 3, {'weight': 731}), 
        (0, 4, {'weight': 1946}), (0, 5, {'weight': 606}), (0, 6, {'weight': 2139}),
        (0, 7, {'weight': 2182}), (0, 8, {'weight': 594}), (0, 9, {'weight': 760}),
        (0, 10, {'weight': 1587}), (0, 11, {'weight': 701}), (0, 12, {'weight': 226}),
        (0, 13, {'weight': 906}), (0, 14, {'weight': 595}), (0, 15, {'weight': 1747}),
        (0, 16, {'weight': 1888}), (0, 17, {'weight': 1765}), (0, 18, {'weight': 1545}),
        (0, 19, {'weight': 1321}),
        
        # BOS (1) connections
        (1, 0, {'weight': 946}), (1, 2, {'weight': 1754}), (1, 3, {'weight': 1562}),
        (1, 4, {'weight': 2611}), (1, 5, {'weight': 867}), (1, 6, {'weight': 2704}),
        (1, 7, {'weight': 2496}), (1, 8, {'weight': 1258}), (1, 9, {'weight': 188}),
        (1, 10, {'weight': 2300}), (1, 11, {'weight': 1605}), (1, 12, {'weight': 841}),
        (1, 13, {'weight': 1123}), (1, 14, {'weight': 632}), (1, 15, {'weight': 2376}),
        (1, 16, {'weight': 1111}), (1, 17, {'weight': 987}), (1, 18, {'weight': 468}),
        (1, 19, {'weight': 1234}),
        
        # DEN (2) connections
        (2, 0, {'weight': 1199}), (2, 1, {'weight': 1754}), (2, 3, {'weight': 641}),
        (2, 4, {'weight': 862}), (2, 5, {'weight': 888}), (2, 6, {'weight': 967}),
        (2, 7, {'weight': 1024}), (2, 8, {'weight': 1726}), (2, 9, {'weight': 1626}),
        (2, 10, {'weight': 586}), (2, 11, {'weight': 879}), (2, 12, {'weight': 1335}),
        (2, 13, {'weight': 680}), (2, 14, {'weight': 1123}), (2, 15, {'weight': 628}),
        (2, 16, {'weight': 991}), (2, 17, {'weight': 853}), (2, 18, {'weight': 391}),
        (2, 19, {'weight': 775}),
        
        # DFW (3) connections
        (3, 0, {'weight': 731}), (3, 1, {'weight': 1562}), (3, 2, {'weight': 641}),
        (3, 4, {'weight': 1235}), (3, 5, {'weight': 802}), (3, 6, {'weight': 1464}),
        (3, 7, {'weight': 1660}), (3, 8, {'weight': 1111}), (3, 9, {'weight': 1391}),
        (3, 10, {'weight': 868}), (3, 11, {'weight': 225}), (3, 12, {'weight': 937}),
        (3, 13, {'weight': 862}), (3, 14, {'weight': 986}), (3, 15, {'weight': 1055}),
        (3, 16, {'weight': 1001}), (3, 17, {'weight': 1179}), (3, 18, {'weight': 989}), 
        (3, 19, {'weight': 189}),
        
        # LAX (4) connections
        (4, 0, {'weight': 1946}), (4, 1, {'weight': 2611}), (4, 2, {'weight': 862}),
        (4, 3, {'weight': 1235}), (4, 5, {'weight': 1745}), (4, 6, {'weight': 337}),
        (4, 7, {'weight': 954}), (4, 8, {'weight': 2342}), (4, 9, {'weight': 2475}),
        (4, 10, {'weight': 370}), (4, 11, {'weight': 1374}), (4, 12, {'weight': 2125}),
        (4, 13, {'weight': 1535}), (4, 14, {'weight': 1979}), (4, 15, {'weight': 236}),
        (4, 16, {'weight': 834}), (4, 17, {'weight': 109}), (4, 18, {'weight': 590}),
        (4, 19, {'weight': 1247}),
        
        # ORD (5) connections
        (5, 0, {'weight': 606}), (5, 1, {'weight': 867}), (5, 2, {'weight': 888}),
        (5, 3, {'weight': 802}), (5, 4, {'weight': 1745}), (5, 6, {'weight': 1846}),
        (5, 7, {'weight': 1721}), (5, 8, {'weight': 1197}), (5, 9, {'weight': 740}),
        (5, 10, {'weight': 1453}), (5, 11, {'weight': 940}), (5, 12, {'weight': 599}),
        (5, 13, {'weight': 355}), (5, 14, {'weight': 235}), (5, 15, {'weight': 1521}),
        (5, 16, {'weight': 1800}), (5, 17, {'weight': 765}), (5, 18, {'weight': 1249}),
        (5, 19, {'weight': 1995}),
        
        # SFO (6) connections
        (6, 0, {'weight': 2139}), (6, 1, {'weight': 2704}), (6, 2, {'weight': 967}),
        (6, 3, {'weight': 1464}), (6, 4, {'weight': 337}), (6, 5, {'weight': 1846}),
        (6, 7, {'weight': 679}), (6, 8, {'weight': 2585}), (6, 9, {'weight': 2586}),
        (6, 10, {'weight': 651}), (6, 11, {'weight': 1635}), (6, 12, {'weight': 2296}),
        (6, 13, {'weight': 1584}), (6, 14, {'weight': 2077}), (6, 15, {'weight': 414}),
        (6, 16, {'weight': 550}), (6, 17, {'weight': 447}), (6, 18, {'weight': 600}),
        (6, 19, {'weight': 1560}), 

        # SEA (7) connections
        (7, 0, {'weight': 2182}), (7, 1, {'weight': 2496}), (7, 2, {'weight': 1024}),
        (7, 3, {'weight': 1660}), (7, 4, {'weight': 954}), (7, 5, {'weight': 1721}),
        (7, 6, {'weight': 679}), (7, 8, {'weight': 2734}), (7, 9, {'weight': 2408}),
        (7, 10, {'weight': 1107}), (7, 11, {'weight': 1891}), (7, 12, {'weight': 2356}),
        (7, 13, {'weight': 1399}), (7, 14, {'weight': 1927}), (7, 15, {'weight': 867}), 
        (7, 16, {'weight': 129}), (7, 17, {'weight': 1050}), (7, 18, {'weight': 701}), 
        (7, 19, {'weight': 1773}),
        
        # MIA (8) connections
        (8, 0, {'weight': 594}), (8, 1, {'weight': 1258}), (8, 2, {'weight': 1726}),
        (8, 3, {'weight': 1111}), (8, 4, {'weight': 2342}), (8, 5, {'weight': 1197}),
        (8, 6, {'weight': 2585}), (8, 7, {'weight': 2734}), (8, 9, {'weight': 1092}),
        (8, 10, {'weight': 1972}), (8, 11, {'weight': 964}), (8, 12, {'weight': 649}),
        (8, 13, {'weight': 1511}), (8, 14, {'weight': 1152}), (8, 15, {'weight': 2174}),
        (8, 16, {'weight': 2701}), (8, 17, {'weight': 2271}), (8, 18, {'weight': 2002}),
        (8, 19, {'weight': 1085}),
        
        # JFK (9) connections
        (9, 0, {'weight': 760}), (9, 1, {'weight': 188}), (9, 2, {'weight': 1626}),
        (9, 3, {'weight': 1391}), (9, 4, {'weight': 2475}), (9, 5, {'weight': 740}),
        (9, 6, {'weight': 2586}), (9, 7, {'weight': 2408}), (9, 8, {'weight': 1092}),
        (9, 10, {'weight': 2153}), (9, 11, {'weight': 1428}), (9, 12, {'weight': 544}),
        (9, 13, {'weight': 1018}), (9, 14, {'weight': 509}), (9, 15, {'weight': 2248}),
        (9, 16, {'weight': 2438}), (9, 17, {'weight': 2425}), (9, 18, {'weight': 1990}),
        (9, 19, {'weight': 1521}),
        
        # PHX (10)
        (10, 0, {'weight': 1587}), (10, 1, {'weight': 2300}), (10, 2, {'weight': 586}),
        (10, 3, {'weight': 868}), (10, 4, {'weight': 370}), (10, 5, {'weight': 1453}),
        (10, 6, {'weight': 651}), (10, 7, {'weight': 1107}), (10, 8, {'weight': 1972}),
        (10, 9, {'weight': 2153}), (10, 11, {'weight': 1009}), (10, 12, {'weight': 1773}),
        (10, 13, {'weight': 1280}), (10, 14, {'weight': 1690}), (10, 15, {'weight': 256}),
        (10, 16, {'weight': 1009}), (10, 17, {'weight': 304}), (10, 18, {'weight': 504}),
        (10, 19, {'weight': 872}),
        
        # IAH (11) connections
        (11, 0, {'weight': 701}), (11, 1, {'weight': 1605}), (11, 2, {'weight': 879}),
        (11, 3, {'weight': 225}), (11, 4, {'weight': 1374}), (11, 5, {'weight': 940}),
        (11, 6, {'weight': 1635}), (11, 7, {'weight': 1891}), (11, 8, {'weight': 964}),
        (11, 9, {'weight': 1428}), (11, 10, {'weight': 1009}), (11, 12, {'weight': 928}),
        (11, 13, {'weight': 1034}), (11, 14, {'weight': 1093}), (11, 15, {'weight': 1224}),
        (11, 16, {'weight': 1902}), (11, 17, {'weight': 1303}), (11, 18, {'weight': 1200}),
        (11, 19, {'weight': 140}),
        
        # CLT (12) connections
        (12, 0, {'weight': 226}), (12, 1, {'weight': 841}), (12, 2, {'weight': 1335}),
        (12, 3, {'weight': 937}), (12, 4, {'weight': 2125}), (12, 5, {'weight': 599}),
        (12, 6, {'weight': 2296}), (12, 7, {'weight': 2356}), (12, 8, {'weight': 649}),
        (12, 9, {'weight': 544}), (12, 10, {'weight': 1773}), (12, 11, {'weight': 928}),
        (12, 13, {'weight': 906}), (12, 14, {'weight': 505}), (12, 15, {'weight': 1887}),
        (12, 16, {'weight': 2377}), (12, 17, {'weight': 2015}), (12, 18, {'weight': 1652}),
        (12, 19, {'weight': 1068}),
        
        # MSP (13) connections
        (13, 0, {'weight': 906}), (13, 1, {'weight': 1123}), (13, 2, {'weight': 680}),
        (13, 3, {'weight': 862}), (13, 4, {'weight': 1535}), (13, 5, {'weight': 355}),
        (13, 6, {'weight': 1584}), (13, 7, {'weight': 1399}), (13, 8, {'weight': 1511}),
        (13, 9, {'weight': 1018}), (13, 10, {'weight': 1280}), (13, 11, {'weight': 1034}),
        (13, 12, {'weight': 906}), (13, 14, {'weight': 528}), (13, 15, {'weight': 1299}),
        (13, 16, {'weight': 1427}), (13, 17, {'weight': 1426}), (13, 18, {'weight': 991}),
        (13, 19, {'weight': 974}),
        
        # DTW (14) connections
        (14, 0, {'weight': 595}), (14, 1, {'weight': 632}), (14, 2, {'weight': 1123}),
        (14, 3, {'weight': 986}), (14, 4, {'weight': 1979}), (14, 5, {'weight': 235}),
        (14, 6, {'weight': 2077}), (14, 7, {'weight': 1927}), (14, 8, {'weight': 1152}),
        (14, 9, {'weight': 509}), (14, 10, {'weight': 1690}), (14, 11, {'weight': 1093}),
        (14, 12, {'weight': 505}), (14, 13, {'weight': 528}), (14, 15, {'weight': 1765}),
        (14, 16, {'weight': 1938}), (14, 17, {'weight': 1888}), (14, 18, {'weight': 1359}),
        (14, 19, {'weight': 1105}),
        
        # LAS (15) connections
        (15, 0, {'weight': 1747}), (15, 1, {'weight': 2376}), (15, 2, {'weight': 628}),
        (15, 3, {'weight': 1055}), (15, 4, {'weight': 236}), (15, 5, {'weight': 1521}),
        (15, 6, {'weight': 414}), (15, 7, {'weight': 867}), (15, 8, {'weight': 2174}),
        (15, 9, {'weight': 2248}), (15, 10, {'weight': 256}), (15, 11, {'weight': 1224}),
        (15, 12, {'weight': 1887}), (15, 13, {'weight': 1299}), (15, 14, {'weight': 1765}),
        (15, 16, {'weight': 763}), (15, 17, {'weight': 258}), (15, 18, {'weight': 362}),
        (15, 19, {'weight': 1167}),
        
        # PDX (16) connections
        (16, 0, {'weight': 1888}), (16, 1, {'weight': 1111}), (16, 2, {'weight': 991}),
        (16, 3, {'weight': 1001}), (16, 4, {'weight': 834}), (16, 5, {'weight': 1800}),
        (16, 6, {'weight': 550}), (16, 7, {'weight': 129}), (16, 8, {'weight': 2701}),
        (16, 9, {'weight': 2438}), (16, 10, {'weight': 1009}), (16, 11, {'weight': 1902}),
        (16, 12, {'weight': 2377}), (16, 13, {'weight': 1427}), (16, 14, {'weight': 1938}),
        (16, 15, {'weight': 763}), (16, 17, {'weight': 923}), (16, 18, {'weight': 630}),
        (16, 19, {'weight': 1745}),
        
        # SAN (17) connections
        (17, 0, {'weight': 1765}), (17, 1, {'weight': 987}), (17, 2, {'weight': 853}),
        (17, 3, {'weight': 1179}), (17, 4, {'weight': 109}), (17, 5, {'weight': 765}),
        (17, 6, {'weight': 447}), (17, 7, {'weight': 1050}), (17, 8, {'weight': 2271}),
        (17, 9, {'weight': 2425}), (17, 10, {'weight': 304}), (17, 11, {'weight': 1303}),
        (17, 12, {'weight': 2015}), (17, 13, {'weight': 1426}), (17, 14, {'weight': 1888}),
        (17, 15, {'weight': 258}), (17, 16, {'weight': 923}), (17, 18, {'weight': 626}),
        (17, 19, {'weight': 1141}),
        
        # SLC (18) connections
        (18, 0, {'weight': 1545}), (18, 1, {'weight': 468}), (18, 2, {'weight': 391}),
        (18, 3, {'weight': 989}), (18, 4, {'weight': 590}), (18, 5, {'weight': 1249}),
        (18, 6, {'weight': 600}), (18, 7, {'weight': 701}), (18, 8, {'weight': 2002}),
        (18, 9, {'weight': 1990}), (18, 10, {'weight': 504}), (18, 11, {'weight': 1200}),
        (18, 12, {'weight': 1652}), (18, 13, {'weight': 991}), (18, 14, {'weight': 1359}),
        (18, 15, {'weight': 362}), (18, 16, {'weight': 630}), (18, 17, {'weight': 626}),
        (18, 19, {'weight': 920}),
        
        # AUS (19) connections
        (19, 0, {'weight': 1321}), (19, 1, {'weight': 1234}), (19, 2, {'weight': 775}),
        (19, 3, {'weight': 189}), (19, 4, {'weight': 1247}), (19, 5, {'weight': 1995}),
        (19, 6, {'weight': 1560}), (19, 7, {'weight': 1773}), (19, 8, {'weight': 1085}),
        (19, 9, {'weight': 1521}), (19, 10, {'weight': 872}), (19, 11, {'weight': 140}),
        (19, 12, {'weight': 1068}), (19, 13, {'weight': 974}), (19, 14, {'weight': 1105}),
        (19, 15, {'weight': 1167}), (19, 16, {'weight': 1745}), (19, 17, {'weight': 1141}),
        (19, 18, {'weight': 920}),
    ]


    Gn.add_edges_from(edges)
    
    # Run Optimization Algorithms
    run_cuckoo(Gn)
    run_whale(Gn)
    run_ant(Gn)

if __name__ == "__main__":
    main()
