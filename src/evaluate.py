import networkx as nx
import pandas as pd
import datetime
from cuckoo import CuckooSearch
from whale import WhaleOptimization

def run_whale_optimization(opt_func, constraints, nsols, b, a, a_step):
    whale = WhaleOptimization(opt_func, constraints, nsols, b, a, a_step, maximize=False)

def run_cuckoo_search(graph, num_cuckoos, max_iterations, beta):
    cuckoo = CuckooSearch(graph, num_cuckoos, max_iterations, beta)

    start = datetime.datetime.now()
    best_path, best_fitness = cuckoo.optimize()
    end = datetime.datetime.now()

    cuckoo_time = end - start

    cuckoo_test_data = pd.DataFrame(cuckoo.test_results,columns = ["iterations","fitness_value","test_cases"])

    print("Optimal path: ", best_path)
    print("Optimal path cost: ", best_fitness)
    print("CSA total Exec time => ", cuckoo_time.total_seconds())
    cuckoo_test_data.to_csv("cuckoo_test_data_results.csv")

def main():
    #Put traveling salesman problem here
    graph = nx.DiGraph()

    #Add nodes to the graph
    for i in range(11):
        graph.add_node(i)
        
    edges = [(0, 1,{'weight': 1}), (1, 3,{'weight': 2}), (1, 2,{'weight': 1}),(2, 4,{'weight': 2}),
            (3, 2,{'weight': 2}),(3, 4,{'weight': 1}),(3, 5,{'weight': 2}),(3, 7,{'weight': 4}),
            (4, 5,{'weight': 1}),(4, 6,{'weight': 2}),(5, 7,{'weight': 2}),(5, 8,{'weight': 3}),
            (6, 7,{'weight': 1}),(7, 9,{'weight': 2}),(8, 10,{'weight': 2}),(9, 10,{'weight': 1})]
    graph.add_edges_from(edges)

    run_cuckoo_search(graph, num_cuckoos = 30, max_iterations=1000, beta=0.27)
    # run_whale_optimization(opt_func, constraints, nsols, b, a, a_step)

if __name__ == "__main__":
    main()