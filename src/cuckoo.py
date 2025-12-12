#Citation: https://github.com/docwza/cuckoo-search
import random
import networkx as nx
import numpy as np
import math

class Cuckoo:
    def __init__(self, path, G, eps = 0.9):
        self.path = path
        self.G = G
        self.nodes = list(G.nodes)
        self.eps = eps
        self.fitness = self.calculate_fitness()
    
    """
    Function to Compute fitness value.
    """   
    def calculate_fitness(self):
        fitness = 0.0
        
        for i in range(1, len(self.path)):
            curr_node = self.path[i-1]
            next_node = self.path[i]
            if self.G.has_edge(curr_node, next_node):
                fitness += self.G[curr_node][next_node]['weight']
            else:
                fitness += 9999  # Penalty for missing edges
        
        # Return inverse: lower cost = higher fitness
        return 1.0 / (fitness + self.eps)

    def generate_new_path(self):
        """
        This function generates a random solution (a random path) in the graph
        """
        nodes = list(self.G.nodes)
        start = nodes[0]
        end = nodes[-1]
        samples = list(nx.all_simple_paths(self.G, start, end))
        for i in range(len(samples)):
            if len(samples[i]) != len(nodes):
                extra_nodes = [node for node in nodes if node not in samples[i]]
                random.shuffle(extra_nodes)
                samples[i] = samples[i] + extra_nodes

        sample_node = random.choice(samples)
        return sample_node

class CuckooSearch:
    def __init__(self, G, num_cuckoos, max_iterations, beta):
        self.G = G
        self.nodes = list(G.nodes)
        self.num_cuckoos = num_cuckoos
        self.max_iterations = max_iterations
        self.beta = beta
        self.cuckoos = [Cuckoo(random.sample(self.nodes, len(self.nodes)), self.G) for _ in range(self.num_cuckoos)]
        self.test_results = []
        self.test_cases = 0
        self.convergence_log = []  # NEW: Track actual best cost per iteration
    
    def levy_flight(self):
        sigma = (math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / (math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, 1)
        v = np.random.normal(0, 1, 1)
        step = u / (abs(v) ** (1 / self.beta))
        return step

    def optimize(self):
        for i in range(self.max_iterations):
            for j in range(self.num_cuckoos):
                cuckoo = self.cuckoos[j]
                step = self.levy_flight()
                new_path = cuckoo.generate_new_path()
                new_cuckoo = Cuckoo(new_path, self.G)
                if new_cuckoo.fitness > cuckoo.fitness:
                    self.cuckoos[j] = new_cuckoo
                    self.test_cases+=1
            
            self.cuckoos = sorted(self.cuckoos, key=lambda x: x.fitness, reverse=True)
            best_path = self.cuckoos[0].path
            best_fitness = self.cuckoos[0].fitness
            
            # NEW: Calculate actual tour cost for this iteration
            actual_cost = 0
            for k in range(1, len(best_path)):
                if self.G.has_edge(best_path[k-1], best_path[k]):
                    actual_cost += self.G[best_path[k-1]][best_path[k]]['weight']
                else:
                    actual_cost += 9999
            
            self.convergence_log.append({
                'iteration': i,
                'best_cost': actual_cost,
                'best_fitness': best_fitness,
                'replacements': self.test_cases,
                'path_length': len(best_path)
            })
            
            self.test_results.append([i, best_fitness, self.test_cases])
        
        best_path = self.cuckoos[0].path
        best_fitness = self.cuckoos[0].fitness
        
        return best_path, best_fitness