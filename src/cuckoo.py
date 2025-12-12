# Citation: https://github.com/docwza/cuckoo-search
import random
import networkx as nx # Used for graph handling
import numpy as np
import math

class Cuckoo:
    """
    Represents a single cuckoo (a solution, or 'nest').
    In this context, a solution is a path through a graph G.
    """
    def __init__(self, path, G, eps = 0.9):
        # The path (solution) represented by a list of nodes
        self.path = path 
        # The graph (problem space)
        self.G = G 
        self.nodes = list(G.nodes)
        # Small constant to prevent division by zero in fitness calculation
        self.eps = eps 
        # The quality of the solution (higher is better)
        self.fitness = self.calculate_fitness() 
    
    def calculate_fitness(self):
        """
        Function to compute the fitness value of the path.
        Fitness is calculated as the inverse of the total path cost (weight).
        """   
        fitness = 0.0
        
        # Calculate the total weight (cost) of the path
        for i in range(1, len(self.path)):
            curr_node = self.path[i-1]
            next_node = self.path[i]
            if self.G.has_edge(curr_node, next_node):
                # Add the edge weight (cost)
                fitness += self.G[curr_node][next_node]['weight']
            else:
                # Add a high penalty if the path segment is invalid
                fitness += 9999 
        
        # Return inverse: lower cost results in higher fitness
        # 1.0 / (cost + eps)
        return 1.0 / (fitness + self.eps) # Had to do this to switch to min problem

    # Huge Bottleneck (this is from the original code)
    # def generate_new_path(self):
    #     """
    #     Generates a random path in the graph.
    #     This simulates the cuckoo laying an egg in a random, new nest location.
    #     """
    #     nodes = list(self.G.nodes)
    #     start = nodes[0]
    #     end = nodes[-1]
        
    #     # Get all simple paths (no repeated nodes) between start and end
    #     samples = list(nx.all_simple_paths(self.G, start, end)) # THE BOTTLENECK
        
    #     # Ensure all paths include all nodes if this is a Traveling Salesman style problem
    #     for i in range(len(samples)):
    #         if len(samples[i]) != len(nodes):
    #             # Add missing nodes randomly to the end of the path
    #             extra_nodes = [node for node in nodes if node not in samples[i]]
    #             random.shuffle(extra_nodes)
    #             samples[i] = samples[i] + extra_nodes

    #     # Select one of the completed paths randomly
    #     sample_node = random.choice(samples)
    #     return sample_node

    # Built with help from Gemini
    def generate_new_path(self):
        """
        Generates a random path using nearest neighbor + randomization.
        Much faster than enumerating all paths.
        """
        nodes = list(self.G.nodes)
        
        # Start from node 0 (or random start)
        start = nodes[0]
        unvisited = set(nodes) - {start}
        path = [start]
        current = start
        
        while unvisited:
            # Get neighbors that haven't been visited
            neighbors = [n for n in self.G.neighbors(current) if n in unvisited]
            
            # If no valid neighbors, pick any unvisited node
            if not neighbors:
                neighbors = list(unvisited)
            
            # Add randomness: sometimes pick nearest, sometimes random
            if random.random() < 0.7:  # 70% nearest neighbor
                # Pick neighbor with minimum edge weight
                next_node = min(neighbors, 
                            key=lambda n: self.G[current][n]['weight'] if self.G.has_edge(current, n) else 9999)
            else:  # 30% random exploration
                next_node = random.choice(neighbors)
            
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        
        return path

class CuckooSearch:
    """
    The main Cuckoo Search optimizer class, managing the population of cuckoos (solutions).
    """
    def __init__(self, G, num_cuckoos, max_iterations, beta):
        # The problem graph
        self.G = G
        self.nodes = list(G.nodes)
        self.num_cuckoos = num_cuckoos
        self.max_iterations = max_iterations
        # Exponent for the Levy flight step size distribution (often between 1 and 2)
        self.beta = beta 
        
        # Initialize the population of cuckoos (nests) with random paths
        self.cuckoos = [Cuckoo(random.sample(self.nodes, len(self.nodes)), self.G) 
                        for _ in range(self.num_cuckoos)]
        self.test_results = []
        self.test_cases = 0
        self.convergence_log = [] # Track actual best cost per iteration
    
    def levy_flight(self):
        """
        Calculates a step size for a Levy flight, which is a random walk
        characterized by long jumps and short steps. This enhances exploration.
        """
        # Calculate the standard deviation (sigma) based on beta
        sigma = (math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                 (math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        
        # Get random values from normal distributions (u for step size, v for denominator)
        u = np.random.normal(0, sigma, 1)
        v = np.random.normal(0, 1, 1)
        
        # Calculate the Levy flight step size
        step = u / (abs(v) ** (1 / self.beta))
        return step

    def optimize(self):
        """
        The main optimization loop.
        """
        for i in range(self.max_iterations):
            # Levy Flight / Egg Laying (Generating new solutions)
            for j in range(self.num_cuckoos):
                cuckoo = self.cuckoos[j]
                step = self.levy_flight() # The step size is used conceptually/indirectly here
                
                # Generate a new candidate solution (a new 'egg' / path)
                new_path = cuckoo.generate_new_path()
                new_cuckoo = Cuckoo(new_path, self.G)
                
                # Replacing Nests (Selecting the better solution)
                # If the new egg (solution) is better than the current nest owner's solution
                if new_cuckoo.fitness > cuckoo.fitness:
                    self.cuckoos[j] = new_cuckoo # Replace the old solution
                    self.test_cases+=1
            
            # Sort the population to find the current best
            self.cuckoos = sorted(self.cuckoos, key=lambda x: x.fitness, reverse=True)
            best_path = self.cuckoos[0].path
            best_fitness = self.cuckoos[0].fitness
            
            # Log convergence data (calculating the actual cost/weight)
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
            
            # Store results for later analysis
            self.test_results.append([i, best_fitness, self.test_cases])
        
        # Final best solution found
        best_path = self.cuckoos[0].path
        best_fitness = self.cuckoos[0].fitness
        
        return best_path, best_fitness