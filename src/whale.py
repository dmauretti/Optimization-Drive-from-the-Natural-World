#Citation: https://github.com/docwza/woa/tree/master
import numpy as np

class WhaleOptimization():
    """
    Implements the Whale Optimization Algorithm (WOA) to find the best solution
    (minimum or maximum) for a given optimization function within a defined search space.
    """
    
    def __init__(self, opt_func, graph, constraints, nsols, b, a, a_step, maximize=False):
        """
        Initializes the WOA optimizer.

        :param opt_func: The function to be optimized (calculates fitness/cost).
        :param graph: Contextual data needed by opt_func (e.g., a graph structure).
        :param constraints: Boundaries for the search space (min/max for each dimension).
        :param nsols: Number of whales in the population.
        :param b: Constant for defining the logarithmic spiral shape (used in the attack phase).
        :param a: Initial value of the 'a' parameter (linearly decreases over iterations).
        :param a_step: The amount 'a' decreases by each iteration.
        :param maximize: If True, the goal is to maximize the fitness (otherwise, minimize).
        """
        self._opt_func = opt_func
        self.graph = graph
        self._constraints = constraints
        self._sols = self._init_solutions(nsols) # Initialize the population of solutions (whales) randomly
        self._b = b
        self._a = a                  # Controls the transition from exploration to exploitation
        self._a_step = a_step        # Amount to reduce 'a' by in each step
        self._maximize = maximize
        self._best_solutions = []    # History of the best solution found in each iteration
        self.convergence_log = []    # Track convergence details
        
    def get_solutions(self):
        """Returns the current population of solutions (whales)."""
        return self._sols
                                                                  
    def optimize(self):
        """
        The main optimization step (one iteration/generation).
        The population updates its positions based on the current best solution 
        and the value of 'A' (derived from 'a').
        """
        # Evaluate and find the current best solution (leader)
        ranked_sol = self._rank_solutions()
        best_sol = ranked_sol[0]
        
        new_sols = [best_sol] # The best solution remains in the new population
                                                                 
        # Update the position of every other whale
        for s in ranked_sol[1:]:
            # 50/50 chance to either attack/encircle OR use the spiral movement
            if np.random.uniform(0.0, 1.0) > 0.5: 
                # Encircling or Searching for prey
                A = self._compute_A() # Coefficient vector A
                norm_A = np.linalg.norm(A) # Magnitude of A
                
                # If |A| < 1: Encircling Prey (Exploitation)
                if norm_A < 1.0:
                    new_s = self._encircle(s, best_sol, A)
                # If |A| >= 1: Search Prey (Exploration)
                else: 
                    # Select a random solution from the current population
                    random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                    new_s = self._search(s, random_sol, A)
            else:
                # Bubble-net Attacking (Exploitation, alternative to encircling)
                new_s = self._attack(s, best_sol)
                
            # Ensure the new solution is within the defined boundaries
            new_sols.append(self._constrain_solution(new_s))

        # Update the population and the 'a' parameter for the next iteration
        self._sols = np.stack(new_sols)
        self._a -= self._a_step # Linearly decrease 'a'
        
        # Log convergence data
        current_best = self._best_solutions[-1]
        self.convergence_log.append({
            'iteration': len(self._best_solutions) - 1,
            'best_cost': current_best[0],
            'best_x': current_best[1][0], 
            'best_y': current_best[1][1]
        })

    def _init_solutions(self, nsols):
        """Initializes solutions (whales) uniformly randomly within the defined constraints."""
        sols = []
        # For each dimension (constraint), generate random values
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))
                                                                            
        # Stack the results to get an array of [nsols, dimensions]
        sols = np.stack(sols, axis=-1) 
        return sols

    def _constrain_solution(self, sol):
        """Ensures solution values are valid with respect to their boundary constraints."""
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            # Check lower bound
            if c[0] > s:
                s = c[0]
            # Check upper bound
            elif c[1] < s:
                s = c[1]
            constrain_s.append(s)
        return constrain_s

    def _rank_solutions(self):
        """Evaluates and sorts solutions (whales) to find the current best (leader)."""
        # Calculate the fitness/cost for all current solutions
        fitness = self._opt_func(self.graph, self._sols[:, 0], self._sols[:, 1]) 
        
        # Pair fitness with its corresponding solution
        sol_fitness = [(f, s) for f, s in zip(fitness, self._sols)]
   
        # Sort solutions based on fitness (min/max depending on self._maximize)
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self._maximize))
        
        # Store the current best solution (for convergence history)
        self._best_solutions.append(ranked_sol[0]) 

        # Return only the solutions, sorted by rank
        return [ s[1] for s in ranked_sol] 

    def print_best_solutions(self):
        """Prints the history of the best solution found and the overall best solution."""
        print('generation best solution history')
        print('([fitness], [solution])')
        for s in self._best_solutions:
            print(s)
        print('\n')
        print('best solution')
        print('([fitness], [solution])')
        # Find the overall best solution from the history
        print(sorted(self._best_solutions, key=lambda x:x[0], reverse=self._maximize)[0])

    def _compute_A(self):
        """Compute the coefficient vector A (controls exploration/exploitation balance)."""
        # r is a random vector in [0, 1]
        r = np.random.uniform(0.0, 1.0, size=2)
        # The value of 'A' shrinks as 'a' shrinks over generations
        return (2.0*np.multiply(self._a, r))-self._a

    def _compute_C(self):
        """Compute the coefficient vector C (emphasizes or de-emphasizes the target/leader)."""
        # C is a random vector in [0, 2]
        return 2.0*np.random.uniform(0.0, 1.0, size=2)
                                                                 
    def _encircle(self, sol, best_sol, A):
        """Encircling Prey (Exploitation phase, uses A for movement toward best_sol)."""
        # D is the distance to the target (best_sol)
        D = self._encircle_D(sol, best_sol)
        # New position: X(t+1) = X*(t) - A * D
        return best_sol - np.multiply(A, D)
                                                                 
    def _encircle_D(self, sol, best_sol):
        """Calculate the distance D for the encircling movement: D = |C * X*(t) - X(t)|."""
        C = self._compute_C()
        # np.linalg.norm calculates the Euclidean distance/magnitude
        D = np.linalg.norm(np.multiply(C, best_sol)  - sol)
        return D

    def _search(self, sol, rand_sol, A):
        """Search Prey (Exploration phase, uses A for movement toward a random solution)."""
        # D is the distance to a random solution (rand_sol)
        D = self._search_D(sol, rand_sol)
        # New position: X(t+1) = X_rand - A * D
        return rand_sol - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
        """Calculate the distance D for the search movement: D = |C * X_rand - X(t)|."""
        C = self._compute_C()
        return np.linalg.norm(np.multiply(C, rand_sol) - sol)    

    def _attack(self, sol, best_sol):
        """Bubble-net Attacking (Exploitation phase, uses a spiral movement toward best_sol)."""
        # D is the distance between the whale and the best solution
        D = np.linalg.norm(best_sol - sol)
        # L is a random vector in [-1, 1], defines spiral shape
        L = np.random.uniform(-1.0, 1.0, size=2)
        
        # Spiral movement: D' * e^(b*L) * cos(2*pi*L)
        # D' is the distance D
        spiral_movement = np.multiply(np.multiply(D,np.exp(self._b*L)), np.cos(2.0*np.pi*L))
        
        # New position: X(t+1) = X*(t) + Spiral_Movement
        return spiral_movement + best_sol