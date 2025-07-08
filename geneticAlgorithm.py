import numpy as np
import typing
import crossover.baseCrossover as BaseCrossover
import selection.baseSelection as BaseSelection
import mutation.baseMutation as BaseMutation

class GeneticAlgorithm:
    def __init__(self,
                 population_size,
                 generations,
                 selection_method,
                 crossover_method,
                 crossover_rate,
                 mutation_method,
                 mutation_rate,
                 survivor_selector,
                 matrix):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.survivor_selector = survivor_selector
        self.matrix = matrix

        self.num_cities = matrix.shape[0]  
        self.population = np.empty((population_size, self.num_cities), dtype=int)
        self.fitness = np.zeros(population_size)
        self.cost = np.zeros(population_size)
        self.best_individual = None
        self.best_fitness = -np.inf
        self.best_cost = np.inf
        self.local_search_threshold = 0.01  # Threshold for local search improvement
        self.rng = np.random.default_rng()

    def initialize_population(self):
        # Initialize the population with random individuals
        for i in range(self.population_size):
            self.population[i] = self.rng.permutation(self.num_cities) 
        self.evaluate_fitness()

    def evaluate_fitness(self, candidates: typing.Union[np.ndarray, None] = None) -> typing.Union[tuple[np.ndarray, np.ndarray], None]:
        # Evaluate the fitness of each individual in the population
        local_population = candidates if candidates is not None else self.population
        num_candidates = local_population.shape[0]
        total_costs = np.empty(num_candidates)
        fitness = np.empty(num_candidates)

        for i in range(num_candidates):
            cost = 0
            route = local_population[i]
            for j in range(self.num_cities):
                # Calculate the cost of the route, a and b are cities
                a = route[j]
                b = route[(j + 1) % self.num_cities]
                cost += self.matrix[a, b]
            total_costs[i] = cost
            fitness[i] = 1.0 / cost if cost != 0 else np.inf

        if candidates is None:
            self.cost = total_costs
            self.fitness = fitness
            return None
        else:
            return total_costs, fitness

    def local_search(self, individual):
        #Applies 2-opt local search (Baldwinian adaptation)
        best = individual.copy()
        improved = True

        def route_cost(route):
            # Vectorized cost calculation
            shifted = np.roll(route, -1)
            return self.matrix[route, shifted].sum()

        best_cost = route_cost(best)

        while improved:
            improved = False
            for i in range(1, self.num_cities - 1):
                for j in range(i + 1, self.num_cities):
                    if j - i == 1:
                        continue  # Skip adjacent cities
                    new_route = best.copy()
                    new_route[i:j] = best[i:j][::-1]  # Reverse the segment
                    new_cost = route_cost(new_route)
                    if new_cost < best_cost * (1 - self.local_search_threshold):
                        best = new_route
                        best_cost = new_cost
                        improved = True
        return best



    def run(self, verbose: bool = True, log_interval: int = 50) -> tuple[np.ndarray, float]:

        # Vectorized fitness evaluation
        def calculate_costs(population):
            shifted = np.roll(population, -1, axis=1)
            return self.matrix[population, shifted].sum(axis=1)

        # === Initialization ===
        if self.population_size == 0:
            print("Warning: Cannot run with population size 0")
            return None, np.inf
        
        self.initialize_population()
        
        # Track best solution
        best_idx = np.argmax(self.fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.best_cost = self.cost[best_idx]
        
        if verbose:
            print(f"Generation 0: Best Cost = {self.best_cost:.2f}")
        
        # === Main Evolution Loop ===
        for gen in range(1, self.generations + 1):
            # 1. Parent Selection (handles odd population sizes)
            parent_count = self.population_size if self.population_size % 2 == 0 else self.population_size + 1

            parents = self.selection_method.select(self.population, self.fitness, parent_count)
            
            # 2. Crossover with probability

            offspring = self.crossover_method.crossover(parents, self.crossover_rate)
            
            # 3. Mutation with probability and local search
            if offspring.size > 0:  # Only mutate if we have offspring

                offspring = self.mutation_method.mutate(offspring, self.mutation_rate)

                # Apply local search to each offspring individual

                elite_count = max(1, len(offspring) // 10)
                elite_indices = np.argpartition(calculate_costs(offspring), elite_count)[:elite_count]
                for idx in elite_indices:
                    offspring[idx] = self.local_search(offspring[idx])
            
            # 4. Evaluate offspring
            if offspring.size > 0:
                offspring_costs, offspring_fitness = self.evaluate_fitness(offspring)
            else:
                offspring_costs, offspring_fitness = np.array([]), np.array([])
            
            if offspring.size > 0:
                self.population, self.fitness = self.survivor_selector.select_survivors(
                    population=self.population,
                    fitness=self.fitness,
                    offspring=offspring,
                    offspring_fitness=offspring_fitness
                )
                # Update costs (since we're tracking them separately)
                self.cost = calculate_costs(self.population)
            
            # 6. Update best solution
            current_best_idx = np.argmax(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = self.population[current_best_idx].copy()
                self.best_cost = self.cost[current_best_idx]
            
            # 7. Logging
            if verbose and (gen % log_interval == 0 or gen == self.generations):
                print(f"Generation {gen}: Best Cost = {self.best_cost:.2f} | Pop Size = {len(self.population)}")
        
        # Final report
        if verbose:
            print("\n=== Optimization Complete ===")
            print(f"Best Solution: {self.best_individual}")
            print(f"Best Cost: {self.best_cost:.2f}")
            print(f"Best Fitness: {self.best_fitness:.6f}")
        
        return self.best_individual, self.best_cost