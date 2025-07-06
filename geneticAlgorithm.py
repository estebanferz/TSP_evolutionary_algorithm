import numpy as np

class GeneticAlgorithm:
    def __init__(self,
                 population_size,
                 mutation_rate,
                 crossover_rate,
                 selection_method,
                 crossover_method,
                 mutation_method,
                 matrix):
        
        self.population_size = population_size
        self.matrix = matrix
        self.cities = matrix.shape[0]  
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = np.empty((population_size, self.cities), dtype=int)
        self.fitness = np.zeros(population_size)
        self.cost = np.zeros(population_size)

        self.best_individual = None
        self.best_fitness = -np.inf
        self.best_cost = np.inf
        self.rng = np.random.default_rng()

    def initialize_population(self):
        # Initialize the population with random individuals
        for i in range(self.population_size):
            self.population[i] = self.rng.permutation(self.cities) 
            self.evaluate_fitness()
        pass

    def evaluate_fitness(self):
        # Evaluate the fitness of each individual in the population
        pass


    def run(self, generations):
        # Run the genetic algorithm for a specified number of generations
        return self.best_individual, self.best_fitness