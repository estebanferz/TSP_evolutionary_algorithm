import random

class TSPGeneticAlgorithm:
    def __init__(self, distance_matrix, population_size, crossover_prob, mutation_prob, generations):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations
        self.population = []

    def initialize_population(self):
        # Genera soluciones aleatorias
        for _ in range(self.population_size):
            individual = list(range(self.num_cities))
            random.shuffle(individual)
            self.population.append(individual)

    def evaluate_fitness(self, individual):
        # Calcula la distancia total del recorrido (fitness)
        distance = 0
        for i in range(len(individual)):
            from_city = individual[i]
            to_city = individual[(i + 1) % len(individual)]
            distance += self.distance_matrix[from_city][to_city]
        return 1 / distance  # fitness inversamente proporcional a la distancia

    def select_parents(self):
        # Placeholder: se definirá más adelante (ruleta o torneo)
        pass

    def crossover(self, parent1, parent2):
        # Placeholder: se definirá más adelante (OX, PMX, etc.)
        pass

    def mutate(self, individual):
        # Placeholder: se definirá más adelante (swap, inversion, etc.)
        pass

    def local_search(self, individual):
        # Placeholder: mejora por búsqueda local (adaptación baldwiniana)
        return individual

    def run(self):
        self.initialize_population()

        for generation in range(self.generations):
            # Evaluar fitness
            fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]

            # Seleccionar padres
            parents = self.select_parents()

            # Generar nueva población
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)

                if random.random() < self.crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                if random.random() < self.mutation_prob:
                    self.mutate(child1)
                    self.mutate(child2)

                child1 = self.local_search(child1)
                child2 = self.local_search(child2)

                new_population.extend([child1, child2])

            # Reemplazar la población
            self.population = new_population[:self.population_size]

        # Retornar el mejor individuo
        best_individual = min(self.population, key=lambda ind: 1 / self.evaluate_fitness(ind))
        return best_individual, 1 / self.evaluate_fitness(best_individual)
