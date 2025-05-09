import random

class TSPGeneticAlgorithm:
    def __init__(self, distance_matrix, population_size, crossover_prob, mutation_prob, generations, selection_method="torneo", crossover_method="OX"):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations
        self.selection_method = selection_method
        self.crossover_method = crossover_method
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
        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]

        if self.selection_method == "ruleta":
            total_fitness = sum(fitness_scores)
            probabilities = [f / total_fitness for f in fitness_scores]
            parents = random.choices(self.population, weights=probabilities, k=self.population_size)
            return parents

        elif self.selection_method == "torneo":
            tournament_size = 3  # lo podés hacer configurable después
            parents = []
            for _ in range(self.population_size):
                tournament = random.sample(self.population, tournament_size)
                best = max(tournament, key=lambda ind: self.evaluate_fitness(ind))
                parents.append(best)
            return parents

        else:
            raise ValueError("Método de selección no reconocido")

    def crossover(self, parent1, parent2):
        if self.crossover_method == "OX":
            return self.order_crossover(parent1, parent2)
        elif self.crossover_method == "PMX":
            return self.pmx_crossover(parent1, parent2)
        else:
            raise ValueError("Método de cruzamiento no reconocido")

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        child1 = [None] * size
        child2 = [None] * size

        # Elegimos dos puntos de corte al azar
        start, end = sorted(random.sample(range(size), 2))

        # Copiamos el segmento entre start y end de cada padre
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

    # Completamos el resto del hijo desde el otro padre, en orden y sin duplicar
        def fill_child(child, parent):
            current_pos = end % size
            for gene in parent:
                if gene not in child:
                    child[current_pos] = gene
                    current_pos = (current_pos + 1) % size
            return child

        child1 = fill_child(child1, parent2)
        child2 = fill_child(child2, parent1)

        return child1, child2

    def pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        child1 = [None] * size
        child2 = [None] * size

        # 1. Elegir dos puntos de corte
        start, end = sorted(random.sample(range(size), 2))

        # 2. Copiar el segmento intermedio directamente
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # 3. Mapeo cruzado (usamos diccionarios)
        mapping1 = {parent2[i]: parent1[i] for i in range(start, end)}
        mapping2 = {parent1[i]: parent2[i] for i in range(start, end)}

        def fill_pmx(child, parent, mapping):
            for i in range(size):
                if i >= start and i < end:
                    continue  # ya copiado
                gene = parent[i]
                while gene in child:
                    gene = mapping.get(gene, gene)
                child[i] = gene
            return child

        child1 = fill_pmx(child1, parent2, mapping1)
        child2 = fill_pmx(child2, parent1, mapping2)

        return child1, child2

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
