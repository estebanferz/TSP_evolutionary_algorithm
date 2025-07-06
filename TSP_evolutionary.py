import random
import time

class TSPGeneticAlgorithm:
    def __init__(self,
                 distance_matrix,
                 population_size,
                 crossover_prob,
                 mutation_prob,
                 generations,
                 selection_method="torneo",
                 crossover_method="OX",
                 mutation_method="swap"):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.population = []

    random.seed(42)

    def initialize_population(self):
        self.population = []
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

        # Paso 1: seleccionar puntos de corte
        start, end = sorted(random.sample(range(size), 2))

        # Paso 2: copiar segmento intermedio
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # Paso 3: completar hijo 1
        for i in range(size):
            if child1[i] is not None:
                continue  # ya está ocupado por el segmento
            gene = parent2[i]
            mapped_gene = gene
            while mapped_gene in child1[start:end]:
                mapped_gene = parent2[parent1.index(mapped_gene)]
            child1[i] = mapped_gene

        # Paso 4: completar hijo 2
        for i in range(size):
            if child2[i] is not None:
                continue
            gene = parent1[i]
            mapped_gene = gene
            while mapped_gene in child2[start:end]:
                mapped_gene = parent1[parent2.index(mapped_gene)]
            child2[i] = mapped_gene

        # Paso 5: validación (opcional en producción)
        if len(set(child1)) != size or len(set(child2)) != size:
            print("  ⚠️ PMX falló: duplicados detectados")
            raise ValueError("PMX produjo hijos inválidos")

        return child1, child2



    def mutate(self, individual):
        if self.mutation_method == "swap":
            self.swap_mutation(individual)
        elif self.mutation_method == "inversion":
            self.inversion_mutation(individual)
        else:
            raise ValueError("Método de mutación no reconocido")

    def swap_mutation(self, individual):
        """Intercambia dos ciudades aleatorias en la permutación."""
        i, j = random.sample(range(self.num_cities), 2)
        individual[i], individual[j] = individual[j], individual[i]
    
    def inversion_mutation(self, individual):
        """Invierte el orden de un segmento aleatorio de la permutación."""
        i, j = sorted(random.sample(range(self.num_cities), 2))
        # Revertimos el segmento individual[i:j]
        individual[i:j] = individual[i:j][::-1]


    def local_search(self, individual):
        """Aplica mejora 2-opt al individuo (adaptación baldwiniana)"""
        best = individual[:]
        improved = True

        def route_cost(route):
            cost = 0
            for i in range(len(route)):
                cost += self.distance_matrix[route[i]][route[(i + 1) % self.num_cities]]
            return cost

        best_cost = route_cost(best)

        while improved:
            improved = False
            for i in range(1, self.num_cities - 1):
                for j in range(i + 1, self.num_cities):
                    if j - i == 1:
                        continue  # ciudades adyacentes, no conviene invertir
                    new_route = best[:]
                    new_route[i:j] = reversed(best[i:j])
                    new_cost = route_cost(new_route)
                    if new_cost < best_cost:
                        best = new_route
                        best_cost = new_cost
                        improved = True
            # cuando no encuentra mejoras, termina
        return best

    def run(self):
        # 1. Inicialización de población
        self.initialize_population()
        
        print(self.population)
        # Preparamos el historial de best fitness
        history = []

        # 2. Medimos el tiempo de inicio
        start_time = time.time()

        # 3. Ciclo evolutivo
        for generation in range(self.generations):

            if generation % 1 == 0:
                elapsed = time.time() - start_time
                print(f"[Gen {generation}/{self.generations}] tiempo transcurrido: {elapsed:.1f}s")

            # Evaluar fitness de la población actual                
            print("-> inicio iteración", generation)
            fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
            print("   fitness ok")
            best_fitness = max(fitness_scores)
            print("   selección ok")
            history.append(best_fitness)

            if generation > 20 and history[-1] == history[-20]:
                print("Convergencia detectada. Deteniendo...")
                break

            # Selección de padres
            parents = self.select_parents()

            # Generar nueva población
            counter = 0
            new_population = []
            while len(new_population) < self.population_size:

                counter += 1
                ##print(f"  * iteración bucle hijos #{counter}, tamaño población actual: {len(new_population)}")


                p1, p2 = random.sample(parents, 2)
                try:
                    c1, c2 = self.crossover(p1, p2)
                except Exception as e:
                    print("  !! Exception en crossover:", e)
                    # Para no trabar la ejecución, generamos copias:
                    c1, c2 = p1[:], p2[:]
                    print("  -> usé copia fallback")



                # Mutación en cada hijo
                if random.random() < self.mutation_prob:
                    self.mutate(c1)
                if random.random() < self.mutation_prob:
                    self.mutate(c2)

                # Búsqueda local (Baldwiniana)
                if random.random() < 0.3:
                    c1 = self.local_search(c1)
                if random.random() < 0.3:
                    c2 = self.local_search(c2)


                new_population.extend([c1, c2])

            # Reemplazar la población (modelo generacional)
            self.population = new_population[:self.population_size]
            print("-> fin iteración", generation)

        # 4. Medimos el tiempo final
        elapsed_time = time.time() - start_time

        # 5. Seleccionamos el mejor individuo final
        best_individual = max(self.population, key=self.evaluate_fitness)
        best_cost = 1 / self.evaluate_fitness(best_individual)

        # 6. Devolvemos toda la información útil
        return {
            "best_individual": best_individual,
            "best_cost": best_cost,
            "history": history,
            "elapsed_time": elapsed_time
        }


def read_atsp(filename):
    """
    Lee un archivo .atsp de TSPLIB y devuelve la matriz de distancias.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    dimension = None
    matrix = []
    reading_weights = False

    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            # Extraer valor después de ':' o ' '
            dimension = int(line.split()[-1])
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            reading_weights = True
            continue
        elif reading_weights:
            if line == "EOF" or len(matrix) == dimension:
                break
            # Analizar la línea de pesos
            row = [int(x) for x in line.split()]
            matrix.append(row)

    if dimension is None:
        raise ValueError("No se encontró la DIMENSION en el archivo .atsp")
    if len(matrix) != dimension:
        raise ValueError(f"Se esperaban {dimension} filas, pero se obtuvieron {len(matrix)}")

    return matrix




# Crear instancia

distance_matrix = read_atsp("instancias/p43.atsp")
print("Dimensión real:", len(distance_matrix))


ga = TSPGeneticAlgorithm(
    distance_matrix=distance_matrix,
    population_size=50,
    crossover_prob=0.9,
    mutation_prob=0.1,
    generations=200,
    selection_method="torneo",
    crossover_method="PMX",
    mutation_method="inversion"
)

# Ejecutar
result = ga.run()

# Mostrar resultados
print("Mejor ruta:", result["best_individual"])
print("Costo:", result["best_cost"])
print("Tiempo (s):", result["elapsed_time"])
print("Fitness por generación:", result["history"])


##[9, 36, 37, 21, 24, 25, 39, 41, 42, 38, 40, 26, 15, 20, 19, 18, 17, 16, 6, 12, 34, 33, 14, 13, 11, 32, 31, 29, 30, 4, 0, 35, 22, 23, 28, 27, 2, 1, 3, 8, 10, 7, 5]
##[18, 19, 15, 17, 12, 7, 6, 29, 28, 1, 36, 2, 3, 27, 30, 33, 13, 11, 10, 32, 31, 8, 9, 14, 34, 4, 0, 35, 38, 39, 41, 40, 42, 26, 24, 23, 22, 21, 25, 37, 5, 16, 20] Costo: 5676.0
##[27, 32, 31, 9, 7, 25, 21, 22, 23, 24, 28, 6, 33, 12, 13, 10, 8, 36, 35, 0, 2, 3, 4, 37, 39, 38, 42, 40, 41, 26, 15, 19, 18, 20, 16, 17, 5, 29, 30, 34, 14, 11, 1] Costo: 5738.0

