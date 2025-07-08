##IMPORTS
import numpy as np
import tsplib95
import time

from geneticAlgorithm import GeneticAlgorithm as ga
from selection.tournamentSelection import TournamentSelection
from selection.rouletteSelection import RouletteSelection
from crossover.OXCrossover import OXCrossover
from crossover.PMXCrossover import PMXCrossover
from mutation.inversionMutation import InversionMutation
from mutation.swapMutation import SwapMutation
from survivor_selection.elitismSurvivorSelection import ElitismSurvivorSelection
from survivor_selection.steadyStateSurvivorSelection import SteadyStateSurvivorSelection

##VALORES CONFIGURABLES
POPULATION = 100
GENERATIONS = 200
TOURNAMENT_SIZE = 3 #Used in tournament parent selection
MUTATION_RATE = 0.2  # Probability of mutation
CROSSOVER_RATE = 1.0  # Probability of crossover


##FUNCIONES
def initialize_matrix(filename) -> np.array:
    problem = tsplib95.load(filename)
    matriz_costos = np.array(problem.edge_weights)

    return matriz_costos


## MAIN FUNCTION
if __name__ == "__main__":
    matrix = initialize_matrix("instances/p43.atsp")

    selection_method = TournamentSelection()  # either TournamentSelection(tournament_size) or RouletteSelection()
    crossover_method = PMXCrossover()  # either PMXCrossover() or OXCrossover()
    mutation_method = InversionMutation()  # either InversionMutation() or SwapMutation()
    survivor_selector = ElitismSurvivorSelection(0.1)  # either ElitismSurvivorSelection(elitism_rate) or SteadyStateSurvivorSelection(replacement_rate)
    
    # --- Inicializar y correr el algoritmo ---
    genetic_algorithm = ga(
        population_size=POPULATION,
        generations=GENERATIONS,
        matrix=matrix,
        selection_method=selection_method,
        crossover_method=crossover_method,
        crossover_rate=CROSSOVER_RATE,
        mutation_method=mutation_method,
        mutation_rate=MUTATION_RATE,
        survivor_selector=survivor_selector
    )

    print("\nEmpezadno ejecucion del algoritmo genético...")
    tiempo_de_inicio = time.perf_counter()

    best_route, best_cost = genetic_algorithm.run(verbose=True, log_interval=50)

    tiempo_de_finalizacion = time.perf_counter()
    tiempo_transcurrido = tiempo_de_finalizacion - tiempo_de_inicio

    print(f"\nEjecucion del algoritmo finalizada.")
    print(f"Tiempo de ejecución: {tiempo_transcurrido:.4f} segundos")

