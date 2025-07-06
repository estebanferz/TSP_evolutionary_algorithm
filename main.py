##IMPORTS
import numpy as np
import tsplib95
from geneticAlgorithm import GeneticAlgorithm as ga


##VALORES CONFIGURABLES
POPULATION = 100
GENERATIONS = 200
TOURNAMENT_SIZE = 3 #Used in tournament parent selection
CROSSOVER_METHOD = "PMX"  # PMX, OX
SELECTION_METHOD = "tournament"  # tournament, roulette
MUTATION_METHOD = "swap"  # swap, inversion
MUTATION_RATE = 0.1  # Probability of mutation


##FUNCIONES
def initialize_matrix(filename) -> np.array:
    problem = tsplib95.load(filename)
    matriz_costos = np.array(problem.edge_weights)

    return matriz_costos


## MAIN FUNCTION
if __name__ == "__main__":
    matrix = initialize_matrix("instances/p43.atsp")


    ## run the genetic algorithm
    ga.run()


    print("Cost matrix initialized:")
    print(matrix)
    print("Genetic Algorithm Simulation")

