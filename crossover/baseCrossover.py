import abc
import numpy as np

class BaseCrossover(abc.ABC):

    @abc.abstractmethod
    def crossover(self, population: np.ndarray, fitness: np.ndarray, parents: int) -> np.ndarray:
        # Perform crossover between two parents to create offspring
        pass