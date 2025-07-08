import abc
import numpy as np

class BaseSurvivorSelection(abc.ABC):
    @abc.abstractmethod
    def select_survivors(self, population: np.ndarray, fitness: np.ndarray, offspring: np.ndarray, offspring_fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Selects survivors from the current population and offspring.
        pass