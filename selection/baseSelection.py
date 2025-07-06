import abc
import numpy as np

class BaseSelection(abc.ABC):

    @abc.abstractmethod
    def selection(self, population: np.ndarray, fitness:np.ndarray, parents: int) -> np.ndarray:
        # Perform selection of parents from the population based on their fitness.
        pass
