import abc
import numpy as np


class BaseMutation(abc.ABC):
    @abc.abstractmethod
    def mutate(self, parent: np.ndarray, mutation_rate: float) -> np.ndarray:
        # Perform mutation on the population based on the mutation rate
        pass