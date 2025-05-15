from abc import ABC, abstractmethod

class CoordinateStrategy(ABC):
    """Base class for all coordinate strategies in benchmark-ev-peak-shaving.

    Parameters
    ----------
    random_state : int or RandomState instance, optional (default=None)
        Controls the randomness of the Strategy.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state
    
    @abstractmethod
    def get_coordinate_SoC_charging(self, *args, **kwargs):
        """
        Determines the coordinated charging and SoC
        """
        raise NotImplementedError

class AggregateCoordinateStrategy(CoordinateStrategy):
    """Basic class for all aggregate stratgies in benchmark-wv-peak-shaving.
    """

    def __init__(self, random_state=None):
        super.__init__(random_state=random_state)

class DistributedCoordinateStrategy(CoordinateStrategy):
    def __init__(self, random_state=None):
        super().__init__(random_state)
    
    