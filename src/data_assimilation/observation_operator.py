from abc import abstractmethod, ABC



class BaseObservationOperator(ABC):

    @abstractmethod
    def get_observations(self, state):
        pass