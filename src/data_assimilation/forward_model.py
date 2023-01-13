    
from abc import abstractmethod, ABC


class BaseForwardModel(ABC):
    
    @abstractmethod
    def update_params(self, params):
        pass

    @abstractmethod
    def compute_forward_model(self, q):
        pass
