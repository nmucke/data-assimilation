
from abc import abstractmethod, ABC


class BaseModelError(ABC):

    @abstractmethod
    def get_model_error(self, state_ensemble, pars_ensemble):
        pass

    @abstractmethod
    def add_model_error(self, state_ensemble, pars_ensemble):
        pass
    