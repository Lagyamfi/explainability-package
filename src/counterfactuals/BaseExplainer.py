import abc

from counterfactuals import utils


class BaseExplainer(abc.ABC):
    """Base class for counterfactual modeling and explanation."""

    def __init__(self, model, data_x, data_y, name=None):
        self.ml_model = model
        # self.dataset
        self.name = name
        self.explainer_data = None
        self.explainer_model = None

    @abc.abstractmethod
    def _prep_data(self, data_x, data_y):
        data_dice = utils.prep_data_for_dice(data_x, data_y)
        return data_dice

    @abc.abstractmethod
    def get_query(self):
        ...

    @abc.abstractmethod
    def get_counterfactuals(self):
        ...

    @abc.abstractmethod
    def load_explainer(self):
        ...

    @abc.abstractmethod
    def save_explainer(self):
        ...
