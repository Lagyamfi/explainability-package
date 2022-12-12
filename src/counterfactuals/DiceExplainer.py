from counterfactuals.BaseExplainer import BaseExplainer
from counterfactuals import utils

import dice_ml
import pandas as pd

class DiceExplainer(BaseExplainer):
    """Class for the Dice explainer."""
    def __init__(self, model, data_x, data_y, name=None):
        super().__init__(model, data_x, data_y, name)
        self.data_x = data_x  # TODO: combine into proper data object
        self.data_y = data_y  # TODO: combine into proper data object
        self.explainer_data = self._prep_data()
        self._explainer_model = None
        self._predictions = None
        self._explainer_object = None

    def _prep_data(self):
        data_dice = utils.prep_data_for_dice(self.data_x, self.data_y)
        return data_dice

    def _prep_model(self, **kwargs):
        model_dice = dice_ml.Model(model=self.ml_model, backend="PYT", model_type='classifier')  # TODO: fix backend return value
        explainer = dice_ml.Dice(self.explainer_data, model_dice, method="random", **kwargs)
        self._explainer_model = explainer

    def get_query(self, predicted, expected, dataframe=False):
        if self._predictions is None:
            self._predictions = self.ml_model.predict(self.data_x).numpy()
        queries = utils.get_query(self._predictions, self.data_y, predicted=predicted, expected=expected,)
        if len(queries) == 0:
            return f"No queries found for predicted={predicted!r} and expected={expected!r}"
        if dataframe:
            return self.explainer_data.data_df.iloc[queries]
        return queries

    def get_counterfactuals(self, query_instance=None, **kwargs):
        if self._explainer_model is None:
            self._prep_model()
        dice_exp = self._explainer_model.generate_counterfactuals(
            query_instance,
            total_CFs=5,
            proximity_weight=2,
            diversity_weight=5,
            posthoc_sparsity_algorithm="binary",
            **kwargs
        )
        self._explainer_object = dice_exp

    def save_explainer(self):
        explainer = {
            "name": self.name,
            "explainer_data": self.explainer_data,
            "explainer_model": self.explainer_model,
        }
        return explainer

    def load_explainer(self):
        pass

    def __repr__(self) -> str:
        return super().__repr__()  # TODO: add explainer-specific info
