from pathlib import Path
import tempfile

import pytest
import pandas as pd
import numpy as np
import torch

from counterfactuals import Model, Data
from counterfactuals.PytorchModel import PytorchModel
from counterfactuals.TensorflowModel import TensorflowModel
from counterfactuals.SklearnModel import SklearnModel


class TestInitialize:
    """Test the Model class"""

    def test_initialize_Model_empty(self):
        """Test the Model class"""
        model = Model.Model()
        assert model.backend == "pytorch"
        assert model.name is None

    def test_load(self):
        """Test the _load method"""
        model = Model.Model(name="test")
        model.set_up(10, 10, 10, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.pt"
            model.save(model_path)
            model_loaded = Model.Model()
            model_loaded.load(model_path)
        assert model_loaded.backend == "pytorch"

    def test_load_invalid_model(self):
        """Test the _load method with invalid model"""
        invalid_backend = "tensorflow"
        valid_backend = "pytorch"
        invalid_model = Model.Model(backend=invalid_backend)
        model = Model.Model(backend=valid_backend)
        with pytest.raises(ValueError) as info:
            model.load(invalid_model)
        expected = f"Model must be a {valid_backend} model"
        assert expected in str(info.value)

    def test_load_invalid_path(self):
        """Test the _load method with invalid path"""
        model = Model.Model()
        source = "invalid_path"
        with pytest.raises(FileNotFoundError) as info:
            model.load(source)
        expected = f"File {source!r} does not exist"
        assert expected in str(info.value)

    @pytest.mark.parametrize(
        "backend, class_name",
        [
            ("pytorch", "PytorchModel"),
            ("tensorflow", "TensorflowModel"),
            ("sklearn", "SklearnModel"),
        ],
    )
    def test_decide_implementation(self, backend, class_name):
        """Test getting the right backend implementation method
        Given a backend in string format,
        when the model is initialized,
        the correct backend implementation is returned confirmed through the class name
        """
        model = Model.Model(backend=backend)
        assert isinstance(model._model, eval(class_name))

    def test_invalid_backend(self):
        """Test invalid backend value"""
        backend = "invalid"
        with pytest.raises(ValueError) as info:
            Model.Model(backend=backend)
        expected = f"Invalid Backend: {backend!r} not supported"
        assert expected in str(info.value)

    def test_pytorch_setup(self, get_model):
        """Test the _setup method with valid data"""
        input_dim = 10
        output_dim = 2
        hidden_layers = [10, 10]
        model_type = "MLP"
        model = get_model
        model.set_up(input_dim, *hidden_layers, output_dim, model_type=model_type)
        assert model._model is not None
        assert isinstance(model._model, PytorchModel)
        # assert isinstance(model._model, torch.nn.Module)
        assert (
            repr(model._model)
            == f"Model(backend=pytorch, name=test, model={model_type})"
        )

    def test_setup_invalid(self, get_model):
        """Test the _setup method with invalid data"""
        input_dim = 10
        output_dim = 2
        hidden_layers = [10, 10]
        model_type = "invalid"
        model = get_model
        with pytest.raises(AssertionError) as info:
            model.set_up(input_dim, *hidden_layers, output_dim, model_type=model_type)
        expected = f"Invalid model type : {model_type!r}, must be MLP or CNN"
        assert expected in str(info.value)

    def test_save(self, get_model):
        """Test the save method"""
        model = get_model
        model.set_up(10, 10, 10, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model.save(model_path)
            assert model_path.exists()

    def test_save_invalid(self):
        """Test the save method when model not set up"""
        model = Model.Model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            with pytest.raises(ValueError) as info:
                model.save(model_path)
            expected = "No model set up or loaded"
            assert expected in str(info.value)


class TestTrain:
    """Test the train method"""

    def test_failed_train(self, get_model):
        """Test the train method when model not set up"""
        model = Model.Model()
        with pytest.raises(ValueError) as info:
            model.trainer()
        expected = "No model set up or loaded"
        assert expected in str(info.value)
        assert model._is_trained is False

        """Test the train method with no data"""
        model = get_model
        model.set_up(10, 10, 10, 2)
        with pytest.raises(ValueError) as info:
            model.trainer()
        expected = "No training data provided"
        assert expected in str(info.value)
        assert model._is_trained is False
        assert model._model._trained is False

    def test_train(self, get_data, set_model):
        """Test the training method"""
        data = get_data
        model = set_model
        assert model._is_trained is False
        model.trainer(*data.train_data, epochs=1)
        assert model._model._model is not None
        assert model._is_trained is True
        assert model._model._trained is True
        assert model._model._model.state_dict() is not None

    def test_model_trained(self, get_data, set_model):
        """Test if model is trained already"""
        data = get_data
        model = set_model
        model.trainer(*data.train_data, epochs=1)
        assert model._is_trained is True
        with pytest.raises(ValueError) as info:
            model.trainer(*data.train_data, epochs=1)
        expected = "Model already trained"
        assert expected in str(info.value)


class TestEvaluate:
    """Test the evaluate method"""

    def test_predict(self, get_trained_model, get_data):
        """Test the predict method"""
        model = get_trained_model
        test_data = get_data.val_data[0]
        assert len(model.predict(test_data)) == len(test_data)

    def test_predict_invalid(self, get_model):
        pass

    def test_predict_model_not_trained(self, get_model, get_data):
        model = get_model
        data = get_data
        test_data = data.val_data[0]
        with pytest.raises(ValueError) as info:
            model.predict(test_data)
        expected = "No model set up or loaded"
        assert expected in str(info.value)

    def test_predictions_in_range(self, get_trained_model, get_data):
        """Test the predict method"""
        model = get_trained_model
        test_data = get_data.val_data[0]
        predictions = model.predict(test_data)
        assert all(predictions >= 0)
        assert all(predictions <= 10)

    def test_evaluate(self, get_model):
        """Test the evaluate method"""
        pass

    def test_evaluate_invalid_data(self, get_model):
        pass

    def test_evaluate_model_not_trained(self, get_model, get_data):
        model = get_model
        data = get_data
        test_data = data.val_data
        with pytest.raises(ValueError) as info:
            model.evaluate(*test_data)
        expected = "No model set up or loaded"
        assert expected in str(info.value)

    def test_evaluate_model_trained(self, get_trained_model, get_data):
        model = get_trained_model
        data = get_data
        test_data = data.val_data
        assert model.evaluate(*test_data) is not None
        evals = model.evaluate(*test_data)
        assert isinstance(evals, dict)
        assert all(i in evals for i in ["accuracy", "predictions", "loss"])
        assert len(evals["predictions"]) == len(test_data[0])
        assert evals["accuracy"] >= 0 and evals["accuracy"] <= 1
        assert evals["loss"] >= 0
        assert np.array_equal(evals["predictions"], model.predict(test_data[0]))
