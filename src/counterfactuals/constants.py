# constants used by the counterfactuals package

class Backend:
    """The backend to use for the model"""
    sklearn = "sklearn"
    tensorflow = "tensorflow"
    pytorch = "pytorch"

    ALL = [sklearn, tensorflow, pytorch]
