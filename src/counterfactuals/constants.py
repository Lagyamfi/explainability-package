# constants used by the counterfactuals package
import enum

class Backend(enum.Enum):
    """The backend to use for the model"""
    sklearn = "sklearn"
    tensorflow = "tensorflow"
    pytorch = "pytorch"

    ALL = [sklearn, tensorflow, pytorch]
