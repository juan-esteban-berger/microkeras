from .initialize import initialize
from .add import add
from .build import build
from .copy import copy
from .compile import compile
from .fit import fit
from .evaluate import evaluate
from .predict import predict
from .save import save
from .load import load

class Sequential:
    """
    Sequential groups a linear stack of layers into a model.

    The Sequential model is a linear stack of layers, useful for
    straightforward architectures. Layers are added via the constructor
    or through the .add() method.

    Attributes:
        layers (list): List of layers in the model.
    """

    def __init__(self, layers):
        """
        Initialize the Sequential model.

        Parameters:
            layers (list): Initial list of layers to add to the model.
        """
        initialize(self, layers)

    add = add
    """
    :no-index:
    """

    build = build
    """
    :no-index:
    """

    copy = copy
    """
    :no-index:
    """

    compile = compile
    """
    :no-index:
    """

    fit = fit
    """
    :no-index:
    """

    evaluate = evaluate
    """
    :no-index:
    """

    predict = predict
    """
    :no-index:
    """

    save = save
    """
    :no-index:
    """

    @classmethod
    def load(cls, filename):
        """
        Load a model from a file.

        Parameters:
            filename (str): Path to the file containing the saved model.

        Returns:
            Sequential: Loaded model instance.
        """
        return load(cls, filename)
