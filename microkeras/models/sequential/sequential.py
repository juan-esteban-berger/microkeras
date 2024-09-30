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

    Methods:
    add: Add a layer to the model.
    build: Build the model by initializing all layers.
    copy: Create a deep copy of the model.
    compile: Configure the model for training.
    fit: Train the model on data.
    evaluate: Evaluate the model on test data.
    predict: Generate predictions.
    save: Save the model to a file.
    load: Load a model from a file (class method).
    """
    def __init__(self, layers):
        """
        Initialize the Sequential model.

        Parameters:
        layers (list): Initial list of layers to add to the model.
        """
        initialize(self, layers)

    add = add
    build = build
    copy = copy
    compile = compile
    fit = fit
    evaluate = evaluate
    predict = predict
    save = save

    @classmethod
    def load(cls, filename):
        return load(cls, filename)
