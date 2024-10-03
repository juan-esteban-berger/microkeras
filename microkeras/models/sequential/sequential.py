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
    The Sequential model is a linear stack of layers, useful for
    straightforward neural network architectures. Layers are added via the constructor
    or through the `.add()` method.

    Attributes:
        layers (list): List of Layer instances in the model.

    Example:
        ```python
        model = Sequential([
            Dense(64, activation='relu', input_shape=(784,)),
            Dense(10, activation='softmax')
        ])
        ```
    """

    def __init__(self, layers):
        """
        Initialize the Sequential model.

        Args:
            layers (list): Initial list of Layer instances to add to the model.
        """
        initialize(self, layers)
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
        """
        Load a model from a file.

        Args:
            filename (str): Path to the file containing the saved model.

        Returns:
            Sequential: Loaded model instance.

        Example:
            ```python
            loaded_model = Sequential.load('my_model.json')
            ```
        """
        return load(cls, filename)
