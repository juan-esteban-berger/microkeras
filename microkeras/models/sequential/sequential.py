from .initialize import initialize
from .add import add
from .build import build
from .copy import copy
from .compile import compile
from .fit import fit
from .evaluate import evaluate
from .predict import predict

class Sequential:
    def __init__(self, layers):
        initialize(self, layers)

    add = add
    build = build
    copy = copy
    compile = compile
    fit = fit
    evaluate = evaluate
    predict = predict
