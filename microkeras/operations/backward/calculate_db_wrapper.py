from microkeras.operations.backward.calculate_db import calculate_db

def calculate_db_wrapper(model, i, m):
    return calculate_db(model.layers[i].dZ, m)
