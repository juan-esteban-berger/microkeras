import pytest
import numpy as np
from microkeras.optimizers import select_batches

def test_sgd_select_batches():
    print()
    print("Select batches function test:")

    # Create sample input data and labels
    X_train = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [0.6, 0.7, 0.8, 0.9, 1.0],
                        [1.1, 1.2, 1.3, 1.4, 1.5]])
    Y_train = np.array([[1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0]])

    print("Input data shape:", X_train.shape)
    print("Labels shape:", Y_train.shape)

    # Test parameters
    batch_size = 3

    # Select batches
    X_batch, Y_batch = select_batches(X_train, Y_train, batch_size)

    print("\nSelected batch shapes:")
    print("X_batch shape:", X_batch.shape)
    print("Y_batch shape:", Y_batch.shape)

    # Check shapes
    assert X_batch.shape == (3, batch_size), f"Expected X_batch shape (3, {batch_size}), but got {X_batch.shape}"
    assert Y_batch.shape == (2, batch_size), f"Expected Y_batch shape (2, {batch_size}), but got {Y_batch.shape}"

    # Check that selected indices are unique and correspond for both X and Y
    for i in range(batch_size):
        x_col = X_batch[:, i]
        y_col = Y_batch[:, i]
        
        matching_indices = np.where((X_train == x_col.reshape(-1, 1)).all(axis=0) & 
                                    (Y_train == y_col.reshape(-1, 1)).all(axis=0))[0]
        
        assert len(matching_indices) == 1, f"Column {i} in batch not found exactly once in original data"

    # Test with different batch sizes
    for test_batch_size in [1, 2, 4, 5]:
        X_test_batch, Y_test_batch = select_batches(X_train, Y_train, test_batch_size)
        assert X_test_batch.shape == (3, test_batch_size), f"Incorrect X shape for batch size {test_batch_size}"
        assert Y_test_batch.shape == (2, test_batch_size), f"Incorrect Y shape for batch size {test_batch_size}"

    print("\nSelect batches function test passed!")
