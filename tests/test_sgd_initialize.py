import pytest
from microkeras.optimizers import SGD

def test_sgd_initialize():
    print()
    print("SGD initialization test:")
    
    # Test with default learning rate
    sgd_default = SGD()
    print(f"Default learning rate: {sgd_default.learning_rate}")
    assert sgd_default.learning_rate == 0.01, "Default learning rate should be 0.01"
    
    # Test with custom learning rate
    custom_lr = 0.001
    sgd_custom = SGD(learning_rate=custom_lr)
    print(f"Custom learning rate: {sgd_custom.learning_rate}")
    assert sgd_custom.learning_rate == custom_lr, f"Learning rate should be {custom_lr}"
    
    print("SGD initialization test passed!")
