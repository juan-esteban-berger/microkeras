# Running Tests

To ensure the library is functioning correctly, you can run the included tests. This guide will show you how to run all tests or specific tests for MicroKeras.

## Prerequisites

Before running the tests, make sure you have installed MicroKeras and its dependencies, including pytest. If you haven't installed pytest, you can do so using pip:

```bash
pip install pytest
```

## Running All Tests

To run all tests for MicroKeras, use the following command from the root directory of the project:

```bash
pytest -v -s tests
```

This command will run all test files in the `tests` directory and its subdirectories.

## Running Specific Tests

If you want to run a specific test file, you can specify the path to that file. For example, to run the tests for the sigmoid activation function:

```bash
pytest -v -s tests/test_sigmoid.py
```

Replace `test_sigmoid.py` with the name of the specific test file you want to run.
