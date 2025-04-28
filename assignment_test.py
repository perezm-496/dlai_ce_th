"""
assignment_test.py - Test module, not exactly unit tests but employs similar principles.

This module contains tests designed to verify the correctness of student 
implementations for the exercise ce_th_assignment.ipynb. The tests will:

1. Check the accuracy and functionality of the student's code.
2. Provide clear and informative feedback for any incorrect solutions, aiding 
   learners in identifying and correcting their mistakes.

Students can execute this test module within their notebooks to independently 
assess the correctness of their code.

Usage:
- Ensure this file is in the same directory as the student's implementation.
- Run the tests to see immediate feedback as described in the notebook.
"""

import numpy as np
from solution import my_perceptron_forward

def test_task1(perceptron_forward):
    # Define the sigmoid activation function to ensure accuracy
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Custom comparison function to avoid exceptions and check array similarity
    def equality_check(arr1, arr2, shape=(3,)):
        if type(arr1) != np.ndarray:
            return False
        if arr1.shape != shape:
            return False
        return np.allclose(arr1, arr2)  # Closeness check

    # Test 1: Check simple forward pass with 5 features and 3 neurons
    def test1():
        # Set a random seed for reproducibility
        np.random.seed = 2025 
        X = np.random.rand(5)
        W = np.random.rand(3, 5)
        b = np.random.rand(3)
        expected = my_perceptron_forward(X, W, b)

        # Attempt to call the student function and handle exceptions gracefully
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)  # Fallback output

        # Check if the student output matches the expected output
        if equality_check(answ, expected, expected.shape):
            print("✅ Test Case 1: 5 features, 3 neurons ... passed")
        else:
            print("❌ Test Case 1 Failed")
            print("Suggestions:")
            for sub_test in [test1_1, test1_2, test1_3]:
                suggestion = sub_test()
                if suggestion is not None:
                    print(suggestion)
                    break
            print("\nThe forward result wasn't as expected. Consider checking:")
            print("   - Order of matrix multiplication.")
            print("   - Presence of bias term.")
            print("   - Proper application of the activation function.")
            print("   - Correct use of np.dot or np.matmul for matrix product.")

    # Sub-tests to pinpoint specific errors
    # SubTest 1: Missing Activation.
    def test1_1():
        np.random.seed = 2025
        X = np.random.rand(5)
        W = np.random.rand(3, 5)
        b = np.random.rand(3)
        expected_error = np.matmul(W, X) + b  # missing activation
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)
        if equality_check(answ, expected_error, expected_error.shape):
            return "🔧 You might have forgotten to apply the activation function."
        return None
    
    # SubTest 2: Missing bias term.
    def test1_2():
        np.random.seed = 2025
        X = np.random.rand(5)
        W = np.random.rand(3, 5)
        b = np.random.rand(3)
        expected_error = sigmoid(np.matmul(W, X))  # missing bias
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)
        if equality_check(answ, expected_error, shape=expected_error.shape):
            return "🔧 Did you add the bias term?"
        return None

    # SubTest 3: Incorrect order of multiplication.
    def test1_3():
        np.random.seed = 2025
        X = np.random.rand(3)
        W = np.random.rand(3, 3)
        b = np.random.rand(3)
        expected_error = sigmoid(np.matmul(X, W) + b)  # incorrect product
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)
        if equality_check(answ, expected_error, shape=expected_error.shape):
            return "🔧 Check if you are using the correct product order W×X."
        return None

    # Test 2: More outputs than features scenario
    def test2():
        np.random.seed = 2002
        X = np.random.rand(6)
        W = np.random.rand(10, 6)
        b = np.random.rand(10)
        expected = my_perceptron_forward(X, W, b)
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(10)
        if equality_check(answ, expected, shape=expected.shape):
            print("✅ Test Case 2: 6 features, 10 neurons ... passed")
        else:
            print("❌ Test Case 2 Failed")
            print("\nResult not as expected when creating 10 neurons for 6 inputs. Check:")
            print("   - Order of matrix multiplication.")
            print("   - Presence of bias term.")
            print("   - Proper application of the activation function.")
            print("   - Correct use of np.dot or np.matmul.")

    # Test 3: Validate custom activation functions functionality
    def test3():
        np.random.seed = 2021
        X = np.random.rand(6)
        W = np.random.rand(10, 6)
        b = np.random.rand(10)
        # Example custom activation function (ReLU)
        g = lambda x: np.maximum(x, 0.0) 

        expected = my_perceptron_forward(X, W, b, activation=g)
        try:
            answ = perceptron_forward(X, W, b, activation=g)
        except Exception:
            answ = np.zeros(10)
        if equality_check(answ, expected, shape=expected.shape):
            print("✅ Test Case 3: custom activation function ... passed")
        else:
            print("❌ Test Case 3 Failed")
            print("\nExpected result not met with custom activation function."
                  " Check if the 'activation' parameter is used instead of the default 'sigmoid'.")

    # Test 4: Evaluate boundary cases (single input, single neuron)
    def test4():
        np.random.seed = 2000
        X = np.random.rand(1)
        W = np.random.rand(1, 1)
        b = np.random.rand(1)

        expected = my_perceptron_forward(X, W, b, activation=sigmoid)
        try:
            answ = perceptron_forward(X, W, b, activation=sigmoid)
        except Exception as e:
            answ = np.zeros(10)
        if equality_check(answ, expected, shape=expected.shape):
            print("✅ Test Case 4: Single input, single output ... passed")
        else:
            print("❌ Test Case 4 Failed")
            print("\nUnexpected result with single input (X.shape=(1,)) and single neuron (W.shape=(1,1)). Check:")
            print("   - Order of matrix multiplication.")
            print("   - Presence of bias term.")
            print("   - Proper application of activation function.")
            print("   - Correct use of np.dot or np.matmul.")

    # Test 5: Zero weights, zero bias
    def test5():
        np.random.seed = 2000
        X = np.random.rand(10)
        W = np.zeros((1, 10))
        b = np.zeros(1)

        expected = my_perceptron_forward(X, W, b, activation=sigmoid)
        try:
            answ = perceptron_forward(X, W, b, activation=sigmoid)
        except Exception as e:
            answ = np.zeros(10)
        if equality_check(answ, expected, shape=expected.shape):
            print("✅ Test Case 5: zero weights, zero bias ... passed")
        else:
            print("❌ Test Case 5 Failed")
            print("\nUnexpected result with zero weights and bias. Consider reviewing:")
            print("   - Order of matrix multiplication.")
            print("   - Presence of bias term.")
            print("   - Proper application of activation function.")
            print("   - Correct use of np.dot or np.matmul.")

    # Run all test cases
    test1()
    test2()
    test3()
    test4()
    test5()

    # End test