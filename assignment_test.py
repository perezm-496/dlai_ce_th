
"""
assignment_test.py - Test module, not really unitest although it makes unitest.
My decition to change it so I can give better feedback without the traceback.

This module contains unit tests designed to verify the correctness of student 
implementations for the exercise ce_th_assignment.ipynb. The tests will:

1. Check the accuracy and functionality of the student's code.
2. Provide clear and informative feedback for any incorrect solutions, helping 
   learners identify and correct their mistakes.

Students can execute this test module within their notebooks to independently 
assess the correctness of their code.

Usage:
- Ensure this file is in the same directory as the student's implementation.
- Run the tests to see immediate results and feedback as described in the notebook.
"""
import numpy as np
from solution import my_perceptron_forward

def test_task1(perceptron_forward):
    # To prevent a changed simoid to produce bad results
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    # Test 1: Check the activation function is aplied
    def activation_applied():
        X = np.array([1.0, 1.0, 1.0])
        W = 10*np.identity(3)
        b = np.zeros((1, 3))
        expected = my_perceptron_forward(X, W, b)
        output = perceptron_forward(X, W, b)
        fail_msg = (
            "\n[❌Activation applied sample test FAILED]\n"
            "The forward result is not the expected,\n"
            "it seeems like you fail to apply the activation function.\n"
        )
        if np.allclose(output, expected):
            print("✅ Test: Activation function applied ... passed ")
        else:
            print(fail_msg)
    
    # Test 2: Check the bias is added
    def bias_added():
        X = np.array([1.0, 1.0, 1.0])
        W = np.zeros((3,3))
        b = 10*np.ones((1, 3))
        expected = my_perceptron_forward(X, W, b)
        output = perceptron_forward(X, W, b)
        fail_msg = (
            "\n[❌Bias added sample test FAILED]\n"
            "The forward result is not the expected,\n"
            "it seeems like you might not have added the bias.\n"
        )
        if np.allclose(output, expected):
            print("✅ Test: Bias added ... passed ")
        else:
            print(fail_msg)
  
    # Test 3: correctness_sample using the provided example
    def correctness_sample():
        X = np.array([0.9, 0.7, 0.3])
        W = np.array([
            [1, 0, 1],
            [-1, 0, -1],
            [0.1, 1, 0.1]])
        b = np.array([-1.0, 0.1, 0.001])
        expected_output = my_perceptron_forward(X, W, b, activation=sigmoid)
        output = perceptron_forward(X, W, b)
        fail_msg = (
            "\n[❌Correctness Sample test FAILED]\n"
            "The forward result is not the expected, you might want to check the following:\n"
            "   - The order of matrix multiplication.\n"
            "   - Is the bias term present?\n"
            "   - The activation function is correctly applied.\n"
            "    - Did you use the np.dot to compute the matrix product?\n"
        )
        if np.allclose(output, expected_output):
            print("✅ Test: Basic pass forward ... passed ")
        else:  
            print(fail_msg)
        

    # Test 4: correctness_3_1
    def correctness_3_1():
        X = np.array([1.0, 2.0, 3.0])
        W = np.array([
            [1, 4, 7]
            ])
        b = np.array([0.1,])
        try:
            expected_output = sigmoid(np.dot(W, X) + b)
            output = perceptron_forward(X, W, b)
        except ValueError:
            expected_output = np.zeros((2,2))
            output = np.zeros((1,2))
        fail_msg = (
            "\n[❌Three inputs single output test FAILED]\n"
            "The forward result is not the expected.\n"
            "   - Check the order of matrix multiplication.\n"            
            "   - Recall that although np.matmul and np.dot are similar have small similarities\n"
            "     specially when using 1D arrays.\n"
        )
        if output.shape == expected_output.shape:
            print("✅ Test: Three inputs single output test ... passed ")
        else:
            print(fail_msg)
            

    # Test 3: correctness_2_3
    def correctness_2_3():
        X = np.array([1.0, 2.0])
        W = np.array([
            [1, 2],
            [3, 4],
            [5, 6]])
        b = np.array([0.1, 0.2, 0.3])
        expected_output = sigmoid(np.dot(W, X) + b)
        output = perceptron_forward(X, W, b)
        assert np.allclose(output, expected_output), "Failed correctness_2_3 test"
        print("correctness_2_3 test passed")

    # Test 4: zero weights
    def zero_weights():
        X = np.array([1.0, 2.0, 3.0])
        W = np.zeros((3, 3))
        b = np.array([0, 0, 0])
        expected_output = sigmoid(b)
        output = perceptron_forward(X, W, b)
        assert np.allclose(output, expected_output), "Failed zero_weights test"
        print("zero_weights test passed")

    # Test 5: zero inputs
    def zero_inputs():
        X = np.zeros(3)
        W = np.identity(3)
        b = np.array([0, 0, 0])
        expected_output = sigmoid(b)
        output = perceptron_forward(X, W, b)
        assert np.allclose(output, expected_output), "Failed zero_inputs test"
        print("zero_inputs test passed")

    # Test 6: different activation
    def different_activation():
        def relu(z):
            return np.maximum(0, z)
          
        X = np.array([1, 2])
        W = np.array([
            [1, -1],
            [-1, 1],
            [1, 1]])
        b = np.array([-1, 1, 0])
        expected_output = relu(np.dot(W, X) + b)
        output = perceptron_forward(X, W, b, activation=relu)
        assert np.allclose(output, expected_output), "Failed different_activation test"
        print("different_activation test passed")

    # Test 7: bias well managed
    def bias_well_managed():
        X = np.array([1.0, 0.0, 0.0])
        W = np.identity(3)
        b = np.array([1.0, 1.0, 1.0])
        expected_output = sigmoid(X + b)
        output = perceptron_forward(X, W, b)

        assert np.allclose(output, expected_output), "Failed bias_well_managed test"
        print("bias_well_managed test passed")

    # Test 8: weights correctly applied
    def weights_correctly_applied():
        X = np.array([0.0, 1.0, 2.0])
        W = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]])
        b = np.array([0.0, 0.0, 0.0])
        expected_output = sigmoid(np.dot(W, X) + b)
        output = perceptron_forward(X, W, b)
        assert np.allclose(output, expected_output), "Failed weights_correctly_applied test"
        print("weights_correctly_applied test passed")

    # Run all tests
    activation_applied()
    bias_added()
    correctness_sample()
    correctness_3_1()
    correctness_2_3()
    zero_weights()
    zero_inputs()
    different_activation()
    bias_well_managed()
    weights_correctly_applied()
