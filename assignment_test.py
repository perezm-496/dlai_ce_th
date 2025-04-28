
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
import traceback
from solution import my_perceptron_forward

def test_task1(perceptron_forward):
    # To prevent a changed simoid to produce bad results
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    # Test 1: Check simple pass forward 5 features, 3 neurons
    def test1():
        np.random.seed = 2025 # reproduce
        X = np.random.rand( 5)
        W = np.random.rand( 3, 5)
        b = np.random.rand(3)
        expected = my_perceptron_forward(X, W, b)
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)
        if np.allclose(expected, answ):
            print("✅ Test Case 1: 5 features, three neurons ... passed")
        else:
            print("❌ First Test Failed")
            sub_test_list = [test1_1, test1_2, test1_3]
            print("Sugestion:")
            for sub_test in sub_test_list:
                sugestion = sub_test()
                if sugestion is not None:
                    print(sugestion)
                    break
            msg = (
                "\nThe forward result is not the expected, you might want to check the following:\n"
                "   - The order of matrix multiplication.\n"
                "   - Is the bias term present?\n"
                "   - The activation function is correctly applied.\n"
                "    - Did you use the np.dot or np.matmul to compute the matrix product correctly?\n"
            )
            print(msg)


    # SubTest 3: Wrong order or product
    def test1_3():
        np.random.seed = 2025 # reproduce
        X = np.random.rand( 5)
        W = np.random.rand( 5, 3)
        b = np.random.rand(3)
        expected_error = sigmoid(np.matmul(X,W))  # bad product
        print(expected_error)
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)
        if np.allclose(expected_error, answ):
            return "🔧  Are you sure to be using the correct product order W×X?"
        else:
            return None

    # SubTest 2: Forgot bias
    def test1_2():
        np.random.seed = 2025 # reproduce
        X = np.random.rand( 5)
        W = np.random.rand( 3, 5)
        b = np.random.rand(3)
        expected_error = sigmoid(np.matmul(W, X))  # missing bias
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)
        if np.allclose(expected_error, answ):
            return "🔧 Did you add the bias term?"
        else:
            return None
    
    # SubTest 1: Missing Activation.
    def test1_1():
        np.random.seed = 2025 # reproduce
        X = np.random.rand( 5)
        W = np.random.rand( 3, 5)
        b = np.random.rand(3)
        expected_error = np.matmul(W, X) + b # missing activation
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(3)
        if np.allclose(expected_error, answ):
            return "🔧 Seems like you might have forgoten to apply the activation function."
        else:
            return None

    # Test 2: Checking more outputs than features
    def test2():
        np.random.seed = 2002 # reproduce
        X = np.random.rand(6)
        W = np.random.rand( 10, 6)
        b = np.random.rand(10)
        expected = my_perceptron_forward(X, W, b)
        try:
            answ = perceptron_forward(X, W, b)
        except Exception:
            answ = np.zeros(10)
        if np.allclose(expected, answ):
            print("✅ Test Case 2: 6 features, ten neurons ... passed")
        else:
            print("❌ Test 2 Failed")
            msg = (
                "\nThe forward result is not the expected,"
                " when creating 10 neurons for six inputs the value was not the expected."
                "   - The order of matrix multiplication.\n"
                "   - Is the bias term present?\n"
                "   - The activation function is correctly applied.\n"
                "    - Did you use the np.dot or np.matmul to compute the matrix product correctly?\n"
            )
            print(msg)

    # Test 3: Test that it work with any activation function.
    def test3():
        np.random.seed = 2021 # reproduce
        X = np.random.rand(6)
        W = np.random.rand( 10, 6)
        b = np.random.rand(10)

        g = lambda x: np.maximum(x, 0.0)

        expected = my_perceptron_forward(X, W, b, activation=g)
        try:
            answ = perceptron_forward(X, W, b, activation=g)
        except Exception:
            answ = np.zeros(10)
        if np.allclose(expected, answ):
            print("✅ Test Case 3: custom activation function ... passed")
        else:
            print("❌ Test 3 Failed")
            msg = (
                "\nThe forward result is not the expected,"
                "when using an activation function diferent to the default value."
                "You might want to check the following:\n"
                "   - Are you using the parameter sent to the function 'activation' and not the default value 'sigmoid'."
            )
            print(msg)
        
    # Test 4: Boundary cases single input, single output
    def test4():
        np.random.seed = 2000
        X = np.random.rand(1)
        W = np.random.rand(1,1)
        b = np.random.rand(1)

        expected = my_perceptron_forward(X, W, b, activation=sigmoid)
        try:
            answ = perceptron_forward(X, W, b, activation=sigmoid)
        except Exception as e:
            traceback.print_exc()
            answ = np.zeros(10)
        if np.allclose(expected, answ):
            print("✅ Test Case 4: Single input, single output ... passed.")
        else:
            print("❌ Test 4 Failed")
            msg = (
                "\nThe forward result is not the expected,"
                "when using a single input (X.shape=(1,)), and a single neuron (W.shape=(1,1))."
                "you might want to check the following:\n"
                "   - The order of matrix multiplication.\n"
                "   - Is the bias term present?\n"
                "   - The activation function is correctly applied.\n"
                "   - Did you use the np.dot or np.matmul to compute the matrix product correctly?\n"
            )
            print(msg)

    # Test 5: Zero weights, zero bias
    def test5():
        np.random.seed = 2000
        X = np.random.rand(10)
        W = np.random.zero(1,10)
        b = np.random.zero(1)

        expected = my_perceptron_forward(X, W, b, activation=sigmoid)
        try:
            answ = perceptron_forward(X, W, b, activation=sigmoid)
        except Exception as e:
            traceback.print_exc()
            answ = np.zeros(10)
        if np.allclose(expected, answ):
            print("✅ Test Case 5: zero weights, zero bias ... passed")
        else:
            print("❌ Test 5 Failed")
            msg = (
                "\nThe forward result is not the expected,"
                "when using weights and bias zero. "
                "You might want to check the following:\n"
                "   - The order of matrix multiplication.\n"
                "   - Is the bias term present?\n"
                "   - The activation function is correctly applied.\n"
                "   - Did you use the np.dot or np.matmul to compute the matrix product correctly?\n"
            )
            print(msg)

    # Test -- regsiter that tests are passed.
    test1()
    test2()
    test3()
    test4()
    test5()
    # End test