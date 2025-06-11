"""Calculating Activation Functions like ReLU, Softmax, Sigmoid, ELU
"""

import math
import random
import numpy as np

def is_int(*args):
    """Check if any value in *args is int, if not return False flag
    Returns:
        Boolen: True or False
    """
    name = ['tp', 'fp', 'fn']
    #* Can use assert to check for error but not recommend, TypeError is more specific
    for i, arg in enumerate(args):
        if not isinstance(arg, int):
            print(f"{name[i]} must be int")
            return False

    return True

def calc_f1_score(tp, fp, fn):
    """Evaluate f_1 score using tp, fp and fn
    Args:
        tp (int): true positive
        fp (int): false positive
        fn (int): false negative
    Returns:
        f1_score
    """

    if not is_int(tp, fp, fn):
        #? return None if tp or fp or fn is not INT
        return None

    if tp < 0 or fp < 0 or fn < 0:
        #? Make sure all values is greater or equal to zero
        print('tp and fp and fn must be greater than or equal to zero')
        return None

    try:
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1_score = 2 * (precision*recall)/(precision + recall)

        # print(f'precision is {precision}')
        # print(f'recall is {recall}')
        # print(f'f1_score is {f1_score}')

        # return precision, recall, f1_score
        return f1_score

    except ZeroDivisionError:
        print("Error: cannot perform division if (tp+fp) or (tp+fn) is zero")
        return None


def sigmoid(x):
    return 1 / (1+math.e**(-x))

def relu(x):
    #? Add this to global func later
    if x <= 0:
        return 0
    else:
        return x


def elu(x, alpha=0.01):
    if x <= 0:
        return alpha*(math.e**x - 1)
    elif x > 0:
        return x

activations = {
    'sigmoid': sigmoid,
    'relu': relu,
    'elu': elu,
}

def interactive_activation_function(x=3, input_activation='sigmoid'):
    input_activation = input(str('Input activation Function (sigmoid|relu|elu):'))

    if not isinstance(x, (int, float)):
        return "Warning, x must be a number, not anything else"

    activation = activations.get(input_activation)

    if activation: # if activation exist
        print(f'Input x = {x}')
        print(f'{input_activation}: f({x}) = {activation(x)}')
    elif not activation:
        print(f'{input_activation} is not supported')
    else:
        raise ValueError("Invalid Operation")


def mae(y, y_hat, n):
    y = np.array(y)
    y_hat = np.array(y_hat)

    print('y_hat:', y_hat)

    return np.sum(np.abs(y - y_hat)) / n


def mse(y, y_hat, n):
    y = np.array(y)
    y_hat = np.array(y_hat)

    print('y:', y)
    print('y_hat:', y_hat)

    return sum((y - y_hat)**2) / n


def rmse(y, y_hat, n):
    y = np.array(y)
    y_hat = np.array(y_hat)
    print('y_hat:', y_hat)

    return math.sqrt(sum((y - y_hat)**2) / n)


loss_functions = {
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
}

def interactive_loss_function(input_loss_func='mse'):
    """calculate user input loss function

    Args:
        input_loss_func (str, optional): loss function name. Defaults to 'mse'.

    Raises:
        ValueError: if num_sample is not integer

    Returns:
        loss_name: loss function name
        sample: sample_i
        predict predicted_value_i
        target: target_value_i
        loss: loss_i return from loss_function
    """

    num_samples_str = input("Input number of samples (integer number) which are generated:")

    if not num_samples_str.isnumeric():
        return "number of samples must be an integer number"

    num_samples = int(num_samples_str)
    input_loss_func = input('Enter loss function (mae|mse|rmse):')

    #? samples = (target_i, predict_i) - [y, y_hat]
    y = [random.uniform(0, 10) for _ in range(num_samples)]
    y_pred = [random.uniform(0, 10) for _ in range(num_samples)]
    n = len(y)


    loss_func = loss_functions.get(input_loss_func)

    if loss_func: # if loss_func exist
        loss_result = loss_func(y, y_pred, n)
        # print(f'resulkt: {loss_result}')

        #? Print result
        for i in range(num_samples):
            print(f'loss_name: {input_loss_func.upper()}, sample: {i}, pred: {y_pred[i]},'
                    f' target: {y[i]}')

        print(f'Final {input_loss_func.upper()}: {np.sum(loss_result)}')

    elif not loss_func:
        print(f'{input_loss_func} is not supported')
    else:
        raise ValueError("Invalid Operation")

def calc_factorial(n):
    """calculating 3! = 3 * (3-1) * (3-2), stop if (3-2 <= 1 or n-2 <= 1)

    Args:
        n (int): factorial number
    """
    if n <= 1: #? stop when n reached 1, as 1 is the smallest num in the fatorial
        return n

    return n * calc_factorial(n - 1)


def approx_sin(x, n):
    """
        Approximate the sine of x using the Taylor series expansion.

        Parameters:
        x (float): The input angle in radians.
        n (int): Number of terms in the Taylor series expansion.

        Returns:
            float: Approximate value of sin(x) using n+1 terms.
    """

    result = 0
    for i in range(0, n+1): # 0 to 3 (yes 4 numbers)

        numeratorrr = ((-1) ** i) * (x ** (2 * i + 1))
        denominator = calc_factorial(2 * i + 1)

        result += (numeratorrr / denominator)

    return result

def approx_cos(x, n):
    result = 1
    for i in range(1, n+1):
        numeratorrr = ((-1)**i) * (x**(2*i))
        denominator = calc_factorial(2*i)

        result += (numeratorrr / denominator)

    return result

def approx_sinh(x, n):
    result = 0
    for i in range(0, n+1): # 0 to 3 (yes 4 numbers)

        numeratorrr = (x ** (2 * i + 1))
        denominator = calc_factorial(2 * i + 1)

        result += (numeratorrr / denominator)

    return result

def approx_cosh(x, n):
    result = 1
    for i in range(1, n+1): # 0 to 3 (yes 4 numbers)

        numeratorrr = x ** (2 * i)
        denominator = calc_factorial(2 * i)

        result += (numeratorrr / denominator)

    return result

evaluation_func = {
    'sin': approx_sin,
    'cos': approx_cos,
    'sinh': approx_sinh,
    'cosh': approx_cosh,
}


if __name__ == '__main__':
    assert round(calc_f1_score(tp=2, fp=3, fn=5), 2) == 0.33
    print(round(calc_f1_score(tp=2, fp=4, fn=5), 2))

    # print(calc_f1_score(tp=2, fp=3, fn=5))

    # interactive_activation_function()
    # assert round(relu(1)) == 1
    # print(round(sigmoid(3), 2))


    # interactive_loss_function()


    # x = 3.14  # 90 degrees
    # n = 10

    # eval_func = evaluation_func.get('cosh')
    # if eval_func:
    #     result = eval_func(x, n)
    #     print(round(result, 2))
    # else:
    #     raise ValueError("Invalid Operation")