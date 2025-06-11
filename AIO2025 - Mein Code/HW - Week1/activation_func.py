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

def evaluate_f1_components(tp, fp, fn):
    """Evaluate f_1 score using tp, fp and fn
    Args:
        tp (int): true positive
        fp (int): false positive
        fn (int): false negative
    Returns:
        tp, fp, fn
    """

    if not is_int(tp, fp, fn):
        #? return None if tp or fp or fn is not INT
        return None
    if tp < 0 or fp < 0 or fn < 0:
        #? Make sure all values is greater or equal to zero
        print('tp and fp and fn must be greater than or equal to zero')
        return None

    try:
        precision = tp/(tp+fp)
        recall = tp/tp+fn
        f1_score = 2 * (precision*recall)/(precision + recall)

        print(f'precision is {precision}')
        print(f'recall is {recall}')
        print(f'f1_score is {f1_score}')

        return precision, recall, f1_score

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


def elu(x, alpha=0.05):
    if x <= 0:
        return alpha*(math.e**x - 1)
    else:
        return x

activations = {
    'sigmoid': sigmoid,
    'relu': relu,
    'elu': elu,
}

def interactive_activation_function(x=5, input_activation='sigmoid'):
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

def factorial(div):
    return factorial(div - 1)


def sin(x, n):
    x = np.array(x)
    div = (2*n + 1)
    result = 0
    for i in range(n):
        result += ((-1)**n) * (x**(2*n  + 1)) / factorial(div)


def cos(x, n):
    pass

def sinh(x, n):
    pass

def cosh(x, n):
    pass

evaluation_func = {
    'sin': sin,
    'cos': mse,
    'sinh': rmse,
    'cosh': cosh
}

if __name__ == '__main__':
    # evaluate_f1_components(tp=2, fp=3, fn=4.4)
    # interactive_activation_function()
    # interactive_loss_function()
    x = 5
    n = 10
    eval_func = evaluation_func.get('sin')
    if eval_func:
        result = eval_func(x, n)
        print(result)
    else:
        raise ValueError("Invalid Operation")