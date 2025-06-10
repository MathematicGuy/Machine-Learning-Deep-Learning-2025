"""Calculating Activation Functions like ReLU, Softmax, Sigmoid, ELU
"""

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


evaluate_f1_components(tp=2, fp=3, fn=4.4)
