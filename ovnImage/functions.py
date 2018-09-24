import os
from itertools import combinations


def potencia(input, min_n_items=1, max_n_items=None):
    """Calcula y devuelve el conjunto potencia del
       conjunto input.
    """
    if max_n_items is None:
        max_n_items = len(input) + 1

    out = []
    for i in range(min_n_items, max_n_items):

        out += list(combinations(input, i))

    return out


"""
    FILE SYSTEM FUNCTIONS
"""


def check_dir(path):
    """
    Check if the specified path exists into file system. If not exists, it is created
    :param path: String
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
