import torch
from collections.abc import Iterable


def _ntuple(n):
    '''Function for handling scalar and iterable layer arguments'''

    def parse(x):
        '''Closure for parsing layer args'''
        if isinstance(x, Iterable):
            return x
        return tuple([x for i in range(n)])

    return parse


# Typedef
_pair = _ntuple(2)