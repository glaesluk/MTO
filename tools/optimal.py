import numpy as np
import os
import sys
import pickle
from tools.tree_join import score_tree
from itertools import combinations
from tqdm import tqdm
import math

def optimal(final_routes, n = 10):
    if input("That could last very long. continue?") == "y":
        global minium
        minimum = 1000000
        pbar = tqdm(desc="Calculating synthesis order", total=math.comb(len(final_routes), n))
        for subset in combinations(final_routes, n):
            score = score_tree({'smiles':'START>>TARGETS', 'children': subset}, 'num_unique_interm')
            if score < minimum:
                minimum = score
            pbar.update(1)
        pbar.close()
        return minimum
    else:
        return None

if __name__ == '__main__':
    print("This file should not be executed as main")