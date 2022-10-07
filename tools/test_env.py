#!/usr/bin/python3
from audioop import reverse
import logging
import numpy as np
import os
import sys
import pickle
import pulp
import random
import argparse
from collections import Counter, namedtuple
from time import time 
from threading import Thread, Event

from tools.tree_join import find_shorter_paths, score_tree, nodeGenerator, connected_components
from tqdm import tqdm
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpInteger
from tools.SA import SimAnn
from tools.Tabu import Tabu
from tools.MTO import MTO
from random import randint
from random import choice


if __name__ == '__main__':
    # route_dir = 'C:/Users/GNQBG/OneDrive - Bayer/Personal Data/AI4Synthesis_service_calculate_routes/routes'
    # route_dir = '/tmp/newroutes/V8/01'
    parser = argparse.ArgumentParser()
    parser.add_argument('testdir', type=str)
    parser.add_argument('number_smallTree', type=int)
    parser.add_argument('timelimit_seconds',type=int)
    args = parser.parse_args()
    test_dir = args.testdir
    num_mol_smallTree = args.number_smallTree
    timelimit = args.timelimit_seconds

    all_files = list(os.listdir(test_dir))
    all_files.sort()

    solutions = {}
    times = {}

    for dir in all_files:

        route_dir = os.path.join(test_dir, dir)
        try:
            with open(os.path.join(route_dir, "trees.pkl"), 'rb') as fp:
                trees = pickle.load(fp)
                sets = None
        except:
            with open (os.path.join(route_dir, "sets.pkl"), 'rb') as fp:
                sets = pickle.load(fp)
                trees = None

        print("\n Solving {} now: \n".format(dir))
        optimal = False
        aco = MTO(timelimit = timelimit, num_mol_smallTree = num_mol_smallTree)
        aco.add_sets(trees=trees,sets=sets)
       
        start = time()

        print("Greedy: ---------------------------")
        path = aco.simple_greedy(num_mol_smallTree)
        aco.best_path = path
        solutions[dir] = [aco.costFunc(path)]
        print("Greedy:", aco.costFunc(path))

        time1 = time()

        print("IP-Solver:-----------------------------")
        # path, optimal = aco.run_IP(timelimit=aco.timelimit/2, startsol=path, num_mol_smallTree=aco.num_mol_smallTree)
        
        # path = aco.simAnn(start=path, timelimit=timelimit)
        solutions[dir].append(aco.costFunc(path))
        time3 = time()
        
        print("Simm. Ann:-----------------------------")
        # path, optimal = aco.run_IP(timelimit=aco.timelimit/2, num_mol_smallTree=aco.num_mol_smallTree)
        # path = aco.greedy_routes()
        # path = aco.select_targets_greedy(path,num_mol_smallTree, weighted=False)
        # solutions[dir].append(aco.costFunc(path))
     
        if not aco.optimal:
            print("simulated Annealing: ---------------------------")
            path = aco.simAnn(start=path, timelimit=timelimit/2)
            solutions[dir].append(aco.costFunc(path))
        
        # final_routes = aco.find_shorter_path(path)
        time4 = time()
        times[dir] = ['%.0f'%(time1-start), '%.0f'%(time3-time1), '%.0f'%(time4-time3)]
        print("---------------------------------------------")


    for set in solutions.keys():
        print(set)
        print("Routen: {:<30} Score: {:<30} Time: {}".format(set, str(solutions[set]),str(times[set])))
        print("---------------------------------------------")
       
