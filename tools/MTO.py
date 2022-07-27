#!/usr/bin/python3
from audioop import reverse
from distutils.file_util import write_file
import logging
import numpy as np
import os
import sys
import pickle
import pulp
import random
import argparse
from collections import Counter, namedtuple
from multiprocessing import Queue, Pool

from sqlalchemy import false, true, union
from sympy import Sum
from tools.tree_join import find_shorter_paths, score_tree, nodeGenerator, connected_components
from tools.report import write_multiroute_report, plot_routes, plot_routes_order
from tools.optim_colony import AntColony
from tqdm import tqdm
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpInteger
from tools.SA import SimAnn
from tools.Tabu import Tabu
from random import randint
from random import choice
from math import exp
from math import log
from time import time

def get_buyables(molecule):
    queue = [molecule]
    while len(queue) > 0:
        curr = queue.pop()
        if '>>' not in curr['smiles']:
            yield curr['smiles']
        if len(curr['children']) > 0:
            for child in curr['children']:
                queue.append(child)

def get_intermediates(molecule):
    #returns a list of all intermediates of the molecule
    queue = [molecule]
    while len(queue) > 0:
        curr = queue.pop()
        if len(curr['children']) > 0:
            for child in curr['children']:
                queue.append(child)
            if '>>' not in curr['smiles']:
                yield curr['smiles']

class MTO:
    def __init__(self, timelimit = 100, num_mol_smallTree = 25):
        self.n = 0
        self.m = 0
        self.optimal = False
        self.best_path = {}
        self.num_mol_smallTree = num_mol_smallTree
        self.timelimit = timelimit

    def add_sets(self, trees = None, sets = None):
        """ Sets the specific parameters of a problem. In our case, that means
        a list of tree sets.

        Parameters
        ----------
        trees : list(list(dict))
            A list of tree sets. The i-th element of the set contains all
            possible trees for the i-th molecule. Every tree is represented as
            a nested dictionary.
        """
        self.trees = trees
        if sets == None:
            '''
            # collects all ways to build intermediates from all routes
            ways_to_interms = {}
            stable = False
            while not stable:
                stable = True
                for target in self.trees:
                    for t in target:
                        for node in nodeGenerator(t):
                            try:
                                if node not in ways_to_interms[node['smiles']]:
                                    ways_to_interms[node['smiles']].update(node)
                                    stable = False
                            except:
                                ways_to_interms[node['smiles']]={node}
                                stable = False
            # Build the Trees again from the scretch
            new_trees=[]
            # Tuple to keep track of where on a tree we still have to look for options
            workplace = namedtuple('workplace', ['elem','dir'])
            for target in self.trees:
                new_trees.append([])
                for t in target:
                    # Look for each route, if there are different ways to build it. 
                    route_options = ways_to_interms.copy()
                    start = workplace(t, [])
                    queue = [start]
                    # still need way to acces the dir-path in the tree...
                    while len(queue) > 0:
                        curr, queue = queue[0], queue[1:]
                        for i in range(len(curr.elem(curr.dir)['children'])):
                            for option in route_options[curr.elem['children'][i]['smiles']]:
                                curr.elem['children'][i] = route_options[curr.elem['children'][i]['smiles']][option]
                                curr.dir.append[i]
                                queue.append(workplace(curr.elem,curr.dir))

                        if len(route_options[node['smiles']]) > 0:
                            node['children'] = route_options[node['smiles']].pop(0)['children']
                            if len(route_options[node['smiles']])>0:
                                queue.append()
            '''
            I = []
            for molecule in self.trees:
                new_row = []
                for idx, route in enumerate(molecule):
                    intermediates = set(get_intermediates(route))
                    new_row.append(intermediates)
                I.append(new_row)
            self.problem = I
        else:
            self.problem = sets
        self.n = len(self.problem)
        self.m = max(map(len, self.problem))

        print("loaded problem. Dimension {} x {}".format(self.n, self.m))

    def statistics(self):
        routes = 0
        acc_route_length = 0
        intermediates = Counter()
        max_route_length = 0
        for d in range(len(self.problem)):
            routes += len(self.problem[d])
            for r in range(len(self.problem[d])):
                intermediates.update(self.problem[d][r])
                route_length = len(self.problem[d][r])
                if route_length > max_route_length:
                    max_route_length = route_length
                acc_route_length += route_length
        average_route_length = acc_route_length/routes     

        print("targets: {:5}\n".format(self.n))
        print("routes: {:5}\n".format(routes)) 
        print("max. routes/target: {:5}\n".format(self.m))
        print("avrg. routes/target: {:5}\n".format(routes/self.n))
        print("max. routelength: {:5}\n".format(max_route_length))
        print("avrg. routelength: {:5}\n".format(average_route_length))
        print("different intermediates: {:5}\n".format(len(intermediates)))
        print("max. appearance: {:5}\n".format(max(map(lambda x:intermediates[x],intermediates)))) 
        print("avrg. appearance:{:5}".format(sum(map(lambda x:intermediates[x],intermediates))/len(intermediates)))              

    def run_IP(self,startsol = None, timelimit = 100, num_mol_smallTree=25):
        """ Attempts to solve the exact IP-problem within the time limit

        Parameters
        ----------
        startsol : dict
            Feasable solution to warmstart the solver with.

        timelimit : int
            Timelimit in seconds for the solver

        num_mol_smallTree : int
            number of Targets that should be selected

        Returns
        -------
        dict
            best solution found {target_id:route_id}
        """
        # Set of all individual intermediates
        all_buyables = set()
        # list of sets. Intermediates used for the d-th target
        buyables_target = []

        # I[d][r] is the set of intermediates for molecule d and route r
        I = self.problem
        for molecule in I:
            new_buyables = set()
            for route in molecule:
                all_buyables.update(route)
                new_buyables.update(route)
            buyables_target.append(new_buyables)
        all_buyables = list(all_buyables)
            
        # Create a binary variable that indicates whether a specific intermediate is used
        y = LpVariable.dicts("buyable", all_buyables, lowBound=0, upBound=1, cat = 'Binary')
        # Auxilary variable
        # e = LpVariable.dicts("dummy_variable", all_buyables, lowBound=0, upBound=1, cat = 'Binary')
        # Create a binary variable that indicates whether molecule d uses route r
        x = LpVariable.dicts("molecule_uses_route", (list(range(self.n)), list(range(self.m))), lowBound=0, upBound=1, cat='Binary')
        # variable to contain Product of x and y
        '''
        p = LpVariable.dicts("product_x_and_y", (list(range(self.n)), list(range(self.m)), all_buyables), lowBound=0, upBound=1, cat='Binary')
        p_var = {}
        for d in range(self.n):
            for r in range(len(I[d])):
                for buy in I[d][r]:
                    p_var[(d,r,buy)] = pulp.LpVariable("p_({},{})x{}".format(d,r,buy),lowBound=0, upBound=1, cat='Binary')
        '''
        # Create a binary variable that indacate whether a target is selected
        z = LpVariable.dicts("target", list(range(self.n)), lowBound=0, upBound=1, cat = 'Binary')
        # Create the problem
        model = LpProblem(name="Minimize intermediates", sense=LpMinimize)
        # Set the objective function: minimize the number of buyables
        model += lpSum(y[buy] for buy in all_buyables)
        print("reading constraints...")
        # First constraint: each molecule uses z[d] of its routes
        # x[d][r] = molecule d uses route r
        # list of dict{buyable:list of routes} that indicates which routes per target contain a specific buyable
        targets_buyables = []
        for d in range(self.n):
            model += (lpSum([x[d][r] for r in range(len(I[d]))]) == z[d], "Molecule_{}_uses_{}_recipe".format(d,z[d]))
            model += (lpSum([x[d][r] for r in range(len(I[d]))]) <= 1, "Molecule_{}_uses_one_recipe".format(d))
            # Second constraint: forces all intermediates to be used in their respective route and molecule           
            # routes in which a specific intermediate is used
            routes_buyables = {x:[] for x in all_buyables}
            for r in range(len(I[d])):
                for buy in I[d][r]:
                    routes_buyables[buy].append(r)
                    # model += (y[buy] >= x[d][r], "forces_all_buyables_{}-{}-{}".format(d,r, buy))
                    '''
                    # test linearization of constraint x[d][r] * y[h] == z[i]
                    model += p_var[(d,r,buy)] <= x[d][r]
                    model += p_var[(d,r,buy)] <= y[buy]
                    model += p_var[(d,r,buy)] >= x[d][r]+y[buy]-1
                    '''
            targets_buyables.append(routes_buyables)
            for buyable in buyables_target[d]:
               if len(buyables_target[d]) > 0:
                    model += (y[buyable] >= lpSum(x[d][r] for r in routes_buyables[buyable]), "forces_all_buyables_{}-{}".format(d, buyable))
        '''
        # additional constraints: sum of total routes that used buyable is maximum k
        for buyable in all_buyables:
            model += (y[buyable]*num_mol_smallTree >= lpSum(lpSum(x[d][r] for r in targets_buyables[d][buyable]) for d in range(self.n)))
        # testing new constraints: if buy exists in the routes, then y[buy] = 1
        for buy in all_buyables:
            model += (lpSum(lpSum(x[d][r] for r in targets_buyables[d][buy]) for d in range(self.n)) <= num_mol_smallTree * e[buy], "Sum_of_{}".format(buy))
            model += (1 - num_mol_smallTree*(1-e[buy]) <= y[buy], "Sum_of_{}_2".format(buy))
        
            for r in range(len(I[d])):
                try:
                    for buyable in I[d][r]:
                        model += (x[d][r] <= y[buyable], "forces_all_buyables_{}-{}-{}".format(d, r, buyable))
                except IndexError:
                    model += (x[d][r] == 0, "out_of_bounds_buyable_{}-{}".format(d, r))
                    x[d][r].setInitialValue(0)
                    x[d][r].fixValue()
        '''
        # Third constraint: Only use num_mol_smallTree Targets in total
        model += (lpSum(z[d] for d in range(self.n)) == num_mol_smallTree, "forces_certain_number_of_targets")
        model += (lpSum(lpSum(x[d][r] for r in range(len(I[d]))) for d in range(self.n)) == num_mol_smallTree, "forces_certain_number_of_routes")
        # Hand over warm-start solution
        if startsol != None:
            print("load warmstart solution")
            for d in range(self.n):
                z[d].setInitialValue(0)
                for r in range(len(self.problem[d])):
                    x[d][r].setInitialValue(0)
            for buy in all_buyables:
                y[buy].setInitialValue(0)

            for target, route in startsol.items():
                x[target][route].setInitialValue(1)
                z[target].setInitialValue(1)
                for interm in I[target][route]:
                    y[interm].setInitialValue(1)

            # print start value
            value = 0
            for mol in all_buyables:
                value += y[mol].varValue
            print("Start value:", value)

        # Solve it
        solution = dict()
        solver = pulp.getSolver('PULP_CBC_CMD', timeLimit=timelimit, msg=True, warmStart = True, keepFiles=True, \
            # options = ["allow 0.99"]  "dualB??","dualT??","primalT","primalW??","allow??","cuto??","inc??","inf??","integerT??","preT??","ratio??","sec??",\
            # "cuto??", "inc 1", "inf??", "preT??", "cpp??", "force??", "idiot??", "maxF??",\
            #      "cutD??","log??","maxN??","maxS??","passC??","passF??","passT??","pumpT??","strong??","trust??",\
            #         "chol??","crash??","cross??","direction??","dualP??","error??", "keepN??", \
            #             "mess?","perturb??", "presolve??","primalP??","printi??", "scal??",\
            #                 "clique??", "combine??", "cost??","cuts??", "Dins??","DivingS??", \
            #                     "Diving??","feas??", "flow", "gomory??", "greedy??", "heur??", "knapsack??", "lift??",\
            #                         "local??", "mixed??", "node??", "preprocess??", "probing??", "reduce??",\
            #                             "residual??", "Rens??", "Rins??", "round??", "sos??", "two??", \
            #                                 "branch??","doH??","miplib??","prio??","solv??","strengthen??", "userCbc??"])
        )

        
        status = model.solve(solver)
        self.optimal = status

        target_value = 0
        for var in model.variables():
            if var.name[0] == 'm' and var.value() > 0:
                # print("{}: {}".format(var.name, var.value()))
                molecule_route = var.name.split('_')
                mol = int(molecule_route[-2])
                route = int(molecule_route[-1])
                solution[mol] = route
            elif var.name[0] == 'b' and var.value() > 0:
                target_value += var.value()
        print("union:", target_value)
        
        self.best_path = solution
        return solution, status

    def simAnn(self,start = None, timelimit = 200):        
        if start == None:
            start=self.simple_greedy(25)
        print("Start Annealing...")
        annealing = SimAnn(self.costFunc, self.problem, start, timelimit=timelimit, greedy =True, cooling_schedule='quadratic')
        annealing.results()
        return annealing.best_state

    def tabuSearch(self, timelimit = 200, start = None):        
        if start == None:
            start=self.greedy(25)
        print("Start Tabu Search ...")
        search = Tabu(self.costFunc, self.problem, start, timelimit=timelimit, greedy = False)
        search.results()
        return search.best_state

    def costFunc(self,path):
        ''' calculates the number of intermedates for a given selection of trees.

        Parameters
        ----------
        path: dict
            selection of trees
            {used target:selected route for that target}

        Returns
        -------
        int
            number of intermediates
        '''
        intermediates = set()
        for target,molecule in path.items():
            intermediates.update(self.problem[target][molecule])
        return len(intermediates)

    def heuristic(self):
        '''Returns a shortest route for all targets
        
        Returns
        -------
        dict
            routes for all targets {target_id:route}
        
        '''
        path = {}
        Depth = namedtuple('len',['Depth', 'route_id'])
        for target_id,target in enumerate(self.problem):
            min_len = Depth(1000000,0)
            for route_id, route in enumerate(target):
                if len(route) < min_len.Depth:
                    min_len = Depth(len(route),route_id)
            path.update({target_id:min_len.route_id})
        return path

    def simple_greedy(self,numMol):
        retval = {}
        union = set()
        pending  = list(range(self.n))
        # wollen einfach 25 mal die Menge hinzufügen, die am wenigsten Elemente hinzufügt.
        # Vllt. als vorläufer zum IP-solver???
        for i in range(numMol):
            # keeping track of best so far [score, target, route]
            smallest_union = [1000,0,0]
            for target in pending:
                for route in range(len(self.problem[target])):
                    new_union = union.union(self.problem[target][route])
                    if len(new_union) < smallest_union[0]:
                        smallest_union = [len(new_union), target, route]
            pending.remove(smallest_union[1])
            union.update(self.problem[smallest_union[1]][smallest_union[2]])
            retval[smallest_union[1]] = smallest_union[2]
        return retval

    def greedy(self, numMol):
        '''
        Gives a greedy solution for numMol targets

        Returns
        -------
        dict
            greedy routes {target:route}
        '''
        path = self.greedy_routes()
        print("Greedy:", self.costFunc(path))
        # select the best numMol Tarets greedy
        path = self.select_targets_greedy(path, numMol)
        print("Greedy:", self.costFunc(path))
        # select best routes for selected targets
        path = self.greedy_routes(path)
        print("Greedy:", self.costFunc(path))
        return dict(sorted(path.items()))

    def greedy_routes(self, path = None):
        '''
        selects greedy routes for the Targets mentioned in path
        
        Parameters
        -------
        path: dict
            contains the targets
        
        Returns
        -------
        dict
            greedy routes {target:route}
        '''
        Union = namedtuple('union',['route_id','len_union','union_total'])
        runs = -1
        if path == None:
            keys = range(len(self.problem))
            path = {}
        else:
            keys = dict(sorted(path.items())).keys()
        # going through all till you won't get better 
        upgrade = True
        while upgrade:
            upgrade = False
            runs += 1
            # for id in reversed(sorted(path.keys())):
            for id in keys:
                molecule = self.problem[id]
                try:
                    shortest2 = Union(path[id], self.costFunc(path), set())
                except:
                    shortest2 = Union(None,1000000, set())
                for idx, mol in enumerate(molecule):
                    path[id] = idx
                    len_now = self.costFunc(path)
                    if  len_now < shortest2.len_union:
                        shortest2 = Union(idx, len_now, set())
                        upgrade = True
                                       
                path[id] = shortest2.route_id
                #if upgrade: break
        # print("Needed {} runs".format(runs))
        return dict(sorted(path.items()))


    @staticmethod
    def get_top_N_mol(all_routes_list, ordered_routes, num_mol=25):
        """Gets the top N molecules regarding the synthesis order and returns their
        indices in the full list as list

        Args:
            all_routes_list (list): All molecules
            ordered_routes (list): Synthesis order top (least effort) to bottom (maximum effort)
            num_mol (int, optional): Number of molecules to retrieve. Defaults to 25.

        Returns:
            list: list of indices in the full molecule list
        """
        indices = []
        for oo in range(num_mol):
            for rr in range(len(all_routes_list)):
                if all_routes_list[rr]["smiles"] == next(iter(ordered_routes[oo][0])):
                    indices.append(rr)
        return indices

    @staticmethod
    def route_to_intermediates(route, keep_reactions=True):
        """ Given a route (written as a nested dictionary), returns a set of all
        its intermediates.

        Parameters
        ----------
        route : dict
            Nested dictionary of intermediates representing a route.
        keep_reactions : bool
            If True, reactions will also be returned as intermediates. Otherwise,
            only true intermediates are returned.

        Returns
        -------
        set
            A set of all the intermediates in this route.
        """
        intermediates = set()

        pending = [route]
        while len(pending) > 0:
            current, pending = pending[0], pending[1:]
            if keep_reactions or '>>' not in current['smiles'] and (len(current['children']) > 0):
                intermediates.add(current['smiles'])
            for child in current['children']:
                pending.append(child)
        return intermediates

    @staticmethod
    def route_to_buyables(route):
        """ Given a route (written as a nested dictionary), returns a set of all
        its buyables.

        Parameters
        ----------
        route : dict
            Nested dictionary of buyables representing a route.

        Returns
        -------
        set
            A set of all the buyables in this route.
        """
        buyables = set()

        pending = [route]
        while len(pending) > 0:
            current, pending = pending[0], pending[1:]
            if ('>>' not in current['smiles']) and (len(current['children']) == 0):
                buyables.add(current['smiles'])
            for child in current['children']:
                pending.append(child)
        return buyables

    @staticmethod
    def collect_all_intermediates(routes, keep_reactions=True):
        """ Given a list of routes, it returns a set of all the intermediates
        present in any of them. Targets are counted as intermediates also.

        Parameters
        ----------
        routes : list(dict)
            List of routes.
        keep_reactions : bool
            If True, reactions will also be returned as intermediates. Otherwise,
            only true intermediates are returned.

        Returns
        -------
        set
            A set with any and all intermediates present in any of the input
            routes.
        """
        intermediates = set()
        for route in routes:
            for node in nodeGenerator(route):
                if (keep_reactions or '>>' not in node['smiles']) and len(node['children']) > 0:
                    intermediates.add(node['smiles'])
        return intermediates

    def select_targets_greedy(self, routes, n =25, weighted = True):
        """ Calculates the order in which intermediates are needed to obtain the
        highest amount of molecules with the fewer number of intermediates.

        Parameters
        ----------
        routes : dict
            A dict of routes for (all) targets. {target:molecule}

        n : int
            Number of how many targets should be seen further forward

        Returns
        -------
        list
            A dictionary with n routes, where the elements (target:route) 
            should be the best fitting together.

        Notes
        -----
        The returned dict has an extra property: it tries to build the largest
        number of elements from the least number of intermediates. This means
        that, given a big enough list of routes, it should be possible to look
        at the graph of #molecules given #intermediates and decide where it
        would be more economical to stop.
        Nevertheless the selection is a greedy step. We step by step exclude the
        route with the most intermediates. Shared intermediates counted as 1/2, 1/3, ... 
        if the other routes using them are not yet "thrown away".
        """

        # Named tuples to make the code easier to read
        Route = namedtuple('Route', 'target intermediates')
        RouteInterms = namedtuple('RouteInterms', 'target idx weight')
        # List that will contain the greedy order of all targets
        retval = []
        # In this code, a route is a set of intermediates.
        pending_molecules = []
        for target, route in routes.items():
            route_interms = self.problem[target][route]
            pending_molecules.append(Route(target, route_interms))

        iter = 50
        # Start of the code that chooses at every step which route should be
        # created next. We attach a progress bar to it.
        pbar = tqdm(desc="Calculating synthesis order", total=len(pending_molecules))
        while len(pending_molecules) > 0:
            iter -= 1
            if iter == 0:
                iter = 50
            # Identify all pending intermediates
            all_interms = Counter()
            loop_routes = []
            # select greedy-routes again every 100 rounds 
            if iter == 50 and len(pending_molecules)>=n and False:
                new_routes = {}
                for r in pending_molecules:
                    new_routes[r.target] = routes[r.target]
                routes = self.greedy_routes(new_routes)
                pending_molecules = []
                for target, route in routes.items():
                    route_interms = self.problem[target][route]
                    pending_molecules.append(Route(target, route_interms))
            for r in pending_molecules:
                all_interms.update(r.intermediates)
            # calculate the weight of a route
            for idx, r in enumerate(pending_molecules):
                # Lighter if many routes use the same intermediates
                if weighted:
                    weight=np.sum([1/all_interms[x] for x in r.intermediates])
                else:
                    weight = 0
                    for x in r.intermediates:
                        if all_interms[x] == 1:
                            weight += 1
                
                loop_routes.append(RouteInterms(r.target, idx, weight))
            # Finds the route with the largest weight of intermediates.
            best_weight = -1
            for mol in loop_routes:
                if mol.weight > best_weight:
                    best_weight =mol.weight
                    chosen = mol
            # loop_routes.sort(key=lambda x: x.weight, reverse=True)
            # chosen = loop_routes[0]
            # Remove the selected route from the "pending" list
            pending_molecules = pending_molecules[:chosen.idx] + pending_molecules[chosen.idx + 1:]
            retval.append(chosen.target)
            # Update the progress bar
            pbar.update(1)
        pbar.close()

        # The list has to be read back-to-front. Therefore, we reverse the list
        retval.reverse()
        path = {}
        for i in range(n):
            path.update({retval[i]:routes[retval[i]]})
        return dict(sorted(path.items()))

    def find_shorter_path(self, path):
        final_routes = []
        for route in path.keys():  
            final_routes.append(self.trees[route][path[route]])
        
        final_routes_score = score_tree({'smiles':'START>>TARGETS', 'children': final_routes}, 'num_unique_interm')
        final_routes_test = find_shorter_paths(final_routes)
        final_routes_test_score = score_tree({'smiles':'START>>TARGETS', 'children': final_routes_test}, 'num_unique_interm')
        if final_routes_test_score < final_routes_score:
            final_routes = final_routes_test
            final_routes_score = final_routes_test_score
            print("Found shorter routes with value:", final_routes_score)
        return final_routes

    def write_file(self, final_routes, outdir):
        final_synth_order = AntColony.get_right_order(final_routes)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        write_multiroute_report(final_routes, outdir, final_synth_order)
        plot_routes(final_routes, os.path.join(outdir, "full_graph"))
        plot_routes_order(final_synth_order, os.path.join(outdir, "synthesis_order.png"))

        print("See results in ",outdir)


if __name__ == '__main__':
    # route_dir = 'C:/Users/GNQBG/OneDrive - Bayer/Personal Data/AI4Synthesis_service_calculate_routes/routes'
    # route_dir = '/tmp/newroutes/V8/01'
    parser = argparse.ArgumentParser()
    parser.add_argument('routedir', type=str)
    parser.add_argument('number_smallTree', type=int)
    parser.add_argument('timelimit_seconds',type=int)
    args = parser.parse_args()
    route_dir = args.routedir
    num_mol_smallTree = args.number_smallTree
    timelimit = args.timelimit_seconds

    all_files = list(os.listdir(route_dir))
    all_files.sort()
    problem = []
    problem_interms = []
    seen_targets = []
    try:
        with open(os.path.join(route_dir, "trees.pkl"), 'rb') as fp:
            trees = pickle.load(fp)
            sets = None

    except:
        with open (os.path.join(route_dir, "sets.pkl"), 'rb') as fp:
            sets = pickle.load(fp)
            trees = None
    '''
    This is for firstly determine the disconnected components

                # Calculate the set of intermediates
                all_interms = set()
                for route in routes:
                    for node in nodeGenerator(route):
                        all_interms.add(node['smiles'])
                problem_interms.append(all_interms)
    
    # Calculate how many disconnected components are there
    all_components = connected_components(problem_interms)
    print("{} component(s) found: {}".format(len(all_components), all_components))

    for component_id, component in enumerate(all_components):
        # Every set of components gets its own report
       
        smaller_problem = list(map(lambda x: problem[x], component))
'''
    solutions = []
    # For development this is set to False. For production,
    # this should be set to True

    aco = MTO()
    aco.add_sets(trees=trees,sets=sets)
    start = time()
    path = aco.simple_greedy(25)

    path, optimal = aco.run_IP(timelimit=timelimit/2, startsol= path, num_mol_smallTree=num_mol_smallTree)
    if not optimal:
        path =aco.simAnn(path,timelimit=timelimit/2)
    end = time()
    print("took {} Seconds to get this solution of {}".format(end-start, aco.costFunc(path)))

    final_routes = aco.find_shorter_path(path)

    outdir = "C:\\Temp\\{}\\{}".format(route_dir.split("\\")[-1],component_id)
    os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz\\bin"
    aco.write_file(path,outdir)  

